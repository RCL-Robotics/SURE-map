"""Train FlowSigmaHead on TartanAir v1 forward optical flow.

TartanAir v1 provides forward flow only: frame t -> frame t+1.  Therefore this
script trains the sigma head on forward rigid reprojection:

  source frame s=t-1 pixels --pred depth/pose/K--> target frame t
  residual = rigid_flow(s->t) - TartanAir_GT_flow(s->t)

The input clip remains chronological.  Deployment can still consume adjacent
frames in the opposite order if needed because the head is learning matching
difficulty, but the training target here is strictly forward.

8-GPU training:
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 \
    training/tests/train_flow_sigma_tartanair.py --max_steps 20000
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

cv2.setNumThreads(0)

HERE = Path(__file__).resolve().parent
TRAIN_ROOT = HERE.parent
REPO_ROOT = TRAIN_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRAIN_ROOT))

from dust3r.datasets.tartanair_flow import TartanAirFlow
from heads.flow_nll import flow_nll_2d
from heads.flow_sigma_head import FlowSigmaHead
from utils.calibration import format_stats, reliability_stats_2d
from utils.data import collate_views, dust3r_to_model_image
from utils.flow import build_pixel_grid, decode_pose_enc_to_c2w


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        world_rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", world_rank))
        torch.cuda.set_device(local_rank)
        return local_rank, world_rank, world_size
    return 0, 0, 1


def log_setup(rank: int, outdir: str):
    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(Path(outdir) / "train.log"))
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [r{rank}] %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(str(rank))


def build_frozen_model(ckpt: str, device: torch.device, scale_frames: int, log):
    from lingbot_map.models.gct_stream import GCTStream

    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        camera_num_iterations=4,
        use_sdpa=False,
        use_gradient_checkpoint=False,
    )
    ckpt_obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_obj.get("model", ckpt_obj), strict=False)
    model.requires_grad_(False).eval().to(device)
    log.info(f"frozen LingBot-Map loaded: embed_dim={model.embed_dim}")
    return model


@torch.no_grad()
def run_frozen_forward(model, images_01: torch.Tensor, num_scale_frames: int):
    if hasattr(model, "clean_kv_cache"):
        model.clean_kv_cache()
    bsz, seq_len = images_01.shape[:2]
    del bsz
    agg, patch_start_idx = model.aggregator(
        images_01,
        selected_idx=[4, 11, 17, 23],
        num_frame_for_scale=num_scale_frames,
        sliding_window_size=-1,
        num_frame_per_block=seq_len,
    )
    pose_enc = model._predict_camera(agg, causal_inference=False)["pose_enc"]
    depth = model._predict_depth(agg, images_01, patch_start_idx)["depth"]
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    _, intrinsics = pose_encoding_to_extri_intri(
        pose_enc,
        image_size_hw=tuple(images_01.shape[-2:]),
        build_intrinsics=True,
    )
    return agg, patch_start_idx, pose_enc, depth, intrinsics


def rigid_flow_src_to_tgt(
    depth_src: torch.Tensor,
    pose_src: torch.Tensor,
    pose_tgt: torch.Tensor,
    intr_src: torch.Tensor,
    intr_tgt: torch.Tensor,
    height: int,
    width: int,
):
    """Project source-frame pixels into target frame using predicted geometry."""
    device = depth_src.device
    dtype = depth_src.dtype
    grid = build_pixel_grid(height, width, device=device, dtype=dtype)
    ray = torch.einsum("bij,hwj->bhwi", torch.linalg.inv(intr_src), grid)
    cam_src = ray * depth_src.unsqueeze(-1)

    r_src = pose_src[:, :3, :3]
    t_src = pose_src[:, :3, 3]
    world = torch.einsum("bij,bhwj->bhwi", r_src, cam_src) + t_src[:, None, None, :]

    r_tgt = pose_tgt[:, :3, :3]
    t_tgt = pose_tgt[:, :3, 3]
    cam_tgt = torch.einsum("bji,bhwj->bhwi", r_tgt, world - t_tgt[:, None, None, :])
    z_tgt = cam_tgt[..., 2]

    proj_h = torch.einsum("bij,bhwj->bhwi", intr_tgt, cam_tgt)
    proj = proj_h[..., :2] / proj_h[..., 2:3].clamp(min=1e-6)
    flow = proj - grid[..., :2].unsqueeze(0)
    mask = (
        (depth_src > 1e-3)
        & (z_tgt > 1e-3)
        & (proj[..., 0] >= 0)
        & (proj[..., 0] < width)
        & (proj[..., 1] >= 0)
        & (proj[..., 1] < height)
        & torch.isfinite(flow).all(-1)
    )
    return flow, mask


def stack_gt_flow(views, device: torch.device):
    return torch.from_numpy(np.stack([v["gt_flow"] for v in views], axis=0)).float().to(device)


def stack_gt_flow_mask(views, device: torch.device):
    return torch.from_numpy(np.stack([v["gt_flow_mask"] for v in views], axis=0)).to(device)


def unsup_sigma_floor_loss(
    log_su: torch.Tensor,
    log_sv: torch.Tensor,
    unsup_mask: torch.Tensor,
    sigma_floor_px: float,
):
    if sigma_floor_px <= 0.0 or not unsup_mask.any():
        return (log_su.sum() + log_sv.sum()) * 0.0
    log_floor = float(2.0 * np.log(sigma_floor_px))
    loss_u = torch.relu(log_floor - log_su)
    loss_v = torch.relu(log_floor - log_sv)
    m = unsup_mask.float()
    return ((loss_u + loss_v) * m).sum() / m.sum().clamp(min=1.0)


def supervise_forward_pair(
    head,
    agg,
    patch_start_idx,
    img01,
    depth,
    pose,
    intrinsics,
    obs_flow,
    flow_label,
    src: int,
    beta_nll: float,
    min_valid_ratio: float,
    unsup_sigma_floor_px: float,
    unsup_lambda: float,
):
    """Supervise source frame src -> target frame src+1 with GT forward flow."""
    bsz, seq_len, _, height, width = img01.shape
    del seq_len
    tgt = src + 1
    log_su, log_sv = head(
        [a[:, src:src + 1] for a in agg],
        [a[:, tgt:tgt + 1] for a in agg],
        img01[:, src:src + 1],
        patch_start_idx,
    )
    rigid, geom_mask = rigid_flow_src_to_tgt(
        depth[:, src, :, :, 0],
        pose[:, src],
        pose[:, tgt],
        intrinsics[:, src],
        intrinsics[:, tgt],
        height,
        width,
    )
    residual = rigid - obs_flow
    mask = (
        geom_mask
        & torch.isfinite(obs_flow).all(-1)
        & torch.isfinite(residual).all(-1)
        & (residual.norm(dim=-1) < 100)
    )
    unsup_mask = geom_mask & (flow_label == 0) if flow_label is not None else torch.zeros_like(mask)
    valid_ratio = mask.float().mean()
    unsup_ratio = unsup_mask.float().mean()
    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
    unsup_loss = unsup_sigma_floor_loss(
        log_su[:, 0],
        log_sv[:, 0],
        unsup_mask,
        unsup_sigma_floor_px,
    )

    if valid_ratio.item() < min_valid_ratio:
        loss = unsup_lambda * unsup_loss if unsup_lambda > 0.0 and unsup_mask.any() else (log_su[:, 0].sum() + log_sv[:, 0].sum()) * 0.0
        mahal = torch.zeros((bsz, height, width), device=img01.device, dtype=img01.dtype)
        return loss, mahal, {
            "resid": residual.detach(),
            "rigid": rigid.detach(),
            "obs": obs_flow.detach(),
            "log_su": log_su[:, 0].detach(),
            "mask": mask.detach(),
            "unsup_mask": unsup_mask.detach(),
            "valid_ratio": valid_ratio.detach(),
            "unsup_ratio": unsup_ratio.detach(),
            "unsup_loss": unsup_loss.detach(),
            "skipped": not (unsup_lambda > 0.0 and unsup_mask.any()),
        }

    loss_nll, mahal = flow_nll_2d(
        log_su[:, 0],
        log_sv[:, 0],
        residual[..., 0],
        residual[..., 1],
        valid_mask=mask,
        beta_nll=beta_nll,
    )
    loss = loss_nll + unsup_lambda * unsup_loss
    return loss, mahal, {
        "resid": residual.detach(),
        "rigid": rigid.detach(),
        "obs": obs_flow.detach(),
        "log_su": log_su[:, 0].detach(),
        "mask": mask.detach(),
        "unsup_mask": unsup_mask.detach(),
        "valid_ratio": valid_ratio.detach(),
        "unsup_ratio": unsup_ratio.detach(),
        "unsup_loss": unsup_loss.detach(),
        "skipped": False,
    }


def flow_color(flow: np.ndarray, mag_max: float):
    fx = np.ascontiguousarray(np.nan_to_num(flow[..., 0]), np.float32)
    fy = np.ascontiguousarray(np.nan_to_num(flow[..., 1]), np.float32)
    mag, ang = cv2.cartToPolar(fx, fy)
    hsv = np.zeros(flow.shape[:2] + (3,), np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (mag_max + 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_heatmap(path: str, rgb01: np.ndarray, diag: dict, src: int, sat: float = 6.0):
    rgb = (rgb01.transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8)
    residual = diag["resid"][0].cpu().numpy()
    rigid = diag["rigid"][0].cpu().numpy()
    obs = diag["obs"][0].cpu().numpy()
    mask = diag["mask"][0].cpu().numpy()
    unsup_mask = diag.get("unsup_mask")
    if unsup_mask is not None:
        unsup_mask = unsup_mask[0].cpu().numpy()
    sigma_u = np.sqrt(np.exp(diag["log_su"][0].cpu().numpy()))
    residual_mag = np.linalg.norm(np.nan_to_num(residual), axis=-1)
    mag_max = np.nanpercentile(np.linalg.norm(obs[mask], axis=-1), 99) if mask.any() else 1.0

    def heat(x, scale, draw_mask=None):
        out = cv2.applyColorMap(np.clip(x / scale * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        if draw_mask is not None:
            out[~draw_mask] = 0
        return out

    def tag(img, text):
        img = img.copy()
        cv2.putText(img, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        return img

    mask_panel = np.zeros_like(rgb)
    mask_panel[mask] = (0, 220, 0)
    if unsup_mask is not None:
        mask_panel[unsup_mask] = (255, 80, 0)

    top = np.hstack([
        tag(rgb, f"RGB source t{src}"),
        tag(flow_color(obs, mag_max), "GT flow t->t+1"),
        tag(flow_color(rigid, mag_max), "rigid(model) t->t+1"),
    ])
    bot = np.hstack([
        tag(heat(residual_mag, sat, mask), "|rigid-GT| residual (supervised only)"),
        tag(heat(sigma_u, sat, None), "pred sigma_u (full image)"),
        tag(mask_panel, f"green=supervised blue=unsup valid={float(diag['valid_ratio']):.4f}"),
    ])
    cv2.imwrite(path, np.vstack([top, bot]))


@torch.no_grad()
def validate(model, head, val_ds, device, args, log, step: int):
    head.eval()
    buf = []
    nseq = min(args.val_sequences, len(val_ds))
    for i in range(nseq):
        clip_len = min(args.max_views, val_ds.num_views)
        try:
            views = val_ds[(i, 0, clip_len)]
        except Exception:
            continue
        gt_flow = stack_gt_flow(views, device)
        flow_label = stack_gt_flow_mask(views, device)
        batch = {k: v.to(device) for k, v in collate_views(views).items()}
        img01 = dust3r_to_model_image(batch["images_dust3r"])
        bsz, seq_len, _, height, width = img01.shape
        del bsz
        agg, patch_start_idx, pose_enc, depth, intrinsics = run_frozen_forward(
            model,
            img01,
            min(args.num_scale_frames, seq_len),
        )
        pose = decode_pose_enc_to_c2w(pose_enc, (height, width), pose_convention=args.pose_convention)
        for src in range(seq_len - 1):
            _, mahal, diag = supervise_forward_pair(
                head,
                agg,
                patch_start_idx,
                img01,
                depth,
                pose,
                intrinsics,
                gt_flow[src][None],
                flow_label[src][None],
                src,
                args.beta_nll,
                args.min_valid_ratio,
                0.0,
                0.0,
            )
            if diag["mask"].any():
                buf.append(mahal[diag["mask"]].cpu())
    if buf:
        log.info(f"VAL@{step}: {format_stats(reliability_stats_2d(torch.cat(buf).unsqueeze(1)))}")
    else:
        log.info(f"VAL@{step}: no valid pixels")
    head.train()


def sample_clip_length(args, rng, device, world_size: int):
    if world_size == 1:
        return int(rng.integers(args.min_views, args.max_views + 1))
    value = torch.empty((), dtype=torch.int64, device=device)
    if dist.get_rank() == 0:
        value.fill_(int(rng.integers(args.min_views, args.max_views + 1)))
    dist.broadcast(value, src=0)
    return int(value.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tartanair_root", default="data/tartanair_v1")
    parser.add_argument("--ckpt", default="checkpoints/lingbot.pt")
    parser.add_argument("--outdir", default="training/outputs/flow_sigma_tartanair_forward")
    parser.add_argument("--min_views", type=int, default=8)
    parser.add_argument("--max_views", type=int, default=24)
    parser.add_argument("--num_scale_frames", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_nll", type=float, default=0.5)
    parser.add_argument("--min_valid_ratio", type=float, default=0.005)
    parser.add_argument("--unsup_sigma_floor_px", type=float, default=3.0)
    parser.add_argument("--unsup_lambda", type=float, default=0.02)
    parser.add_argument("--pose_convention", choices=["w2c", "c2w"], default="c2w")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--viz_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--val_sequences", type=int, default=4)
    args = parser.parse_args()

    assert args.min_views >= 2
    assert args.min_views <= args.max_views
    local_rank, world_rank, world_size = setup_ddp()
    device = torch.device("cuda", local_rank)
    is_rank0 = world_rank == 0
    log = log_setup(world_rank, args.outdir)
    if is_rank0:
        log.info(f"DDP world={world_size} | TartanAir forward flow | {args}")
        (Path(args.outdir) / "heatmaps").mkdir(parents=True, exist_ok=True)
        (Path(args.outdir) / "ckpts").mkdir(parents=True, exist_ok=True)

    model = build_frozen_model(args.ckpt, device, args.num_scale_frames, log)
    head = FlowSigmaHead(
        dim_in=2 * model.embed_dim,
        patch_size=14,
        num_heads=8,
        log_sigma_min=-2.0,
        log_sigma_init=0.0,
    ).to(device)
    head_ddp = DDP(head, device_ids=[local_rank], find_unused_parameters=False) if world_size > 1 else head
    head_mod = head_ddp.module if world_size > 1 else head_ddp

    train_ds = TartanAirFlow(
        ROOT=args.tartanair_root,
        split="train",
        num_views=args.max_views,
        resolution=[(518, 392)],
        aug_crop=0,
        seed=42 + world_rank,
    )
    val_ds = TartanAirFlow(
        ROOT=args.tartanair_root,
        split="test",
        num_views=args.max_views,
        resolution=[(518, 392)],
        aug_crop=0,
        seed=42,
    ) if is_rank0 else None
    if is_rank0:
        log.info(f"train={len(train_ds)} ({train_ds.get_stats()})")
        log.info(f"val={len(val_ds)} ({val_ds.get_stats()})")

    opt = torch.optim.AdamW(head_ddp.parameters(), lr=args.lr, weight_decay=0.01)
    rng = np.random.default_rng(20260703 + 1000 * world_rank)
    stats_buf = deque(maxlen=80)
    last_views = None
    head_ddp.train()
    start_time = time.time()

    for step in range(args.max_steps):
        clip_len = sample_clip_length(args, rng, device, world_size)
        views = None
        for _ in range(50):
            idx = int(rng.integers(len(train_ds)))
            try:
                views = train_ds[(idx, 0, clip_len)]
                break
            except Exception as exc:
                if is_rank0 and step == 0:
                    log.warning(f"sample retry: {exc}")
                views = None
        if views is None:
            views = last_views
        if views is None:
            if world_size > 1:
                dist.barrier()
            continue
        last_views = views

        gt_flow = stack_gt_flow(views, device)
        flow_label = stack_gt_flow_mask(views, device)
        batch = {k: v.to(device) for k, v in collate_views(views).items()}
        img01 = dust3r_to_model_image(batch["images_dust3r"])
        bsz, seq_len, _, height, width = img01.shape
        del bsz
        with torch.no_grad():
            agg, patch_start_idx, pose_enc, depth, intrinsics = run_frozen_forward(
                model,
                img01,
                min(args.num_scale_frames, seq_len),
            )
        pose = decode_pose_enc_to_c2w(pose_enc, (height, width), pose_convention=args.pose_convention)

        loss_total = torch.zeros((), device=device)
        used_pairs = 0
        skipped_pairs = 0
        diag_last = None
        for src in range(seq_len - 1):
            loss_i, mahal, diag = supervise_forward_pair(
                head_ddp,
                agg,
                patch_start_idx,
                img01,
                depth,
                pose,
                intrinsics,
                gt_flow[src][None],
                flow_label[src][None],
                src,
                args.beta_nll,
                args.min_valid_ratio,
                args.unsup_sigma_floor_px,
                args.unsup_lambda,
            )
            finite_loss = torch.isfinite(loss_i)
            # Keep every DDP forward connected to the backward graph on every
            # rank. Different ranks can have different valid-pair counts, and
            # dropping skipped pairs from the graph can deadlock NCCL allreduce.
            loss_total = loss_total + torch.nan_to_num(loss_i, nan=0.0, posinf=0.0, neginf=0.0)
            if diag["skipped"] or not finite_loss:
                skipped_pairs += 1
                continue
            used_pairs += 1
            diag_last = diag
            if diag["mask"].any():
                stats_buf.append(mahal[diag["mask"]].detach().cpu())

        if used_pairs == 0:
            if is_rank0 and step % args.log_every == 0:
                log.info(
                    f"step {step:5d} | S={seq_len:2d} used= 0 skip={skipped_pairs:2d} "
                    f"| no valid pairs above min_valid_ratio={args.min_valid_ratio}"
                )
        loss = loss_total / max(used_pairs, 1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_ddp.parameters(), 1.0)
        opt.step()

        if is_rank0 and step % args.log_every == 0:
            if diag_last is not None and diag_last["mask"].any():
                sig = np.sqrt(np.exp(diag_last["log_su"][diag_last["mask"]].mean().item()))
                mahal_hist = torch.cat(list(stats_buf)) if stats_buf else torch.tensor([float("nan")])
                mahal_med = mahal_hist.median().item()
                valid = float(diag_last["valid_ratio"].item())
                unsup = float(diag_last["unsup_ratio"].item())
                unsup_loss = float(diag_last["unsup_loss"].item())
            else:
                sig, mahal_med, valid, unsup, unsup_loss = float("nan"), float("nan"), 0.0, 0.0, 0.0
            log.info(
                f"step {step:5d} | S={seq_len:2d} used={used_pairs:2d} skip={skipped_pairs:2d} "
                f"| loss {loss.item():8.3f} | <sigma_u>={sig:.2f}px "
                f"| Mahal2 med={mahal_med:.2f}(tgt1.39) | valid={valid:.4f} "
                f"| unsup={unsup:.4f} Lunsup={unsup_loss:.3f} "
                f"| {time.time() - start_time:.0f}s"
            )

        if is_rank0 and step > 0 and step % args.viz_every == 0 and diag_last is not None:
            try:
                src = max(0, seq_len // 2 - 1)
                with torch.no_grad():
                    _, _, diag_viz = supervise_forward_pair(
                        head_mod,
                        agg,
                        patch_start_idx,
                        img01,
                        depth,
                        pose,
                        intrinsics,
                        gt_flow[src][None],
                        flow_label[src][None],
                        src,
                        args.beta_nll,
                        args.min_valid_ratio,
                        args.unsup_sigma_floor_px,
                        args.unsup_lambda,
                    )
                save_heatmap(
                    str(Path(args.outdir) / "heatmaps" / f"step{step:05d}.png"),
                    img01[0, src].detach().cpu().numpy(),
                    diag_viz,
                    src,
                )
                log.info(f"heatmap step{step}")
            except Exception as exc:
                log.warning(f"viz failed at step {step}: {exc}")

        if is_rank0 and step > 0 and step % args.val_every == 0:
            try:
                validate(model, head_mod, val_ds, device, args, log, step)
            except Exception as exc:
                log.warning(f"val failed at step {step}: {exc}")

        if is_rank0 and step > 0 and step % args.save_every == 0:
            try:
                torch.save(
                    {"step": step, "sigma_head": head_mod.state_dict(), "args": vars(args)},
                    Path(args.outdir) / "ckpts" / f"tartanair_forward_step{step:05d}.pt",
                )
            except Exception as exc:
                log.warning(f"save failed at step {step}: {exc}")

        if world_size > 1:
            dist.barrier()

    if is_rank0:
        log.info(f"done {args.max_steps} steps in {time.time() - start_time:.0f}s")
        torch.save(
            {"step": args.max_steps, "sigma_head": head_mod.state_dict(), "args": vars(args)},
            Path(args.outdir) / "ckpts" / f"tartanair_forward_step{args.max_steps:05d}.pt",
        )
        try:
            validate(model, head_mod, val_ds, device, args, log, args.max_steps)
        except Exception as exc:
            log.warning(f"final val failed: {exc}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
