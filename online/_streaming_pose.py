#!/usr/bin/env python3
"""Shared end-to-end SURE-Map streaming pose estimation implementation.

The public runner performs causal LingBot-Map inference, asynchronous segment
scale calibration, and uncertainty-aware local translation correction. It only
writes final and ground-truth trajectories plus ATE metrics; no optimization
matrices or intermediate pose variants are serialized.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import torch.nn.functional as F
import yaml


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))
ENV_BIN = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = ENV_BIN + os.pathsep + os.environ.get("PATH", "")

from lingbot_map.models.gct_stream import GCTStream  # noqa: E402




def build_model(ckpt_path, device, scale_frames=8):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"model checkpoint not found: {ckpt_path}")
    model = GCTStream(
        img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
        kv_cache_sliding_window=64, kv_cache_scale_frames=scale_frames,
        kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
        camera_num_iterations=4, use_sdpa=False, use_gradient_checkpoint=False,
    )
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[model] loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.requires_grad_(False).eval().to(device)
    return model


def umeyama_sim3(src, dst):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, Sig, Vt = np.linalg.svd(cov)
    Dg = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        Dg[2, 2] = -1
    R = U @ Dg @ Vt
    var = (S ** 2).sum() / len(src)
    scale = np.trace(np.diag(Sig) @ Dg) / max(var, 1e-12)
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def write_tum(path, c2w):
    c2w = np.asarray(c2w)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i in range(len(c2w)):
            t = c2w[i, :3, 3]
            q = Rotation.from_matrix(c2w[i, :3, :3]).as_quat()
            f.write(f"{i} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def organized_normals(P):
    dy = torch.zeros_like(P); dx = torch.zeros_like(P)
    dy[1:-1] = P[2:] - P[:-2]; dx[:, 1:-1] = P[:, 2:] - P[:, :-2]
    n = torch.cross(dx, dy, dim=-1)
    return n / (n.norm(dim=-1, keepdim=True) + 1e-9)

def backproj(D, K):
    H, W = D.shape; dev = D.device
    vv, uu = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing="ij")
    ray = torch.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], torch.ones_like(uu)], -1)
    return ray * D[..., None]

def sample_hw(field, grid):
    if field.ndim == 2:
        return F.grid_sample(field[None, None], grid, align_corners=True, padding_mode="zeros")[0, 0]
    return F.grid_sample(field.permute(2, 0, 1)[None], grid, align_corners=True, padding_mode="zeros")[0].permute(1, 2, 0)

def qvals(x):
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return np.zeros(5, dtype=np.float32)
    q = torch.quantile(x.float(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=x.device))
    return q.detach().cpu().numpy().astype(np.float32)

def proj_gn_refine_cur_to_prev_source_sigma(
    D_cur,
    D_prev,
    N_prev,
    T_cur_to_prev,
    K_cur,
    K_prev,
    sigma_u_cur,
    sigma_v_cur,
    iters=3,
    max_depth=80.0,
    plane_sigma_floor=0.01,
    weight_mode="median",
    weight_alpha=2.0,
    weight_min=0.0,
    weight_max=100.0,
    inv_var_ref=None,
    no_sigma=False,
    collect_stats=False,
):
    """Refine current->previous translation with source-frame sigma propagation."""
    H_, W_ = D_cur.shape
    P_cur = backproj(D_cur, K_cur)
    R_cp = T_cur_to_prev[:3, :3]
    t_cp = T_cur_to_prev[:3, 3].clone()
    H_last = torch.eye(3, device=D_cur.device, dtype=D_cur.dtype)
    stats = {}
    iter_stats = []

    for iter_idx in range(iters):
        X = torch.einsum("ij,hwj->hwi", R_cp, P_cur) + t_cp
        z = X[..., 2].clamp(min=1e-6)
        u = K_prev[0, 0] * X[..., 0] / z + K_prev[0, 2]
        v = K_prev[1, 1] * X[..., 1] / z + K_prev[1, 2]
        gx = (u / (W_ - 1)) * 2 - 1
        gy = (v / (H_ - 1)) * 2 - 1
        grid = torch.stack([gx, gy], -1)[None]

        Qd = sample_hw(D_prev, grid)
        n = sample_hw(N_prev, grid)
        Q = torch.stack(
            [
                (u - K_prev[0, 2]) / K_prev[0, 0] * Qd,
                (v - K_prev[1, 2]) / K_prev[1, 1] * Qd,
                Qd,
            ],
            -1,
        )
        r = (n * (X - Q)).sum(-1)
        pdist = (X - Q).norm(dim=-1)
        valid = (
            (D_cur > 0.1)
            & (D_cur < max_depth)
            & (Qd > 0.1)
            & (Qd < max_depth)
            & (X[..., 2] > 0)
            & (u >= 0)
            & (u <= W_ - 1)
            & (v >= 0)
            & (v <= H_ - 1)
            & (n.norm(dim=-1) > 0.5)
        )
        vd = pdist[valid]
        thr = (vd.median() * 3 + 1e-6) if vd.numel() > 0 else torch.tensor(1e9, device=D_cur.device)
        valid = valid & (pdist < thr) & torch.isfinite(r)

        if no_sigma:
            w = valid.float()
            plane_std = None
            floor_frac = None
            coeff_u = None
            coeff_v = None
        else:
            # Source-pixel covariance:
            # dP/du = [D_cur/fx, 0, 0]^T, dP/dv = [0, D_cur/fy, 0]^T.
            # de/dP_cur = n^T R_cp, so only the first two columns contribute.
            nR = torch.einsum("hwi,ij->hwj", n, R_cp)
            coeff_u = nR[..., 0] * (D_cur / K_cur[0, 0])
            coeff_v = nR[..., 1] * (D_cur / K_cur[1, 1])
            sig_u = sigma_u_cur.clamp_min(1e-6)
            sig_v = sigma_v_cur.clamp_min(1e-6)
            floor2 = plane_sigma_floor * plane_sigma_floor
            var_r = (coeff_u * sig_u).square() + (coeff_v * sig_v).square() + floor2
            plane_std = torch.sqrt(var_r).clamp_min(1e-6)
            if weight_mode == "median":
                ref = plane_std[valid].median() if valid.any() else torch.tensor(1.0, device=D_cur.device)
                w_raw = (ref / plane_std).pow(weight_alpha)
            elif weight_mode == "inv_var":
                w_raw = 1.0 / var_r.clamp_min(1e-12)
            elif weight_mode == "inv_var_norm":
                w_raw = 1.0 / var_r.clamp_min(1e-12)
                if inv_var_ref is None:
                    ref = w_raw[valid].median() if valid.any() else torch.tensor(1.0, device=D_cur.device)
                else:
                    ref = torch.as_tensor(inv_var_ref, device=D_cur.device, dtype=D_cur.dtype)
                w_raw = w_raw / ref.clamp_min(1e-12)
            else:
                raise ValueError(weight_mode)
            w = (w_raw * valid.float()).clamp(min=weight_min, max=weight_max) * valid.float()
            floor_frac = floor2 / var_r.clamp_min(1e-12)

        if collect_stats:
            vv_iter = valid
            if vv_iter.any():
                rv = r[vv_iter]
                wv_iter = w[vv_iter]
                wsum = wv_iter.sum()
                rmse = torch.sqrt(rv.square().mean())
                if wsum > 0:
                    wrmse = torch.sqrt((wv_iter * rv.square()).sum() / wsum.clamp_min(1e-12))
                    wmean = wv_iter.mean()
                    wstd = wv_iter.std()
                    ess = wsum.square() / (wv_iter.square().sum() + 1e-12)
                    ess_frac = ess / vv_iter.sum()
                else:
                    wrmse = torch.tensor(float("nan"), device=D_cur.device)
                    wmean = torch.tensor(0.0, device=D_cur.device)
                    wstd = torch.tensor(0.0, device=D_cur.device)
                    ess = torch.tensor(0.0, device=D_cur.device)
                    ess_frac = torch.tensor(0.0, device=D_cur.device)
                iter_stats.append(
                    {
                        "iter": int(iter_idx),
                        "valid": int(vv_iter.sum()),
                        "rmse": float(rmse),
                        "wrmse": float(wrmse),
                        "w_mean": float(wmean),
                        "w_std": float(wstd),
                        "w_ess": float(ess),
                        "w_ess_frac": float(ess_frac),
                    }
                )
            else:
                iter_stats.append({"iter": int(iter_idx), "valid": 0})

        nm = n.reshape(-1, 3)
        rm = r.reshape(-1)
        wm = w.reshape(-1)
        H_last = torch.einsum("k,ki,kj->ij", wm, nm, nm)
        b = -torch.einsum("k,ki->i", wm * rm, nm)
        lam = 1e-2
        try:
            dt = torch.linalg.solve(
                H_last
                + lam
                * torch.eye(3, device=D_cur.device, dtype=D_cur.dtype)
                * (H_last.diagonal().mean() + 1e-6),
                b,
            )
        except RuntimeError:
            dt = torch.zeros(3, device=D_cur.device, dtype=D_cur.dtype)
        tn = T_cur_to_prev[:3, 3].norm() + 1e-4
        if dt.norm() > tn:
            dt = dt * (tn / dt.norm())
        t_cp = t_cp + dt

        if collect_stats and iter_idx == iters - 1:
            vv = valid
            if vv.any():
                if no_sigma:
                    ess = float(vv.sum())
                    stats = {
                        "valid": int(vv.sum()),
                        "w_mean": 1.0,
                        "w_std": 0.0,
                        "w_ess": ess,
                        "w_ess_frac": 1.0,
                        "abs_r_q": qvals(r[vv].abs()),
                        "pdist_q": qvals(pdist[vv]),
                    }
                else:
                    wv = w[vv]
                    ess = wv.sum().square() / (wv.square().sum() + 1e-12)
                    stats = {
                        "valid": int(vv.sum()),
                        "w_mean": float(wv.mean()),
                        "w_std": float(wv.std()),
                        "w_ess": float(ess),
                        "w_ess_frac": float(ess / vv.sum()),
                        "w_q": qvals(wv),
                        "plane_std_q": qvals(plane_std[vv]),
                        "floor_frac_q": qvals(floor_frac[vv]),
                        "abs_coeff_u_q": qvals(coeff_u[vv].abs()),
                        "abs_coeff_v_q": qvals(coeff_v[vv].abs()),
                        "abs_r_q": qvals(r[vv].abs()),
                        "pdist_q": qvals(pdist[vv]),
                    }
            else:
                stats = {"valid": 0}
            if iter_stats:
                first = iter_stats[0]
                last = iter_stats[-1]
                stats["iter_stats"] = iter_stats
                stats["rmse0"] = first.get("rmse", float("nan"))
                stats["rmse_last"] = last.get("rmse", float("nan"))
                stats["wrmse0"] = first.get("wrmse", float("nan"))
                stats["wrmse_last"] = last.get("wrmse", float("nan"))
                stats["rmse_ratio"] = float(
                    last.get("rmse", float("nan")) / (first.get("rmse", 0.0) + 1e-12)
                )
                stats["wrmse_ratio"] = float(
                    last.get("wrmse", float("nan")) / (first.get("wrmse", 0.0) + 1e-12)
                )

    return t_cp, H_last, stats


from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: E402


PAIR_STEPS = (2, 4, 8, 19)


def load_kitti_dataset_cls():
    spec = importlib.util.spec_from_file_location(
        "lingbot_outer_kitti_dataset", REPO / "benchmark" / "datasets" / "kitti.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.KittiDataset


def load_oxford_dataset_cls():
    spec = importlib.util.spec_from_file_location(
        "sure_map_oxford_dataset",
        REPO / "benchmark" / "datasets" / "oxford_spires.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.OxfordSpiresDataset


def load_vbr_dataset_cls():
    spec = importlib.util.spec_from_file_location(
        "sure_map_vbr_dataset",
        REPO / "benchmark" / "datasets" / "vbr.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.VbrDataset


KittiDataset = load_kitti_dataset_cls()
OxfordSpiresDataset = load_oxford_dataset_cls()
VbrDataset = load_vbr_dataset_cls()


def resolve_keyframe_interval(num_frames: int, threshold: int = 320) -> int:
    if num_frames <= threshold:
        return 1
    return int(math.ceil(num_frames / float(threshold)))


def rgb_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0


def load_frame_tensor(ds, seq: str, frame_id: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    item = ds.load_frame_data(seq, int(frame_id))
    if item["pose"] is None:
        raise ValueError(f"Sequence {seq} has no GT pose")
    return (
        rgb_to_tensor(item["rgb"]),
        item["pose"].astype(np.float32),
        item["intrinsics"].astype(np.float32),
    )


def pose3x4_to_4x4(pose: torch.Tensor) -> torch.Tensor:
    out = torch.eye(4, device=pose.device, dtype=torch.float32)
    out[:3, :] = pose.float()
    return out


def decode_batch_pose_intrinsics(
    output: dict[str, torch.Tensor],
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        output["pose_enc"].float(),
        image_size_hw=image_shape,
        build_intrinsics=True,
    )
    poses = torch.stack([pose3x4_to_4x4(p) for p in extrinsic[0]], dim=0)
    return poses, intrinsic[0].float()


def decode_batch_poses(output: dict[str, torch.Tensor], image_shape: tuple[int, int]) -> torch.Tensor:
    poses, _ = decode_batch_pose_intrinsics(output, image_shape)
    return poses


def ate_sim3(c2w: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_t = np.asarray([T[:3, 3] for T in c2w], dtype=np.float64)
    gt_t = np.asarray([T[:3, 3] for T in gt], dtype=np.float64)
    s, R, t = umeyama_sim3(pred_t, gt_t)
    aligned = (s * (R @ pred_t.T).T + t)
    ate = float(np.sqrt(((aligned - gt_t) ** 2).sum(axis=1).mean()))
    return ate, float(s)


def rel_len(a: np.ndarray, b: np.ndarray) -> float:
    rel = np.linalg.inv(a.astype(np.float64)) @ b.astype(np.float64)
    return float(np.linalg.norm(rel[:3, 3]))


def estimate_batch_stream_depth_scale(
    batch_depths: torch.Tensor,
    stream_depth_cache: dict[int, np.ndarray],
    ids: list[int],
    device: torch.device,
    min_depth: float,
    max_depth: float,
    min_valid: int,
) -> tuple[float, int, float]:
    inv_batch_list = []
    inv_stream_list = []
    for local_idx, fid in enumerate(ids):
        if fid not in stream_depth_cache:
            continue
        db = batch_depths[local_idx].float()
        ds = torch.from_numpy(stream_depth_cache[fid].astype(np.float32, copy=False)).to(device)
        valid = (
            torch.isfinite(db)
            & torch.isfinite(ds)
            & (db > min_depth)
            & (db < max_depth)
            & (ds > min_depth)
            & (ds < max_depth)
        )
        if int(valid.sum().item()) == 0:
            continue
        inv_batch_list.append(1.0 / db[valid].clamp(min=1e-6))
        inv_stream_list.append(1.0 / ds[valid].clamp(min=1e-6))

    if not inv_batch_list:
        return 1.0, 0, float("nan")

    inv_batch = torch.cat(inv_batch_list)
    inv_stream = torch.cat(inv_stream_list)
    valid = (
        torch.isfinite(inv_batch)
        & torch.isfinite(inv_stream)
        & (inv_batch > 1e-6)
        & (inv_stream > 1e-6)
    )
    inv_batch = inv_batch[valid]
    inv_stream = inv_stream[valid]
    n_valid = int(inv_batch.numel())
    if n_valid < min_valid:
        return 1.0, n_valid, float("nan")

    s = torch.median((inv_stream / inv_batch).clamp(min=0.1, max=10.0))
    eps = 1e-8
    res0 = inv_stream - s * inv_batch
    delta = torch.quantile(res0.abs().detach(), 0.7).clamp(min=1e-6)
    for _ in range(5):
        res = inv_stream - s * inv_batch
        abs_res = res.abs() + eps
        huber_w = torch.where(abs_res <= delta, torch.ones_like(abs_res), delta / abs_res)
        s_new = torch.sum(huber_w * inv_batch * inv_stream) / (
            torch.sum(huber_w * inv_batch * inv_batch) + eps
        )
        if not torch.isfinite(s_new):
            break
        if torch.abs(s_new - s) <= 1e-4 * (torch.abs(s) + 1e-6):
            s = s_new
            break
        s = s_new

    s = s.clamp(min=0.1, max=10.0)
    norm = torch.median(inv_stream).clamp(min=1e-6)
    final_res = ((inv_stream - s * inv_batch).abs() / norm).median()
    return float(s.item()), n_valid, float(final_res.item())


def load_sigma_head(args, model, device: torch.device):
    training_dir = REPO / "training"
    if str(training_dir) not in sys.path:
        sys.path.insert(0, str(training_dir))
    from heads.flow_sigma_head import FlowSigmaHead

    sigma = FlowSigmaHead(
        dim_in=2 * model.embed_dim,
        patch_size=14,
        num_heads=8,
        log_sigma_min=-2.0,
        log_sigma_init=0.0,
    ).to(device)
    ckpt = torch.load(args.sigma_ckpt, map_location="cpu", weights_only=False)
    sigma.load_state_dict(ckpt["sigma_head"], strict=True)
    sigma.eval()
    return sigma


def clone_agg(agg) -> list[torch.Tensor]:
    if isinstance(agg, torch.Tensor):
        return [agg.detach().float()]
    return [t.detach().float() for t in agg]


def sigma_uv_from_head(
    sigma_head,
    cur_agg: list[torch.Tensor],
    prev_agg: list[torch.Tensor] | None,
    image_01: torch.Tensor,
    patch_start_idx,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if prev_agg is None:
        return None
    lsu, lsv = sigma_head(cur_agg, prev_agg, image_01, patch_start_idx)
    sigma_u = torch.sqrt(torch.exp(lsu[0, 0].float()).clamp_min(1e-12))
    sigma_v = torch.sqrt(torch.exp(lsv[0, 0].float()).clamp_min(1e-12))
    finite = torch.isfinite(sigma_u) & torch.isfinite(sigma_v) & (sigma_u > 0) & (sigma_v > 0)
    if finite.any():
        fill_u = float(sigma_u[finite].median())
        fill_v = float(sigma_v[finite].median())
    else:
        fill_u = fill_v = 1.0
    sigma_u = torch.nan_to_num(sigma_u, nan=fill_u, posinf=fill_u, neginf=fill_u).clamp_min(1e-6)
    sigma_v = torch.nan_to_num(sigma_v, nan=fill_v, posinf=fill_v, neginf=fill_v).clamp_min(1e-6)
    return sigma_u, sigma_v


@torch.no_grad()
def sigma_filtered_mean_flow(
    depth_cur: torch.Tensor,
    pose_cur: torch.Tensor,
    pose_ref: torch.Tensor,
    intr_cur: np.ndarray | torch.Tensor,
    intr_ref: np.ndarray | torch.Tensor,
    sigma_map: torch.Tensor | None,
    sample_stride: int,
    keep_ratio: float,
    min_depth: float,
    max_depth: float,
) -> tuple[float, float, float, int]:
    """Return mean/median flow after keeping lowest-sigma pixels on current frame."""
    device = depth_cur.device
    h, w = depth_cur.shape[:2]
    stride = max(int(sample_stride), 1)
    ys = torch.arange(stride // 2, h, stride, device=device, dtype=torch.float32)
    xs = torch.arange(stride // 2, w, stride, device=device, dtype=torch.float32)
    vv, uu = torch.meshgrid(ys, xs, indexing="ij")
    u = uu.reshape(-1)
    v = vv.reshape(-1)
    yi = v.long().clamp(0, h - 1)
    xi = u.long().clamp(0, w - 1)
    d = depth_cur[yi, xi].float()
    sig = sigma_map[yi, xi].float() if sigma_map is not None else torch.zeros_like(d)

    k_cur = torch.as_tensor(intr_cur, device=device, dtype=torch.float32)
    k_ref = torch.as_tensor(intr_ref, device=device, dtype=torch.float32)
    if k_cur.shape == (3, 3):
        fx, fy, cx, cy = k_cur[0, 0], k_cur[1, 1], k_cur[0, 2], k_cur[1, 2]
    else:
        fx, fy, cx, cy = k_cur[:4]
    if k_ref.shape == (3, 3):
        rfx, rfy, rcx, rcy = k_ref[0, 0], k_ref[1, 1], k_ref[0, 2], k_ref[1, 2]
    else:
        rfx, rfy, rcx, rcy = k_ref[:4]

    valid = (
        torch.isfinite(d)
        & (d > min_depth)
        & (d < max_depth)
    )
    if sigma_map is not None:
        valid = valid & torch.isfinite(sig)
    if int(valid.sum().item()) < 16:
        return float("nan"), float("nan"), 0.0, int(valid.sum().item())

    u0 = u[valid]
    v0 = v[valid]
    d0 = d[valid]
    sig0 = sig[valid]
    x = (u0 - cx) / fx * d0
    y = (v0 - cy) / fy * d0
    pts_cur = torch.stack([x, y, d0, torch.ones_like(d0)], dim=0)
    t_ref_cur = torch.linalg.inv(pose_ref.float()) @ pose_cur.float()
    pts_ref = t_ref_cur @ pts_cur
    z = pts_ref[2].clamp_min(1e-6)
    ur = rfx * pts_ref[0] / z + rcx
    vr = rfy * pts_ref[1] / z + rcy
    proj_valid = (
        (pts_ref[2] > 1e-6)
        & (ur >= 0)
        & (ur <= w - 1)
        & (vr >= 0)
        & (vr <= h - 1)
    )
    if int(proj_valid.sum().item()) < 16:
        return float("nan"), float("nan"), float(proj_valid.float().mean().item()), int(proj_valid.sum().item())

    flow = torch.sqrt((ur[proj_valid] - u0[proj_valid]).square() + (vr[proj_valid] - v0[proj_valid]).square())
    if sigma_map is not None:
        sig_good = sig0[proj_valid]
        q = torch.quantile(sig_good, float(keep_ratio))
        keep = sig_good <= q
        if int(keep.sum().item()) < 16:
            return float("nan"), float("nan"), float(proj_valid.float().mean().item()), int(keep.sum().item())
        kept_flow = flow[keep]
    else:
        kept_flow = flow
    return (
        float(kept_flow.mean().item()),
        float(kept_flow.median().item()),
        float(proj_valid.float().mean().item()),
        int(kept_flow.numel()),
    )


def scale_anchor_observation(
    raw_poses: np.ndarray,
    ids: list[int],
    batch_poses: np.ndarray,
    depth_scale: float,
) -> tuple[float, int, float]:
    if not np.isfinite(depth_scale) or depth_scale <= 1e-6:
        return float("nan"), 0, float("nan")

    batch = batch_poses.astype(np.float64).copy()
    batch[:, :3, 3] /= float(depth_scale)

    step_medians = []
    total_observations = 0
    for step in PAIR_STEPS:
        step_values = []
        for i in range(0, len(ids) - step):
            j = i + step
            ls = rel_len(raw_poses[ids[i]], raw_poses[ids[j]])
            lb = rel_len(batch[i], batch[j])
            if np.isfinite(ls) and np.isfinite(lb) and ls > 1e-6:
                ratio = lb / ls
                if np.isfinite(ratio) and 0.25 < ratio < 4.0:
                    step_values.append(ratio)
        if step_values:
            step_medians.append(float(np.median(np.asarray(step_values, dtype=np.float64))))
            total_observations += len(step_values)
    if not step_medians:
        return float("nan"), 0, float("nan")

    # Equalize temporal baselines so numerous short-step pairs do not dominate.
    arr = np.asarray(step_medians, dtype=np.float64)
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    spread = q75 / max(q25, 1e-9)
    return float(np.median(arr)), int(total_observations), float(spread)


class OnlineScaleAccumulator:
    """Finalize edge scale factors as soon as anchor observations arrive."""

    def __init__(self, num_frames: int, damping: float) -> None:
        self.num_frames = int(num_frames)
        self.damping = float(damping)
        self.edge_factor = np.ones(self.num_frames, dtype=np.float64)
        self.accepted_anchors: list[int] = []
        self.accepted_scales: list[float] = []

    @staticmethod
    def is_valid(obs: float, nobs: int, spread: float) -> bool:
        return (
            np.isfinite(obs)
            and obs > 1e-6
            and int(nobs) >= 8
            and np.isfinite(spread)
            and spread < 1.7
        )

    def add_anchor(self, anchor: int, obs: float, nobs: int, spread: float) -> bool:
        if not self.is_valid(obs, nobs, spread):
            return False

        anchor = int(anchor)
        scale = float(np.exp(self.damping * np.log(float(obs))))

        if not self.accepted_anchors:
            self.edge_factor[1 : min(anchor, self.num_frames - 1) + 1] = scale
        else:
            prev_anchor = self.accepted_anchors[-1]
            prev_scale = self.accepted_scales[-1]
            lo = max(prev_anchor + 1, 1)
            hi = min(anchor, self.num_frames - 1)
            if hi >= lo:
                xs = np.arange(lo, hi + 1, dtype=np.float64)
                denom = max(float(anchor - prev_anchor), 1.0)
                t = (xs - float(prev_anchor)) / denom
                log_s = (1.0 - t) * np.log(prev_scale) + t * np.log(scale)
                self.edge_factor[lo : hi + 1] = np.exp(log_s)

        self.accepted_anchors.append(anchor)
        self.accepted_scales.append(scale)
        return True

    def finalize(self) -> tuple[np.ndarray, int]:
        if self.accepted_anchors:
            last_anchor = self.accepted_anchors[-1]
            last_scale = self.accepted_scales[-1]
            if last_anchor + 1 < self.num_frames:
                self.edge_factor[last_anchor + 1 :] = last_scale
        self.edge_factor[0] = 1.0
        return self.edge_factor.copy(), len(self.accepted_anchors)


def apply_corrections(
    raw_poses: np.ndarray,
    edge_factor: np.ndarray,
    rel_gn_t: np.ndarray,
    gate: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, int]:
    out = [raw_poses[0].astype(np.float64).copy()]
    prev = out[0].copy()
    applied = 0
    alpha = float(alpha)
    for i in range(1, len(raw_poses)):
        e = i - 1
        rel = np.linalg.inv(raw_poses[i - 1].astype(np.float64)) @ raw_poses[i].astype(np.float64)
        raw_t = rel[:3, 3].astype(np.float64)
        raw_norm = float(np.linalg.norm(raw_t))
        base_t = raw_t * float(edge_factor[i])
        base_norm = float(np.linalg.norm(base_t))

        use_gn = e < len(rel_gn_t) and e < len(gate) and bool(gate[e])
        if use_gn and np.isfinite(base_norm) and base_norm > 1e-9 and raw_norm > 1e-9:
            gn = rel_gn_t[e].astype(np.float64)
            gn_norm = float(np.linalg.norm(gn))
            if np.isfinite(gn_norm) and gn_norm > 1e-9:
                fused_t = (1.0 - alpha) * base_t + alpha * gn
                if np.all(np.isfinite(fused_t)):
                    rel[:3, 3] = fused_t
                    applied += 1
                else:
                    rel[:3, 3] = base_t
            else:
                rel[:3, 3] = base_t
        else:
            rel[:3, 3] = base_t

        prev = prev @ rel
        out.append(prev.copy())
    return np.stack(out, axis=0).astype(np.float32), applied


def stat_value(stats: dict, key: str, default: float = np.nan) -> float:
    try:
        return float(stats.get(key, default))
    except (TypeError, ValueError):
        return float(default)


@torch.no_grad()
def run_source_sigma_gn_edge(
    args,
    D_cur: torch.Tensor,
    D_prev: torch.Tensor,
    K_cur: torch.Tensor,
    K_prev: torch.Tensor,
    pose_cur: torch.Tensor,
    pose_prev: torch.Tensor,
    sigma_u: torch.Tensor,
    sigma_v: torch.Tensor,
) -> tuple[np.ndarray, dict[str, float]]:
    with torch.amp.autocast("cuda", enabled=False):
        T_cur_to_prev = torch.linalg.inv(pose_prev.float()) @ pose_cur.float()
        N_prev = organized_normals(backproj(D_prev.float(), K_prev.float()))
        t_sig, H_sig, stats = proj_gn_refine_cur_to_prev_source_sigma(
            D_cur.float(),
            D_prev.float(),
            N_prev,
            T_cur_to_prev.float(),
            K_cur.float(),
            K_prev.float(),
            sigma_u.float(),
            sigma_v.float(),
            iters=args.gn_iters,
            max_depth=args.segment_max_depth,
            plane_sigma_floor=args.plane_sigma_floor,
            weight_mode=args.weight_mode,
            weight_alpha=args.weight_alpha,
            weight_min=args.weight_min,
            weight_max=args.weight_max,
            no_sigma=False,
            collect_stats=True,
        )
        stats = dict(stats)
        if args.use_hessian_validation:
            _, H_no, _ = proj_gn_refine_cur_to_prev_source_sigma(
                D_cur.float(),
                D_prev.float(),
                N_prev,
                T_cur_to_prev.float(),
                K_cur.float(),
                K_prev.float(),
                sigma_u.float(),
                sigma_v.float(),
                iters=args.gn_iters,
                max_depth=args.segment_max_depth,
                plane_sigma_floor=args.plane_sigma_floor,
                weight_mode=args.weight_mode,
                weight_alpha=args.weight_alpha,
                weight_min=args.weight_min,
                weight_max=args.weight_max,
                no_sigma=True,
                collect_stats=False,
            )
            h_sigma = H_sig.detach().float()
            h_plain = H_no.detach().float()
            stats["h_rel_vs_nosigma"] = float(
                torch.linalg.norm(h_sigma - h_plain) /
                (torch.linalg.norm(h_plain) + 1e-12)
            )
        return t_sig.detach().float().cpu().numpy(), stats


def run_batch_anchor(
    args,
    batch_model,
    ids: list[int],
    rgb_cache: dict[int, torch.Tensor],
    depth_cache: dict[int, np.ndarray],
    raw_poses: np.ndarray,
    image_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, int, float, float, int, float]:
    if any(fid not in rgb_cache or fid not in depth_cache for fid in ids):
        return 1.0, 0, float("nan"), float("nan"), 0, float("nan")

    batch = torch.stack([rgb_cache[fid] for fid in ids], dim=0).unsqueeze(0).to(device, dtype=dtype)
    batch_model.clean_kv_cache()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        output = batch_model.forward(
            batch,
            num_frame_for_scale=min(args.nsf, len(ids)),
            num_frame_per_block=len(ids),
            causal_inference=False,
            gather_outputs=True,
        )
    batch_depths = output["depth"][0, :, :, :, 0].float()
    batch_poses = decode_batch_poses(output, image_shape).detach().cpu().numpy().astype(np.float32)
    depth_s, depth_valid, depth_res = estimate_batch_stream_depth_scale(
        batch_depths,
        depth_cache,
        ids,
        device,
        args.segment_min_depth,
        args.segment_max_depth,
        args.segment_min_valid,
    )
    obs, nobs, spread = scale_anchor_observation(raw_poses, ids, batch_poses, depth_s)
    batch_model.clean_kv_cache()
    del output, batch_depths, batch_poses, batch
    return depth_s, depth_valid, depth_res, obs, nobs, spread


def prune_scale_cache(
    rgb_cache: dict[int, torch.Tensor],
    depth_cache: dict[int, np.ndarray],
    min_keep: int,
) -> None:
    """Drop scale keyframes that can no longer participate in future windows."""
    for key in list(rgb_cache.keys()):
        if key < min_keep:
            del rgb_cache[key]
    for key in list(depth_cache.keys()):
        if key < min_keep:
            del depth_cache[key]


def run_sequence(args, seq: str) -> dict[str, float | int | str]:
    seq = str(seq).zfill(2) if args.dataset_type == "kitti" else str(seq)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.dataset_type}_{seq}"
    trajectory_path = outdir / f"{prefix}_sure_map_tum.txt"
    gt_path = outdir / f"{prefix}_gt_tum.txt"

    if args.dataset_type == "kitti":
        ds = KittiDataset(
            args.dataset_root,
            sequences=[seq],
            target_size=[args.target_width, args.target_height],
        )
    elif args.dataset_type == "oxford":
        ds = OxfordSpiresDataset(args.dataset_root, load_img_size=args.target_width)
    elif args.dataset_type == "vbr":
        ds = VbrDataset(
            args.dataset_root,
            scenes=[seq],
            target_size=[args.target_width, args.target_height],
        )
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")
    frame_ids = ds.get_frame_list(seq)
    num_frames = len(frame_ids)
    if num_frames == 0:
        raise ValueError(f"{args.dataset_type} sequence {seq} has no frames")

    first_img, first_gt, _ = load_frame_tensor(ds, seq, frame_ids[0])
    h, w = int(first_img.shape[-2]), int(first_img.shape[-1])
    if (w, h) != (args.target_width, args.target_height):
        raise ValueError(
            f"resize mismatch: configured {args.target_width}x{args.target_height}, "
            f"loaded {w}x{h}"
        )
    print(f"[resize] seq={seq} image={w}x{h}", flush=True)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    scale_frames = min(args.nsf, num_frames)
    model_kfi = resolve_keyframe_interval(num_frames)

    raw_poses = np.zeros((num_frames, 4, 4), dtype=np.float32)
    gt_poses = np.zeros((num_frames, 4, 4), dtype=np.float32)
    pred_intrs = np.zeros((num_frames, 3, 3), dtype=np.float32)
    num_edges = max(num_frames - 1, 0)
    rel_sigma_t = np.zeros((num_edges, 3), dtype=np.float32)
    stats_gate = np.zeros(num_edges, dtype=np.bool_)
    rgb_cache: dict[int, torch.Tensor] = {}
    depth_cache: dict[int, np.ndarray] = {}
    kf_ids: list[int] = []
    fixed_kf_ids: list[int] = []
    anchors: list[int] = []
    scale_acc = OnlineScaleAccumulator(num_frames, args.damping)
    last_anchor_kf_pos: int | None = None

    stream_model = build_model(args.ckpt, device, scale_frames=args.nsf)
    if args.bf16_aggregator:
        stream_model.aggregator = stream_model.aggregator.to(torch.bfloat16)
    stream_model.clean_kv_cache()
    sigma_head = load_sigma_head(args, stream_model, device)
    sigma_stream = torch.cuda.Stream(device=device)
    sigma_request = {"enabled": False, "image": None, "prev_agg": None}
    sigma_result = {"cur_agg": None, "uv": None, "event": None}
    stream_cap = {}

    def reset_sigma_result() -> None:
        sigma_result["cur_agg"] = None
        sigma_result["uv"] = None
        sigma_result["event"] = None

    def stream_hook(_module, _inputs, output) -> None:
        agg = output[0] if isinstance(output, (tuple, list)) else output
        ps = output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else 6
        stream_cap.update(agg=agg, ps=ps)
        reset_sigma_result()
        if (
            not sigma_request["enabled"]
            or sigma_request["prev_agg"] is None
            or sigma_request["image"] is None
        ):
            return
        current_stream = torch.cuda.current_stream(device)
        with torch.cuda.stream(sigma_stream):
            sigma_stream.wait_stream(current_stream)
            cur_agg = clone_agg(agg)
            with torch.amp.autocast("cuda", enabled=False):
                sigma_uv = sigma_uv_from_head(
                    sigma_head,
                    cur_agg,
                    sigma_request["prev_agg"],
                    sigma_request["image"].float(),
                    ps,
                )
            event = torch.cuda.Event()
            event.record(sigma_stream)
        sigma_result["cur_agg"] = cur_agg
        sigma_result["uv"] = sigma_uv
        sigma_result["event"] = event

    stream_model.aggregator.register_forward_hook(stream_hook)

    batch_model = build_model(args.ckpt, device, scale_frames=args.nsf)
    if args.bf16_aggregator:
        batch_model.aggregator = batch_model.aggregator.to(torch.bfloat16)
    batch_model.clean_kv_cache()
    pending_scale = []
    scale_executor = ThreadPoolExecutor(max_workers=1)

    t0 = time.time()
    bootstrap_imgs = []
    for i in range(scale_frames):
        if i == 0:
            img, gt = first_img, first_gt
        else:
            img, gt, _ = load_frame_tensor(ds, seq, frame_ids[i])
        bootstrap_imgs.append(img)
        gt_poses[i] = gt

    prev_agg = None
    last_kf_pose_t = None
    last_kf_intr = None
    last_kf_frame = 0
    recent_imgs: dict[int, torch.Tensor] = {}
    recent_depths: dict[int, np.ndarray] = {}
    recent_intrs: dict[int, np.ndarray] = {}

    def cache_recent(fid: int, img_t: torch.Tensor, depth_t: torch.Tensor, intr_np: np.ndarray) -> None:
        recent_imgs[fid] = img_t
        recent_depths[fid] = depth_t.detach().cpu().numpy().astype(np.float16)
        recent_intrs[fid] = intr_np
        min_recent = max(0, last_kf_frame - args.min_keyframe_gap - 2)
        for key in list(recent_imgs.keys()):
            if key < min_recent and key not in rgb_cache:
                del recent_imgs[key]
        for key in list(recent_depths.keys()):
            if key < min_recent and key not in depth_cache:
                del recent_depths[key]
        for key in list(recent_intrs.keys()):
            if key < min_recent and key not in rgb_cache:
                del recent_intrs[key]

    def add_batch_keyframe(fid: int) -> bool:
        if fid in kf_ids or fid not in recent_imgs or fid not in recent_depths:
            return False
        rgb_cache[fid] = recent_imgs[fid]
        depth_cache[fid] = recent_depths[fid]
        kf_ids.append(fid)
        return True

    def record_scale_result(record, result) -> None:
        depth_s, _, _, obs, nobs, spread = result
        anchor_frame = int(record["anchor_frame"])
        anchors.append(anchor_frame)
        accepted_now = scale_acc.add_anchor(anchor_frame, obs, nobs, spread)
        prune_scale_cache(rgb_cache, depth_cache, int(record["min_keep"]))
        print(
            f"[anchor] seq={seq} frame={anchor_frame}/{num_frames - 1} "
            f"fixed_kfs={record['fixed_kfs']} batch_kfs={record['batch_kfs']} "
            f"depth_s={depth_s:.4f} obs={obs:.4f} nobs={nobs} "
            f"spread={spread:.3f} accepted={int(accepted_now)} async=1",
            flush=True,
        )

    def drain_scale_futures(block: bool = False) -> None:
        while pending_scale and (block or pending_scale[0]["future"].done()):
            record = pending_scale.pop(0)
            result = record["future"].result()
            record_scale_result(record, result)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        scale_in = torch.stack(bootstrap_imgs, dim=0).unsqueeze(0).to(device, dtype=dtype)
        out = stream_model.forward(
            scale_in,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,
            causal_inference=True,
        )
        boot_poses_t, boot_intrs_t = decode_batch_pose_intrinsics(out, (h, w))
        boot_poses_t = boot_poses_t.float()
        boot_intrs_t = boot_intrs_t.float()
        raw_poses[:scale_frames] = boot_poses_t.detach().cpu().numpy().astype(np.float32)
        pred_intrs[:scale_frames] = boot_intrs_t.detach().cpu().numpy().astype(np.float32)
        if "agg" in stream_cap:
            prev_agg = [t[:, -1:].detach().float().contiguous() for t in clone_agg(stream_cap["agg"])]
        for j in range(scale_frames):
            cache_recent(
                j,
                bootstrap_imgs[j],
                out["depth"][0, j, :, :, 0].float(),
                pred_intrs[j],
            )
        rgb_cache[0] = recent_imgs[0]
        depth_cache[0] = recent_depths[0]
        kf_ids.append(0)
        fixed_kf_ids.append(0)
        last_kf_pose_t = boot_poses_t[0].detach().clone()
        last_kf_intr = pred_intrs[0]
        del out, scale_in, boot_poses_t, boot_intrs_t

        for i in range(scale_frames, num_frames):
            img, gt, _ = load_frame_tensor(ds, seq, frame_ids[i])
            gt_poses[i] = gt

            is_model_keyframe = (model_kfi <= 1) or ((i - scale_frames) % model_kfi == 0)
            if not is_model_keyframe:
                stream_model._set_skip_append(True)
            frame_f32 = img.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
            sigma_request["enabled"] = True
            sigma_request["image"] = frame_f32
            sigma_request["prev_agg"] = prev_agg
            frame_in = frame_f32.to(dtype=dtype)
            out = stream_model.forward(
                frame_in,
                num_frame_for_scale=scale_frames,
                num_frame_per_block=1,
                causal_inference=True,
            )
            if not is_model_keyframe:
                stream_model._set_skip_append(False)

            poses_t, intrs_t = decode_batch_pose_intrinsics(out, (h, w))
            pose_t = poses_t[0].float()
            intr_t = intrs_t[0].float()
            raw_poses[i] = pose_t.detach().cpu().numpy().astype(np.float32)
            pred_intrs[i] = intr_t.detach().cpu().numpy().astype(np.float32)

            sigma_uv = None
            cur_agg = None
            if sigma_result["event"] is not None:
                torch.cuda.current_stream(device).wait_event(sigma_result["event"])
            cur_agg = sigma_result["cur_agg"]
            sigma_uv = sigma_result["uv"]
            if cur_agg is None and "agg" in stream_cap:
                cur_agg = clone_agg(stream_cap["agg"])
            sigma_request["enabled"] = False
            sigma_request["image"] = None
            sigma_request["prev_agg"] = None

            current_depth_t = out["depth"][0, 0, :, :, 0].float()
            scale_intr_i = pred_intrs[i]
            cache_recent(i, img, current_depth_t, scale_intr_i)

            edge_idx = i - 1
            if (
                sigma_uv is not None
                and edge_idx >= 0
                and (i - 1) in recent_depths
            ):
                D_prev = torch.from_numpy(recent_depths[i - 1].astype(np.float32, copy=False)).to(device)
                K_prev = torch.from_numpy(pred_intrs[i - 1]).to(device)
                K_cur = intr_t.to(device)
                pose_prev = torch.from_numpy(raw_poses[i - 1]).to(device)
                t_sig, st = run_source_sigma_gn_edge(
                    args,
                    current_depth_t,
                    D_prev,
                    K_cur,
                    K_prev,
                    pose_t,
                    pose_prev,
                    sigma_uv[0],
                    sigma_uv[1],
                )
                rel_sigma_t[edge_idx] = t_sig
                wrmse_ratio = stat_value(st, "wrmse_ratio")
                gate_ok = (
                    bool(st.get("valid", 0))
                    and wrmse_ratio <= args.stats_wrmse_max
                )
                if args.use_hessian_validation:
                    gate_ok = gate_ok and (
                        stat_value(st, "h_rel_vs_nosigma") <= args.hessian_relative_threshold
                    )
                stats_gate[edge_idx] = gate_ok
                del D_prev, K_prev, K_cur, pose_prev

            if cur_agg is not None:
                prev_agg = cur_agg

            accepted_fixed_kf = False
            added_extra_kf = False
            flow_mean = float("nan")
            flow_median = float("nan")
            flow_overlap = 0.0
            fixed_candidate = (
                last_kf_pose_t is not None
                and (i - last_kf_frame) >= args.min_keyframe_gap
            )
            if fixed_candidate:
                flow_mean, flow_median, flow_overlap, _ = sigma_filtered_mean_flow(
                    current_depth_t,
                    pose_t,
                    last_kf_pose_t,
                    scale_intr_i,
                    last_kf_intr,
                    None,
                    args.motion_sample_stride,
                    1.0,
                    args.segment_min_depth,
                    args.segment_max_depth,
                )
                if (
                    np.isfinite(flow_mean)
                    and flow_mean > args.flow_threshold
                    and flow_overlap > args.extra_overlap_threshold
                ):
                    mid = (last_kf_frame + i) // 2
                    if mid != last_kf_frame and mid != i:
                        added_extra_kf = add_batch_keyframe(mid)

                accepted_fixed_kf = add_batch_keyframe(i)
                fixed_kf_ids.append(i)
                last_kf_pose_t = pose_t.detach().clone()
                last_kf_intr = scale_intr_i
                last_kf_frame = i
                if len(fixed_kf_ids) % args.kf_progress_every == 0:
                    print(
                        f"[kf] seq={seq} fixed={len(fixed_kf_ids)} total={len(kf_ids)} frame={i} "
                        f"flow_mean={flow_mean:.2f} flow_med={flow_median:.2f} "
                        f"overlap={flow_overlap:.3f} extra={int(added_extra_kf)}",
                        flush=True,
                    )

            del out, frame_in, frame_f32, poses_t, intrs_t

            if (
                accepted_fixed_kf
                and len(fixed_kf_ids) >= args.segment_history_count
                and len(kf_ids) >= args.segment_history_count
                and (
                    last_anchor_kf_pos is None
                    or (len(fixed_kf_ids) - 1 - last_anchor_kf_pos) >= args.anchor_keyframe_step
                )
            ):
                ids = list(kf_ids[-args.segment_history_count :])
                anchor_frame = ids[-1]
                last_anchor_kf_pos = len(fixed_kf_ids) - 1
                record = {
                    "anchor_frame": anchor_frame,
                    "min_keep": ids[0],
                    "fixed_kfs": len(fixed_kf_ids),
                    "batch_kfs": len(kf_ids),
                }
                rgb_snapshot = {fid: rgb_cache[fid] for fid in ids if fid in rgb_cache}
                depth_snapshot = {fid: depth_cache[fid].copy() for fid in ids if fid in depth_cache}
                future = scale_executor.submit(
                    run_batch_anchor,
                    args,
                    batch_model,
                    ids,
                    rgb_snapshot,
                    depth_snapshot,
                    raw_poses.copy(),
                    (h, w),
                    device,
                    dtype,
                )
                record["future"] = future
                pending_scale.append(record)
                print(
                    f"[anchor-submit] seq={seq} frame={anchor_frame}/{num_frames - 1} "
                    f"fixed_kfs={len(fixed_kf_ids)} batch_kfs={len(kf_ids)} pending={len(pending_scale)}",
                    flush=True,
                )

            drain_scale_futures(block=False)
            if (i + 1) % args.progress_every == 0 or i == num_frames - 1:
                print(f"[stream] seq={seq} frame {i + 1}/{num_frames}", flush=True)

    frontend_elapsed = time.time() - t0
    drain_scale_futures(block=True)
    scale_executor.shutdown(wait=True)

    edge_factor, accepted = scale_acc.finalize()
    final_poses, corrected_edges = apply_corrections(
        raw_poses,
        edge_factor,
        rel_sigma_t,
        stats_gate,
        args.translation_alpha,
    )
    write_tum(trajectory_path, final_poses)
    write_tum(gt_path, gt_poses)

    if num_frames < 2:
        raise ValueError("ATE evaluation requires at least two frames")
    ate_rmse, sim3_scale = ate_sim3(final_poses[1:], gt_poses[1:])
    elapsed = time.time() - t0
    frontend_fps = float(num_frames / max(frontend_elapsed, 1e-9))
    total_fps = float(num_frames / max(elapsed, 1e-9))
    metrics = {
        "sequence": seq,
        "frames": int(num_frames),
        "ate_rmse_m": float(ate_rmse),
        "sim3_scale": float(sim3_scale),
        "use_hessian_validation": bool(args.use_hessian_validation),
        "translation_alpha": float(args.translation_alpha),
        "corrected_edges": int(corrected_edges),
        "scale_anchors": int(accepted),
        "frontend_elapsed_sec": float(frontend_elapsed),
        "frontend_fps": frontend_fps,
        "total_elapsed_sec": float(elapsed),
        "total_fps": total_fps,
        "trajectory": str(trajectory_path),
        "ground_truth": str(gt_path),
    }
    stream_model.clean_kv_cache()
    batch_model.clean_kv_cache()
    del stream_model, batch_model, sigma_head
    torch.cuda.empty_cache()

    print(
        f"[done] seq={seq} frames={num_frames} ATE={ate_rmse:.6f}m "
        f"corrected_edges={corrected_edges}/{num_edges} anchors={accepted}/{len(anchors)} "
        f"dyn_kfs={len(kf_ids)} frontend_fps={frontend_fps:.2f} total_fps={total_fps:.2f} "
        f"saved={trajectory_path}",
        flush=True,
    )
    return metrics


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO / path).resolve()


def require_mapping(config: dict, key: str) -> dict:
    value = config.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a mapping")
    return value


def load_runtime_config(path: Path, dataset_type: str) -> SimpleNamespace:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")

    dataset = require_mapping(config, "dataset")
    checkpoints = require_mapping(config, "checkpoints")
    inference = require_mapping(config, "inference")
    keyframes = require_mapping(config, "keyframes")
    scale = require_mapping(config, "scale_calibration")
    correction = require_mapping(config, "local_correction")
    configured_dataset = str(dataset.get("name", "")).lower()
    if configured_dataset != dataset_type:
        raise ValueError(
            f"dataset.name must be '{dataset_type}' for this runner, got '{configured_dataset}'"
        )

    if str(inference.get("mode", "")).lower() != "strict_streaming":
        raise ValueError("inference.mode must be 'strict_streaming'")
    if str(inference.get("attention", "")).lower() != "flash":
        raise ValueError("inference.attention must be 'flash'")
    if str(inference.get("aggregator", "")).lower() not in {"bf16", "bfloat16"}:
        raise ValueError("inference.aggregator must be 'BF16'")

    input_size = inference.get("input_size")
    if not isinstance(input_size, list) or len(input_size) != 2:
        raise ValueError("inference.input_size must be [width, height]")
    dataset_root = resolve_repo_path(str(dataset["root"]))
    sequences = dataset.get("sequences", "all")
    if isinstance(sequences, str) and sequences.lower() == "all":
        if dataset_type == "kitti":
            sequences = [f"{index:02d}" for index in range(11)]
        elif dataset_type == "oxford":
            sequences = OxfordSpiresDataset(
                str(dataset_root),
                load_img_size=int(input_size[0]),
            ).get_scenes()
        else:
            sequences = VbrDataset(
                str(dataset_root),
                target_size=[int(input_size[0]), int(input_size[1])],
            ).get_scenes()
    elif isinstance(sequences, list):
        sequences = [
            str(sequence).zfill(2) if dataset_type == "kitti" else str(sequence)
            for sequence in sequences
        ]
    else:
        raise ValueError("dataset.sequences must be 'all' or a list")
    if not sequences:
        raise ValueError("dataset.sequences resolved to an empty list")

    use_hessian = bool(correction.get("use_hessian_validation", True))
    return SimpleNamespace(
        dataset_type=dataset_type,
        dataset_root=str(dataset_root),
        seqs=sequences,
        outdir=str(resolve_repo_path(str(config["output_dir"]))),
        ckpt=str(resolve_repo_path(str(checkpoints["backbone"]))),
        sigma_ckpt=str(resolve_repo_path(str(checkpoints["uncertainty"]))),
        device=str(inference.get("device", "cuda")),
        nsf=int(inference["initial_window"]),
        target_width=int(input_size[0]),
        target_height=int(input_size[1]),
        bf16_aggregator=True,
        min_keyframe_gap=int(keyframes["regular_interval"]),
        flow_threshold=float(keyframes["flow_threshold"]),
        extra_overlap_threshold=float(keyframes["geometric_overlap_threshold"]),
        segment_history_count=int(scale["window_size"]),
        anchor_keyframe_step=int(scale["period"]),
        damping=float(scale["damping"]),
        use_hessian_validation=use_hessian,
        translation_alpha=0.2 if use_hessian else 0.04,
        gn_iters=int(correction["gn_iterations"]),
        stats_wrmse_max=float(correction["wrmse_threshold"]),
        # Stable low-level settings shared by the reported KITTI/Oxford results.
        hessian_relative_threshold=0.1,
        segment_min_depth=0.1,
        segment_max_depth=80.0,
        segment_min_valid=512,
        motion_sample_stride=4,
        plane_sigma_floor=0.01,
        weight_mode="median",
        weight_alpha=2.0,
        weight_min=0.0,
        weight_max=100.0,
        kf_progress_every=50,
        progress_every=500,
    )


def cli_main(dataset_type: str, default_config: Path) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
    )
    cli = parser.parse_args()
    args = load_runtime_config(cli.config.resolve(), dataset_type)

    print(
        f"[config] mode=strict_streaming attention=flash aggregator=BF16 "
        f"hessian={int(args.use_hessian_validation)} alpha={args.translation_alpha:.2f}",
        flush=True,
    )
    results = [run_sequence(args, sequence) for sequence in args.seqs]
    dataset_label = {
        "kitti": "KITTI",
        "oxford": "Oxford Spires",
        "vbr": "VBR",
    }[dataset_type]
    summary = {
        "dataset": dataset_label,
        "num_sequences": len(results),
        "mean_ate_rmse_m": float(np.mean([row["ate_rmse_m"] for row in results])),
        "sequences": results,
    }
    summary_path = Path(args.outdir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[summary] sequences={len(results)} "
        f"mean_ATE={summary['mean_ate_rmse_m']:.6f}m saved={summary_path}",
        flush=True,
    )
