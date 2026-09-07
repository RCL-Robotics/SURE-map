from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


TARTANAIR_K = np.array(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float32,
)


def _quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q.astype(np.float64)
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3, dtype=np.float32)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )


def _load_tartanair_poses(path: Path, n: int) -> np.ndarray:
    if not path.exists():
        return np.repeat(np.eye(4, dtype=np.float32)[None], n, axis=0)
    raw = np.loadtxt(path, dtype=np.float32)
    poses = np.repeat(np.eye(4, dtype=np.float32)[None], raw.shape[0], axis=0)
    poses[:, :3, 3] = raw[:, :3]
    for i in range(raw.shape[0]):
        poses[i, :3, :3] = _quat_xyzw_to_mat(raw[i, 3:7])
    if len(poses) < n:
        pad = np.repeat(poses[-1:len(poses)], n - len(poses), axis=0)
        poses = np.concatenate([poses, pad], axis=0)
    return poses[:n]


def _transform_flow_and_mask(
    flow: np.ndarray,
    mask: np.ndarray,
    K_orig: np.ndarray,
    K_proc: np.ndarray,
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same resize/crop implied by BaseMultiViewDataset's K update."""
    h_raw, w_raw = flow.shape[:2]
    rsu = float(K_orig[0, 0] / K_proc[0, 0])
    rsv = float(K_orig[1, 1] / K_proc[1, 1])
    tmp_w = int(round(w_raw / rsu))
    tmp_h = int(round(h_raw / rsv))

    f = cv2.resize(flow, (tmp_w, tmp_h), interpolation=cv2.INTER_LINEAR)
    f[..., 0] /= rsu
    f[..., 1] /= rsv

    # Follow MAC-VO/TartanAir convention: the flow mask marks pixels that should
    # be used for flow supervision. Out-of-FOV labels are still removed later by
    # the geometric projection/in-bounds mask.
    mask_proc = cv2.resize(mask.astype(np.uint8), (tmp_w, tmp_h), interpolation=cv2.INTER_NEAREST)
    m = mask_proc > 0

    off_x = int(round(float(K_orig[0, 2]) / rsu - float(K_proc[0, 2])))
    off_y = int(round(float(K_orig[1, 2]) / rsv - float(K_proc[1, 2])))
    off_x = max(0, min(off_x, tmp_w - out_w))
    off_y = max(0, min(off_y, tmp_h - out_h))

    f = f[off_y:off_y + out_h, off_x:off_x + out_w]
    mask_proc = mask_proc[off_y:off_y + out_h, off_x:off_x + out_w]
    m = m[off_y:off_y + out_h, off_x:off_x + out_w]
    f = f.astype(np.float32, copy=False)
    f[~m] = np.nan
    return f, mask_proc.astype(np.uint8, copy=False)


class TartanAirFlow(BaseMultiViewDataset):
    """TartanAir v1 left-image + GT-flow adapter for sigma-head training.

    Clips are returned in chronological order. gt_flow at view i is the processed
    TartanAir forward optical flow from view i to view i+1. The last view has no
    outgoing flow and is filled with NaNs.
    """

    def __init__(self, ROOT, *args, split_ratio=0.95, **kwargs):
        self.ROOT = ROOT
        self.split_ratio = split_ratio
        self.video = True
        self.is_metric = True
        self.max_interval = 1
        super().__init__(*args, **kwargs)
        self._load_data(self.split)

    def _load_data(self, split=None):
        trajs = []
        root = Path(self.ROOT)
        for env in sorted(p for p in root.iterdir() if p.is_dir()):
            for diff in ("Easy", "Hard"):
                diff_dir = env / diff
                if not diff_dir.is_dir():
                    continue
                for seq in sorted(p for p in diff_dir.iterdir() if p.is_dir()):
                    image_dir = seq / "image_left"
                    flow_dir = seq / "flow"
                    if not image_dir.is_dir() or not flow_dir.is_dir():
                        continue
                    images = sorted(image_dir.glob("*_left.png"))
                    flows = sorted(flow_dir.glob("*_*_flow.npy"))
                    if len(images) >= self.num_views and len(flows) >= self.num_views - 1:
                        trajs.append(seq)

        if split in ("train", "test") and len(trajs) > 1:
            cut = max(1, int(round(len(trajs) * self.split_ratio)))
            trajs = trajs[:cut] if split == "train" else trajs[cut:]

        self.trajs = [str(p.relative_to(root)) for p in trajs]
        self.samples = []
        for tid, rel in enumerate(self.trajs):
            image_dir = root / rel / "image_left"
            n = len(list(image_dir.glob("*_left.png")))
            for start in range(0, n - self.num_views + 1):
                self.samples.append((tid, start))

    def __len__(self):
        return len(self.samples)

    def get_stats(self):
        return f"{len(self.samples)} clips from {len(self.trajs)} TartanAir trajectories"

    def _frame_id(self, image_path: Path) -> int:
        return int(image_path.name.split("_", 1)[0])

    def _get_views(self, idx, resolution, rng, num_views):
        tid, start = self.samples[idx]
        seq_dir = Path(self.ROOT) / self.trajs[tid]
        image_paths = sorted((seq_dir / "image_left").glob("*_left.png"))
        poses = _load_tartanair_poses(seq_dir / "pose_left.txt", len(image_paths))

        raw_indices = list(range(start, start + num_views))

        views = []
        for v, raw_idx in enumerate(raw_indices):
            image_path = image_paths[raw_idx]
            image = imread_cv2(str(image_path))
            depthmap = np.ones(image.shape[:2], dtype=np.float32)
            K_orig = TARTANAIR_K.copy()
            pose = poses[raw_idx].astype(np.float32)

            image, depthmap, K_proc = self._crop_resize_if_necessary(
                image, depthmap, K_orig.copy(), resolution, rng, info=(str(seq_dir), image_path.name)
            )
            out_w, out_h = image.size

            flow = np.full((out_h, out_w, 2), np.nan, dtype=np.float32)
            flow_mask = np.full((out_h, out_w), 255, dtype=np.uint8)
            next_raw = raw_idx + 1
            if v < num_views - 1 and next_raw < len(image_paths):
                fid0 = self._frame_id(image_paths[raw_idx])
                fid1 = self._frame_id(image_paths[next_raw])
                flow_path = seq_dir / "flow" / f"{fid0:06d}_{fid1:06d}_flow.npy"
                mask_path = seq_dir / "flow" / f"{fid0:06d}_{fid1:06d}_mask.npy"
                if flow_path.exists() and mask_path.exists():
                    raw_flow = np.load(flow_path).astype(np.float32)
                    raw_mask = np.load(mask_path)
                    flow, flow_mask = _transform_flow_and_mask(raw_flow, raw_mask, K_orig, K_proc, out_h, out_w)

            img_mask, ray_mask = self.get_img_and_ray_masks(self.is_metric, v, rng, p=[0.85, 0.1, 0.05])
            views.append(dict(
                img=image,
                depthmap=depthmap.astype(np.float32),
                camera_pose=pose,
                camera_intrinsics=np.asarray(K_proc, dtype=np.float32),
                gt_flow=flow.astype(np.float32),
                gt_flow_mask=flow_mask.astype(np.uint8),
                dataset="TartanAir",
                label=str(seq_dir),
                instance=str(image_path),
                is_metric=self.is_metric,
                is_video=True,
                quantile=np.array(1.0, dtype=np.float32),
                img_mask=img_mask,
                ray_mask=ray_mask,
                camera_only=False,
                depth_only=False,
                single_view=False,
                reset=False,
            ))

        assert len(views) == num_views
        return views
