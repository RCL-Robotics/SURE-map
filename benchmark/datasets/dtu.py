"""DTU MVSNet-format dataset loader.

Expected raw layout:
  {raw_data_root}/
    scan1/
      images/00000000.jpg
      depths/00000000.npy          # depth in millimeters
      binary_masks/00000000.png
      cams/00000000_cam.txt        # MVSNet camera file
      pair.txt

Each scan is treated as one benchmark scene.  The loader converts DTU's
millimeter-scale depths and camera translations to meters so the shared point
cloud metrics can use the same thresholds as other indoor datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from benchmark.core.loader import BSSLoader
from benchmark.dataset.base import BaseDataset


DTU_DEPTH_SCALE = 1000.0
DTU_DEPTH_MIN_M = 1e-4
DTU_DEPTH_MAX_M = 10.0
DTU_ICP_THRESHOLD_M = 0.1
DTU_VOXEL_SIZE_M = 4.0 / 512.0


class DtuDataset(BaseDataset):
    """DTU test-set loader for the benchmark framework."""

    def __init__(
        self,
        raw_data_root: str,
        use_mask: bool = True,
        erode_mask: bool = True,
        logger=None,
    ):
        super().__init__(raw_data_root, logger=logger)
        self.use_mask = bool(use_mask)
        self.erode_mask = bool(erode_mask)
        self._frames_cache: Dict[str, List[int]] = {}

    def get_scenes(self) -> List[str]:
        scenes = []
        for d in sorted(self.raw_data_root.iterdir()):
            if (
                d.is_dir()
                and d.name.startswith("scan")
                and (d / "images").is_dir()
                and (d / "cams").is_dir()
                and (d / "depths").is_dir()
            ):
                scenes.append(d.name)
        return scenes

    def get_frame_list(self, scene: str) -> List[int]:
        if scene in self._frames_cache:
            return self._frames_cache[scene]

        image_dir = self.raw_data_root / scene / "images"
        frames = sorted(int(p.stem) for p in image_dir.glob("*.jpg"))
        self._frames_cache[scene] = frames
        return frames

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        scene_dir = self.raw_data_root / scene
        stem = f"{frame_id:08d}"

        rgb_path = scene_dir / "images" / f"{stem}.jpg"
        depth_path = scene_dir / "depths" / f"{stem}.npy"
        mask_path = scene_dir / "binary_masks" / f"{stem}.png"
        cam_path = scene_dir / "cams" / f"{stem}_cam.txt"

        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)

        depth_mm = np.load(depth_path).astype(np.float32)
        depth_mm = np.nan_to_num(depth_mm, posinf=0.0, neginf=0.0, nan=0.0)

        mask = self._load_mask(mask_path, depth_mm.shape)
        depth_m = depth_mm / DTU_DEPTH_SCALE
        if self.use_mask:
            depth_m = np.where(mask, depth_m, 0.0)
        depth_m = depth_m.astype(np.float32)
        depth_m[(depth_m < DTU_DEPTH_MIN_M) | (depth_m > DTU_DEPTH_MAX_M)] = 0.0

        intrinsics, c2w = self._load_mvsnet_camera(cam_path)
        intrinsics_vec = np.array(
            [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]],
            dtype=np.float32,
        )

        return {
            "rgb": rgb,
            "depth": depth_m,
            "mask": mask,
            "pose": c2w,
            "intrinsics": intrinsics_vec,
        }

    def load_global_data(self, scene: str) -> Dict[str, Any]:
        return {}

    def _load_mask(self, mask_path: Path, depth_shape: tuple[int, int]) -> np.ndarray:
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        h, w = depth_shape
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask = mask > 127
        if self.erode_mask:
            kernel = np.ones((10, 10), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1) > 0
        return mask

    @staticmethod
    def _load_mvsnet_camera(cam_path: Path) -> tuple[np.ndarray, np.ndarray]:
        words = cam_path.read_text().split()
        if len(words) < 27:
            raise ValueError(f"Invalid MVSNet camera file: {cam_path}")

        extrinsic = np.array([float(x) for x in words[1:17]], dtype=np.float32).reshape(4, 4)
        intrinsics = np.array([float(x) for x in words[18:27]], dtype=np.float32).reshape(3, 3)

        c2w = np.linalg.inv(extrinsic).astype(np.float32)
        c2w[:3, 3] /= DTU_DEPTH_SCALE
        return intrinsics, c2w

    @staticmethod
    def evaluate_pointcloud(
        gt_loader: BSSLoader,
        pred_loader: BSSLoader,
        logger,
        options: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate DTU reconstruction from depth-backed point clouds.

        This mirrors the existing 7Scenes/NRGBD protocol:
        pixel-corresponded Umeyama for coarse alignment, ICP for fine alignment,
        then Chamfer/precision/recall/F1 at metric thresholds.
        """
        from benchmark.evaluation.points import evaluate_pointcloud as eval_pc
        from benchmark.geometry.registration import (
            apply_transform,
            icp_registration,
            umeyama_registration,
            voxel_downsample,
        )

        options = options or {}
        icp_threshold = float(options.get("icp_threshold", DTU_ICP_THRESHOLD_M))
        voxel_size = float(options.get("voxel_size", DTU_VOXEL_SIZE_M))
        conf_threshold = float(options.get("conf_threshold", 0.0))
        thresholds = options.get("thresholds", [0.05])

        gt_xyzrgb_full, gt_mask_full = gt_loader.load_point_cloud_grid()
        try:
            pred_xyzrgb, pred_mask = pred_loader.load_point_cloud_grid(
                confidence_threshold=conf_threshold
            )
        except Exception as exc:
            logger.warning(f"DTU eval: cannot build pred cloud ({exc})")
            return None

        pred_frame_indices = pred_loader.get_frame_indices()
        gt_xyzrgb_u = gt_xyzrgb_full[pred_frame_indices]
        gt_mask_u = gt_mask_full[pred_frame_indices]

        common_mask = gt_mask_u & pred_mask
        gt_pts_u = gt_xyzrgb_u[common_mask][:, :3]
        pred_pts_u = pred_xyzrgb[common_mask][:, :3]

        if len(gt_pts_u) < 6:
            logger.warning(f"DTU eval: insufficient correspondences ({len(gt_pts_u)})")
            return None

        logger.info(f"DTU eval: Umeyama with {len(gt_pts_u):,} correspondences")
        t_umeyama = umeyama_registration(
            source_points=pred_pts_u,
            target_points=gt_pts_u,
        )

        gt_pts = gt_xyzrgb_full[gt_mask_full][:, :3]
        pred_pts = pred_xyzrgb[common_mask][:, :3]
        pred_after_umeyama = apply_transform(pred_pts, t_umeyama)

        if voxel_size > 0:
            logger.info(
                f"DTU eval: voxel downsampling at {voxel_size:.6f}m "
                f"(pred: {len(pred_after_umeyama):,}, gt: {len(gt_pts):,})"
            )
            pred_ds = voxel_downsample(pred_after_umeyama, voxel_size)
            gt_ds = voxel_downsample(gt_pts, voxel_size)
        else:
            pred_ds = pred_after_umeyama
            gt_ds = gt_pts

        logger.info(f"DTU eval: ICP with threshold {icp_threshold:.3f}m")
        t_icp = icp_registration(
            source_points=pred_ds,
            target_points=gt_ds,
            icp_threshold=icp_threshold,
        )

        pred_eval = apply_transform(pred_ds, t_icp)
        logger.info(
            f"DTU final eval: {len(pred_eval):,} pred pts vs {len(gt_ds):,} gt pts"
        )

        return eval_pc(
            source_points=pred_eval,
            target_points=gt_ds,
            thresholds=thresholds,
        )
