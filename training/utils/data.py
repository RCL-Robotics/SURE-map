"""Dataset → tensor batch utilities for σ-head training.

Salvaged from the previous 3D Cholesky stack — only the bits we still need:
collation of dust3r view dicts, and the ImageNet/[-1,1] normalisation flip
that LingBot-Map's aggregator requires at the model boundary.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch


def collate_views(views_list: List[Dict]) -> Dict[str, torch.Tensor]:
    """Stack a list of view dicts into [1, S, ...] tensors.

    Each view has at least these dust3r-loaded keys:
        img            tensor [3, H, W]      ImgNorm-normalised to [-1, 1]
        depthmap       ndarray [H, W]        GT depth in metres, -1 = invalid
        camera_pose    ndarray [4, 4]        GT pose (CONVENTION: c2w)
        camera_intrinsics ndarray [3, 3]
        pts3d          ndarray [H, W, 3]     world-frame, derived from depth+pose
        valid_mask     ndarray [H, W]        bool
    """
    def _as_tensor(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    def _stack(key):
        return torch.stack([_as_tensor(v[key]) for v in views_list], dim=0).unsqueeze(0)

    return {
        "images_dust3r": _stack("img"),               # [1, S, 3, H, W]  ∈ [-1, 1]
        "depth_gt":      _stack("depthmap"),          # [1, S, H, W]
        "pose_c2w_gt":   _stack("camera_pose"),       # [1, S, 4, 4]
        "K_gt":          _stack("camera_intrinsics"), # [1, S, 3, 3]
        "pts3d_w_gt":    _stack("pts3d"),             # [1, S, H, W, 3]
        "valid_mask":    _stack("valid_mask"),        # [1, S, H, W]
    }


def dust3r_to_model_image(images_dust3r: torch.Tensor) -> torch.Tensor:
    """Convert dust3r (mean=std=0.5 → [-1, 1]) to model-expected [0, 1].

    The LingBot-Map aggregator re-normalises internally with ImageNet stats, so
    it expects raw [0, 1].  Feeding [-1, 1] would double-normalise.
    """
    return (images_dust3r + 1.0) * 0.5
