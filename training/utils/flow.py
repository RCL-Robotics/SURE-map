"""Reprojection-based optical flow + supervision mask for σ-head training.

For frame pair (t, t-1), for each pixel (u, v) in frame t:

  1. Unproject with depth_t:  cam_t_pt = K⁻¹ · [u, v, 1]ᵀ · depth_t(u, v)
  2. Transform to world frame: world_pt = c2w_t · cam_t_pt
  3. Transform to frame t-1's camera frame: cam_tm1_pt = w2c_{t-1} · world_pt
                                                       = inv(c2w_{t-1}) · world_pt
  4. Project: (û, v̂) = π( K · cam_tm1_pt )

We do this with **predicted** geometry (depth_t_pred, c2w_t_pred, c2w_{t-1}_pred)
to get the "predicted reprojection (û, v̂)_pred", and with **GT** geometry to
get "(û, v̂)_gt".  The 2D residual is:

  r_uv = (û_pred - û_gt,  v̂_pred - v̂_gt)

This is scale-invariant: if (depth_pred, t_pred) = s·(depth_gt, t_gt) for any
constant s, both projections land at the same pixel → r_uv = 0.  No anchor-
scale solving needed.

POSE CONVENTION NOTE
--------------------
We accept poses as **c2w** (camera-to-world, i.e. R, t such that
world_pt = R · cam_pt + t).  This is the dust3r-loaded VKitti GT convention.

The lingbot-map model's `_predict_camera` returns `pose_enc` which decodes via
`pose_encoding_to_extri_intri` to a 3×4 matrix.  Whether that matrix is c2w
or w2c is determined by training.  The training script must call
`decode_pose_enc_to_c2w(pose_enc, pose_convention=...)` with the correct flag.
A runtime convention check (see tests/test_flow.py::test_pose_convention) uses
the actual VKitti GT poses to determine which is correct.
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------------------
# Pose convention helpers
# ---------------------------------------------------------------------------

def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Closed-form SE(3) inverse: inv([R|t]) = [R^T | -R^T·t].

    Args:
        T: [..., 4, 4]
    Returns:
        [..., 4, 4]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]                            # [..., 3, 1]
    R_T = R.transpose(-1, -2)                      # [..., 3, 3]
    t_inv = -torch.matmul(R_T, t)                  # [..., 3, 1]
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_T
    T_inv[..., :3, 3:4] = t_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv


def decode_pose_enc_to_c2w(
    pose_enc: torch.Tensor,
    image_size_hw: tuple,
    pose_convention: str = "w2c",
) -> torch.Tensor:
    """Decode model pose_enc to c2w (Twc) 4×4 matrix.

    Args:
        pose_enc:        [B, S, 9] model output
        image_size_hw:   (H, W) for intrinsics reconstruction
        pose_convention:
            "w2c"  → `pose_encoding_to_extri_intri` returns w2c; we invert → c2w
                     (standard VGGT convention)
            "c2w"  → already c2w; no inversion (Twc convention)
            "auto" → not implemented here; caller must determine via test
    Returns:
        c2w: [B, S, 4, 4]
    """
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    extr, _ = pose_encoding_to_extri_intri(
        pose_enc, image_size_hw=image_size_hw, build_intrinsics=True,
    )                                              # [B, S, 3, 4]
    B, S = extr.shape[:2]
    extr_4x4 = torch.zeros(B, S, 4, 4, device=extr.device, dtype=extr.dtype)
    extr_4x4[..., :3, :] = extr
    extr_4x4[..., 3, 3] = 1.0

    if pose_convention == "w2c":
        return se3_inverse(extr_4x4)
    elif pose_convention == "c2w":
        return extr_4x4
    else:
        raise ValueError(
            f"pose_convention must be 'w2c' or 'c2w', got {pose_convention!r}. "
            "Use tests/test_flow.py::detect_pose_convention to determine."
        )


# ---------------------------------------------------------------------------
# Intrinsics + pixel grid utilities
# ---------------------------------------------------------------------------

def build_pixel_grid(H: int, W: int, device, dtype=torch.float32) -> torch.Tensor:
    """Return [H, W, 3] homogeneous pixel coords (u, v, 1)."""
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([u, v, torch.ones_like(u)], dim=-1)   # [H, W, 3]


# ---------------------------------------------------------------------------
# Main reprojection residual + supervision mask
# ---------------------------------------------------------------------------

def reprojection_residual_and_mask(
    depth_t_pred: torch.Tensor,        # [B, H, W]
    depth_t_gt: torch.Tensor,          # [B, H, W]
    valid_mask_t: torch.Tensor,        # [B, H, W]  bool — source frame depth valid
    pose_t_c2w_pred: torch.Tensor,     # [B, 4, 4]
    pose_tm1_c2w_pred: torch.Tensor,   # [B, 4, 4]
    pose_t_c2w_gt: torch.Tensor,       # [B, 4, 4]
    pose_tm1_c2w_gt: torch.Tensor,     # [B, 4, 4]
    K_pred: torch.Tensor,              # [B, 3, 3] intrinsics from model
    K_gt: torch.Tensor,                # [B, 3, 3] intrinsics from dataset
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (residual_uv, mask) for frame pair (t, t-1).

    All inputs are torch tensors with leading batch dim B.  Poses MUST be c2w.

    The pred and gt paths use **separate intrinsics** (model's predicted K from
    pose_enc[..., 7:9] vs dataset's GT K).  If they're identical you can pass
    the same tensor for both.

    Returns:
        residual_uv: [B, H, W, 2] — (u_pred - u_gt,  v_pred - v_gt) in pixels
        mask:        [B, H, W]    — bool; True ↔ pixel contributes to loss
    """
    device = depth_t_pred.device
    dtype = depth_t_pred.dtype

    B = depth_t_pred.shape[0]

    # ---- Pixel ray (in cam_t frame, before depth scaling) ----
    pixel_grid = build_pixel_grid(H, W, device=device, dtype=dtype)   # [H, W, 3]

    # ray_pred = K_pred^{-1} @ [u, v, 1]   (used for unprojecting predicted depth)
    K_pred_inv = torch.linalg.inv(K_pred)                              # [B, 3, 3]
    ray_pred = torch.einsum("bij,hwj->bhwi", K_pred_inv, pixel_grid)   # [B, H, W, 3]

    K_gt_inv = torch.linalg.inv(K_gt)
    ray_gt = torch.einsum("bij,hwj->bhwi", K_gt_inv, pixel_grid)       # [B, H, W, 3]

    # ---- GT path ----
    cam_t_gt = ray_gt * depth_t_gt.unsqueeze(-1)                       # [B, H, W, 3]
    # world = R_t_gt @ cam_t_gt + t_t_gt
    R_t_gt = pose_t_c2w_gt[..., :3, :3]                                # [B, 3, 3]
    t_t_gt = pose_t_c2w_gt[..., :3, 3]                                 # [B, 3]
    world_gt = (
        torch.einsum("bij,bhwj->bhwi", R_t_gt, cam_t_gt)
        + t_t_gt[:, None, None, :]
    )
    # cam_{t-1} = R_{t-1}^T @ (world - t_{t-1})    (inverse of c2w)
    R_tm1_gt = pose_tm1_c2w_gt[..., :3, :3]
    t_tm1_gt = pose_tm1_c2w_gt[..., :3, 3]
    cam_tm1_gt = torch.einsum(
        "bji,bhwj->bhwi", R_tm1_gt, world_gt - t_tm1_gt[:, None, None, :]
    )
    z_gt = cam_tm1_gt[..., 2]                                          # [B, H, W]
    # Project with GT intrinsics
    proj_gt_h = torch.einsum("bij,bhwj->bhwi", K_gt, cam_tm1_gt)       # [B, H, W, 3]
    proj_gt = proj_gt_h[..., :2] / proj_gt_h[..., 2:3].clamp(min=1e-6)

    # ---- Pred path ----
    cam_t_pred = ray_pred * depth_t_pred.unsqueeze(-1)
    R_t_pred = pose_t_c2w_pred[..., :3, :3]
    t_t_pred = pose_t_c2w_pred[..., :3, 3]
    world_pred = (
        torch.einsum("bij,bhwj->bhwi", R_t_pred, cam_t_pred)
        + t_t_pred[:, None, None, :]
    )
    R_tm1_pred = pose_tm1_c2w_pred[..., :3, :3]
    t_tm1_pred = pose_tm1_c2w_pred[..., :3, 3]
    cam_tm1_pred = torch.einsum(
        "bji,bhwj->bhwi", R_tm1_pred, world_pred - t_tm1_pred[:, None, None, :]
    )
    z_pred = cam_tm1_pred[..., 2]
    proj_pred_h = torch.einsum("bij,bhwj->bhwi", K_pred, cam_tm1_pred)
    proj_pred = proj_pred_h[..., :2] / proj_pred_h[..., 2:3].clamp(min=1e-6)

    # ---- Residual ----
    residual_uv = proj_pred - proj_gt                                  # [B, H, W, 2]

    # ---- Mask (the 6 conditions discussed) ----
    mask = (
        valid_mask_t.bool()                                            # 1. GT depth valid
        & (depth_t_pred > 1e-4)                                        # 2. pred depth positive
        & (z_gt > 1e-4)                                                # 3a. GT z > 0
        & (z_pred > 1e-4)                                              # 3b. pred z > 0
        & (proj_gt[..., 0] >= 0) & (proj_gt[..., 0] < W)               # 4. GT proj in bounds
        & (proj_gt[..., 1] >= 0) & (proj_gt[..., 1] < H)
        & (proj_pred[..., 0] >= 0) & (proj_pred[..., 0] < W)           # 5. pred proj in bounds
        & (proj_pred[..., 1] >= 0) & (proj_pred[..., 1] < H)
        & torch.isfinite(residual_uv).all(-1)                          # 6. finite
    )

    return residual_uv, mask


# ---------------------------------------------------------------------------
# Synthetic-data analytical flow (used by tests/test_flow.py)
# ---------------------------------------------------------------------------

def analytical_flow_constant_depth_translation(
    H: int,
    W: int,
    K: torch.Tensor,                  # [3, 3]
    depth_const: float,
    translation_cam: tuple,           # (tx, ty, tz) — camera moves by this in world frame
                                      # (relative to previous frame, i.e. dt = c2w_t.t - c2w_tm1.t)
    dtype=torch.float32,
    device="cpu",
) -> torch.Tensor:
    """For a static scene at constant depth d, with cam_t-1 at origin (R=I) and
    cam_t translated by (tx, ty, tz) in world frame, no rotation:

    Each pixel (u, v) in cam_t at depth d corresponds to world point:
        p_w = (cam_t_translation) + d · K⁻¹ · [u, v, 1]

    Projecting into cam_{t-1} (at origin, identity):
        cam_tm1_pt = p_w  (since pose_tm1 = I)
        pixel_tm1 = K · p_w / p_w.z

    Returns [H, W, 2] expected pixel coords in frame t-1.  Subtract from grid to
    get analytical flow if needed.
    """
    pixel_grid = build_pixel_grid(H, W, device=device, dtype=dtype)    # [H, W, 3]
    K_inv = torch.linalg.inv(K)
    rays = torch.einsum("ij,hwj->hwi", K_inv, pixel_grid)              # [H, W, 3]
    cam_t_pt = rays * depth_const                                      # [H, W, 3]

    # c2w_t = I rotation, translation = translation_cam
    t = torch.tensor(translation_cam, dtype=dtype, device=device)      # [3]
    world_pt = cam_t_pt + t

    # pose_tm1 = identity → cam_tm1 = world_pt
    cam_tm1_pt = world_pt

    # Project
    proj_h = torch.einsum("ij,hwj->hwi", K, cam_tm1_pt)
    proj = proj_h[..., :2] / proj_h[..., 2:3].clamp(min=1e-6)
    return proj  # [H, W, 2]
