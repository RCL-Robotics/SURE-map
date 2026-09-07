#!/usr/bin/env python
"""Point-cloud uncertainty filtering for BSS reconstruction datasets.

This is intentionally offline: it reuses BSS predictions already produced by
the benchmark runner.  Confidence maps are expected to be sigma-derived
confidence (larger = more reliable), e.g. 1 / sqrt(sigma_u^2 + sigma_v^2).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree as KDTree

from benchmark.core.config import ConfigManager
from benchmark.core.loader import BSSLoader
from benchmark.core.storage import BSSManager
from benchmark.evaluation.points import evaluate_pointcloud as eval_pc
from benchmark.geometry.resize import ResizeContext
from benchmark.geometry.registration import (
    apply_transform,
    icp_registration,
    umeyama_registration,
    voxel_downsample,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "sure_map_neural_rgbd.yaml"


def _load_conf_grid(loader: BSSLoader) -> Optional[np.ndarray]:
    conf_list = loader.load_confidence_list()
    if conf_list is None:
        return None
    if any(c is None for c in conf_list):
        return None
    return np.stack([np.asarray(c, dtype=np.float32) for c in conf_list], axis=0)


def _confidence_mask(
    base_mask: np.ndarray,
    conf: Optional[np.ndarray],
    drop_quantile: float,
) -> np.ndarray:
    if conf is None or drop_quantile <= 0:
        return base_mask.copy()
    valid_conf = conf[base_mask & np.isfinite(conf)]
    if valid_conf.size == 0:
        return base_mask.copy()
    thr = np.percentile(valid_conf, drop_quantile * 100.0)
    return base_mask & np.isfinite(conf) & (conf >= thr)


def weighted_umeyama(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Similarity transform source -> target with diagonal correspondence weights."""
    src = source_points[:, :3].astype(np.float64)
    tgt = target_points[:, :3].astype(np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    good = np.isfinite(src).all(axis=1) & np.isfinite(tgt).all(axis=1) & np.isfinite(w) & (w > 0)
    src, tgt, w = src[good], tgt[good], w[good]
    if src.shape[0] < 6:
        return umeyama_registration(source_points, target_points)
    w = w / (w.sum() + 1e-12)

    mu_x = (src * w[:, None]).sum(axis=0)
    mu_y = (tgt * w[:, None]).sum(axis=0)
    x0 = src - mu_x
    y0 = tgt - mu_y
    var_x = (w * np.sum(x0 * x0, axis=1)).sum()
    cov = (y0 * w[:, None]).T @ x0
    u, d, vh = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        s[2, 2] = -1
    r = u @ s @ vh
    scale = np.trace(np.diag(d) @ s) / max(var_x, 1e-12)
    t = mu_y - scale * r @ mu_x
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = scale * r
    out[:3, 3] = t
    return out


def _robust_conf_weights(
    weights: np.ndarray,
    gamma: float = 2.0,
    w_min: float = 0.05,
    w_max: float = 20.0,
) -> np.ndarray:
    """Median-normalize confidence and amplify it with bounded power weights."""
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    good = np.isfinite(w) & (w > 0)
    if not np.any(good):
        return np.ones_like(w, dtype=np.float64)

    med = np.median(w[good])
    if not np.isfinite(med) or med <= 0:
        med = 1.0

    out = np.ones_like(w, dtype=np.float64)
    out[good] = w[good] / (med + 1e-12)
    out = np.power(np.clip(out, 1e-6, None), gamma)
    return np.clip(out, w_min, w_max)


def weighted_voxel_downsample(
    points: np.ndarray,
    weights: np.ndarray,
    voxel_size: float,
    robust: bool = False,
    weight_gamma: float = 2.0,
    weight_min: float = 0.05,
    weight_max: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Voxel fuse points with confidence weights; return fused xyz and mean weights."""
    pts = np.asarray(points[:, :3], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    good = np.isfinite(pts).all(axis=1) & np.isfinite(w) & (w > 0)
    pts, w = pts[good], w[good]
    if robust:
        w = _robust_conf_weights(
            w,
            gamma=weight_gamma,
            w_min=weight_min,
            w_max=weight_max,
        )
    if pts.shape[0] == 0:
        return pts.astype(np.float32), w.astype(np.float32)
    if voxel_size <= 0:
        return pts.astype(np.float32), w.astype(np.float32)

    keys = np.floor(pts / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    sum_w = np.bincount(inv, weights=w, minlength=n)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    fused = np.empty((n, 3), dtype=np.float64)
    for c in range(3):
        fused[:, c] = np.bincount(inv, weights=w * pts[:, c], minlength=n) / np.maximum(sum_w, 1e-12)
    mean_w = sum_w / np.maximum(cnt, 1.0)
    return fused.astype(np.float32), mean_w.astype(np.float32)


def _rigid_fit_weighted(src: np.ndarray, tgt: np.ndarray, weights: np.ndarray) -> np.ndarray:
    src = src.astype(np.float64)
    tgt = tgt.astype(np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    good = np.isfinite(src).all(axis=1) & np.isfinite(tgt).all(axis=1) & np.isfinite(w) & (w > 0)
    src, tgt, w = src[good], tgt[good], w[good]
    if src.shape[0] < 3:
        return np.eye(4, dtype=np.float64)
    w = w / (w.sum() + 1e-12)
    mu_s = (src * w[:, None]).sum(axis=0)
    mu_t = (tgt * w[:, None]).sum(axis=0)
    xs = src - mu_s
    xt = tgt - mu_t
    h = (xs * w[:, None]).T @ xt
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = mu_t - r @ mu_s
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def weighted_icp_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    weights: np.ndarray,
    icp_threshold: float,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> np.ndarray:
    src = np.asarray(source_points[:, :3], dtype=np.float64)
    tgt = np.asarray(target_points[:, :3], dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if src.shape[0] < 3 or tgt.shape[0] < 3:
        return np.eye(4, dtype=np.float64)
    w = np.clip(w / (np.median(w[np.isfinite(w) & (w > 0)]) + 1e-12), 0.05, 20.0)
    tree = KDTree(tgt)
    transform = np.eye(4, dtype=np.float64)
    prev_rmse = np.inf
    for _ in range(max_iterations):
        cur = apply_transform(src, transform)
        dist, idx = tree.query(cur, workers=-1)
        keep = np.isfinite(dist) & (dist < icp_threshold)
        if keep.sum() < 3:
            break
        inc = _rigid_fit_weighted(cur[keep], tgt[idx[keep]], w[keep])
        transform = inc @ transform
        rmse = float(np.sqrt(np.mean(dist[keep] ** 2)))
        if abs(prev_rmse - rmse) < tolerance:
            break
        prev_rmse = rmse
    return transform


def statistical_outlier_filter(points: np.ndarray, nb_neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
    import open3d as o3d

    if len(points) < nb_neighbors + 1:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return points[np.asarray(ind, dtype=np.int64)]


def load_scene_data(
    gt_loader: BSSLoader,
    pred_loader: BSSLoader,
    voxel_size: float,
) -> Dict[str, np.ndarray]:
    """Load and cache all heavy per-scene arrays shared by ablations."""
    gt_xyzrgb_full, gt_mask_full = gt_loader.load_point_cloud_grid()
    pred_xyzrgb, pred_mask_base = pred_loader.load_point_cloud_grid()
    conf = _load_conf_grid(pred_loader)
    pred_frame_indices = pred_loader.get_frame_indices()
    gt_xyzrgb_u = gt_xyzrgb_full[pred_frame_indices]
    gt_mask_u = gt_mask_full[pred_frame_indices]
    gt_pts = gt_xyzrgb_full[gt_mask_full][:, :3]
    gt_ds = voxel_downsample(gt_pts.astype(np.float32), voxel_size) if voxel_size > 0 else gt_pts
    return {
        "gt_xyzrgb_u": gt_xyzrgb_u,
        "gt_mask_u": gt_mask_u,
        "pred_xyzrgb": pred_xyzrgb,
        "pred_mask_base": pred_mask_base,
        "conf": conf,
        "gt_ds": gt_ds,
    }


def eval_scene(
    data: Dict[str, np.ndarray],
    variant: str,
    drop_quantile: float,
    icp_threshold: float,
    voxel_size: float,
    filter_scope: str = "prediction",
    use_weighted_icp: bool = False,
    use_weighted_fusion: bool = False,
    use_weighted_umeyama: bool = False,
    use_geo_filter: bool = False,
    use_robust_fusion_weights: bool = False,
    fusion_weight_gamma: float = 2.0,
    fusion_weight_min: float = 0.05,
    fusion_weight_max: float = 20.0,
) -> Optional[Dict[str, float]]:
    gt_xyzrgb_u = data["gt_xyzrgb_u"]
    gt_mask_u = data["gt_mask_u"]
    pred_xyzrgb = data["pred_xyzrgb"]
    pred_mask_base = data["pred_mask_base"]
    conf = data["conf"]
    gt_ds = data["gt_ds"]

    common_base = gt_mask_u & pred_mask_base
    if filter_scope == "prediction":
        pred_mask = _confidence_mask(pred_mask_base, conf, drop_quantile)
        common_mask = gt_mask_u & pred_mask
    elif filter_scope == "common_valid":
        common_mask = _confidence_mask(common_base, conf, drop_quantile)
    else:
        raise ValueError(f"Unknown point-filtering scope: {filter_scope}")

    if common_mask.sum() < 6:
        return None

    gt_pts_u = gt_xyzrgb_u[common_mask][:, :3]
    pred_pts_u = pred_xyzrgb[common_mask][:, :3]
    if conf is not None:
        conf_u = conf[common_mask]
    else:
        conf_u = np.ones((len(pred_pts_u),), dtype=np.float32)

    if use_weighted_umeyama:
        t_umeyama = weighted_umeyama(pred_pts_u, gt_pts_u, conf_u)
    else:
        t_umeyama = umeyama_registration(pred_pts_u, gt_pts_u)

    pred_after = apply_transform(pred_pts_u, t_umeyama)

    if use_weighted_fusion:
        pred_ds, pred_w = weighted_voxel_downsample(
            pred_after,
            conf_u,
            voxel_size,
            robust=use_robust_fusion_weights,
            weight_gamma=fusion_weight_gamma,
            weight_min=fusion_weight_min,
            weight_max=fusion_weight_max,
        )
    elif voxel_size > 0:
        pred_ds = voxel_downsample(pred_after.astype(np.float32), voxel_size)
        if use_weighted_icp and len(pred_ds) > 0 and len(pred_after) > 0:
            tree_raw = KDTree(pred_after[:, :3])
            _, nn = tree_raw.query(pred_ds[:, :3], workers=-1)
            pred_w = conf_u[nn].astype(np.float32)
        else:
            pred_w = np.ones((len(pred_ds),), dtype=np.float32)
    else:
        pred_ds = pred_after.astype(np.float32)
        pred_w = np.ones((len(pred_ds),), dtype=np.float32)

    if use_geo_filter:
        pred_ds = statistical_outlier_filter(pred_ds)
        pred_w = np.ones((len(pred_ds),), dtype=np.float32)

    if use_weighted_icp:
        t_icp = weighted_icp_registration(pred_ds, gt_ds, pred_w, icp_threshold=icp_threshold)
    else:
        t_icp = icp_registration(pred_ds, gt_ds, icp_threshold=icp_threshold)

    pred_eval = apply_transform(pred_ds, t_icp)
    out = eval_pc(pred_eval, gt_ds, thresholds=[0.05])
    out = {k: float(v) for k, v in out.items() if isinstance(v, (int, float, np.floating))}
    out.update(
        pred_points=float(len(pred_eval)),
        gt_points=float(len(gt_ds)),
        kept_common=float(common_mask.sum()),
        keep_ratio=float(common_mask.sum() / max(common_base.sum(), 1)),
        variant=variant,
    )
    return out


def _avg(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    out = {"num_scenes": len(rows)}
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    for k in keys:
        vals = [r[k] for r in rows if k in r and isinstance(r[k], (int, float))]
        out[k] = float(np.mean(vals))
    return out


def main(default_config: Path = DEFAULT_CONFIG) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(default_config))
    ap.add_argument("--method", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", default=None, help="Comma-separated override")
    ap.add_argument("--unc_filter_percent", type=float, default=None)
    ap.add_argument("--icp_threshold", type=float, default=None)
    ap.add_argument("--voxel_size", type=float, default=None)
    ap.add_argument("--fusion_weight_gamma", type=float, default=2.0)
    ap.add_argument("--fusion_weight_min", type=float, default=0.05)
    ap.add_argument("--fusion_weight_max", type=float, default=20.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = Path(args.config).resolve()
    cfg = ConfigManager(config_path)
    filter_cfg = cfg.get_raw("point_filtering", {})
    if not isinstance(filter_cfg, dict):
        raise TypeError("'point_filtering' must be a mapping")

    workspace = Path(cfg.get_workspace())
    if not workspace.is_absolute():
        workspace = (config_path.parent.parent / workspace).resolve()
    bss = BSSManager(workspace)
    dataset_name = str(filter_cfg.get("dataset", "neural_rgbd"))
    method_name = args.method or str(filter_cfg.get("method", "lingbot_map"))
    filter_percent = (
        float(args.unc_filter_percent)
        if args.unc_filter_percent is not None
        else float(filter_cfg.get("uncertainty_filter_percent", 20.0))
    )
    if not 0.0 <= filter_percent < 100.0:
        raise ValueError("uncertainty_filter_percent must be in [0, 100)")
    filter_scope = str(filter_cfg.get("filter_scope", "prediction"))
    if filter_scope not in {"prediction", "common_valid"}:
        raise ValueError("point_filtering.filter_scope must be 'prediction' or 'common_valid'")
    icp_threshold = (
        float(args.icp_threshold)
        if args.icp_threshold is not None
        else float(filter_cfg.get("icp_threshold", 0.1))
    )
    voxel_size = (
        float(args.voxel_size)
        if args.voxel_size is not None
        else float(filter_cfg.get("voxel_size", 4.0 / 512.0))
    )
    eval_area_budget = filter_cfg.get("eval_area_budget")
    resize_context = None
    if eval_area_budget is not None:
        resize_context = ResizeContext(
            mode="area_budget",
            area_budget=int(eval_area_budget),
            align=int(filter_cfg.get("align", 14)),
        )

    scenes = bss.list_scenes(dataset_name)
    configured_variants = filter_cfg.get("variants", ["raw", "unc_filter"])
    if args.variants is not None:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    elif isinstance(configured_variants, str):
        variants = [v.strip() for v in configured_variants.split(",") if v.strip()]
    else:
        variants = [str(v) for v in configured_variants]

    logging.info(
        "workspace=%s dataset=%s method=%s uncertainty_filter_percent=%.2f filter_scope=%s",
        workspace,
        dataset_name,
        method_name,
        filter_percent,
        filter_scope,
    )

    results: Dict[str, Dict[str, Dict[str, float]]] = {v: {} for v in variants}
    for scene in scenes:
        gt_art = bss.get_artifact(dataset_name, scene)
        pred_art = bss.get_artifact(dataset_name, scene, method_name)
        if not gt_art.is_complete() or not pred_art.is_complete():
            logging.warning("skip incomplete %s", scene)
            continue
        logging.info("scene %s", scene)
        gt_loader = BSSLoader(gt_art, resize_context=resize_context)
        pred_loader = BSSLoader(pred_art, resize_context=resize_context)
        logging.info("  loading shared point grids")
        data = load_scene_data(gt_loader, pred_loader, voxel_size)
        for variant in variants:
            kwargs = {}
            q = 0.0
            if variant == "raw":
                pass
            elif variant == "geo":
                kwargs["use_geo_filter"] = True
            elif variant == "unc_filter":
                q = filter_percent / 100.0
            elif variant.startswith("unc_filter"):
                q = float(variant.replace("unc_filter", "")) / 100.0
            elif variant == "unc_icp":
                kwargs["use_weighted_icp"] = True
            elif variant == "unc_fusion":
                kwargs["use_weighted_fusion"] = True
            elif variant == "unc_fusion_robust":
                kwargs["use_weighted_fusion"] = True
                kwargs["use_robust_fusion_weights"] = True
                kwargs["fusion_weight_gamma"] = args.fusion_weight_gamma
                kwargs["fusion_weight_min"] = args.fusion_weight_min
                kwargs["fusion_weight_max"] = args.fusion_weight_max
            elif variant == "unc_fusion_robust_icp":
                kwargs["use_weighted_fusion"] = True
                kwargs["use_weighted_icp"] = True
                kwargs["use_robust_fusion_weights"] = True
                kwargs["fusion_weight_gamma"] = args.fusion_weight_gamma
                kwargs["fusion_weight_min"] = args.fusion_weight_min
                kwargs["fusion_weight_max"] = args.fusion_weight_max
            elif variant == "unc_reg":
                kwargs["use_weighted_umeyama"] = True
                kwargs["use_weighted_icp"] = True
            else:
                raise ValueError(variant)
            logging.info("  variant %s", variant)
            metric = eval_scene(
                data,
                variant=variant,
                drop_quantile=q,
                icp_threshold=icp_threshold,
                voxel_size=voxel_size,
                filter_scope=filter_scope,
                **kwargs,
            )
            if metric is not None:
                results[variant][scene] = metric

    summary = {
        variant: _avg(list(per_scene.values()))
        for variant, per_scene in results.items()
        if per_scene
    }
    payload = {"per_scene": results, "summary": summary}
    output_filename = str(
        filter_cfg.get("output_filename", f"{dataset_name}_uncertainty_filter.json")
    )
    out = Path(args.out) if args.out else workspace / output_filename
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
