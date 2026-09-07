"""χ²₂ calibration diagnostics for 2D flow σ.

For a well-calibrated 2D diagonal Gaussian:
    Mahalanobis² = r_u² / σ_u² + r_v² / σ_v²    ~  χ²₂

χ²₂ stats: mean = 2, median ≈ 1.386, q95 ≈ 5.991, q99 ≈ 9.210.
"""

from __future__ import annotations

from typing import Dict

import torch


# Inverse CDF of χ²₂ at common probabilities
_CHI2_2_QUANTILES = {
    0.10: 0.2107,
    0.20: 0.4463,
    0.30: 0.7133,
    0.40: 1.0217,
    0.50: 1.3863,
    0.60: 1.8326,
    0.70: 2.4079,
    0.80: 3.2189,
    0.90: 4.6052,
    0.95: 5.9915,
    0.99: 9.2103,
}


def reliability_stats_2d(
    mahalanobis_sq: torch.Tensor,
    max_samples: int = 1_000_000,
) -> Dict[str, float]:
    """Compute 2D χ²₂ calibration diagnostics for a tensor of Mahal² values.

    Args:
        mahalanobis_sq: any-shape tensor; will be flattened.
        max_samples:    torch.quantile errors out > 16M elements — subsample first.

    Returns:
        Flat dict with empirical/target quantiles, mean, ECE.
    """
    m = mahalanobis_sq.flatten().float().cpu()
    n_total = m.numel()
    if n_total > max_samples:
        idx = torch.randperm(n_total)[:max_samples]
        m = m[idx]
    out: Dict[str, float] = {
        "n_samples": int(n_total),
        "n_for_quantile": int(m.numel()),
        "mean_empirical": float(m.mean()),
        "mean_target": 2.0,
    }

    probs = [0.10, 0.50, 0.90, 0.95, 0.99]
    ece = 0.0
    for p in probs:
        emp_q = float(torch.quantile(m, p))
        theo_q = _CHI2_2_QUANTILES[round(p, 2)]
        out[f"q{int(p*100):02d}_empirical"] = emp_q
        out[f"q{int(p*100):02d}_target"] = theo_q
        # Empirical CDF at theoretical quantile — for calibrated head, equals p.
        emp_cdf = float((m <= theo_q).float().mean())
        out[f"cdf_at_q{int(p*100):02d}"] = emp_cdf
        ece += abs(emp_cdf - p)
    out["ECE"] = ece / len(probs)
    return out


def format_stats(stats: Dict[str, float]) -> str:
    """One-line summary."""
    return (
        f"n={stats['n_samples']:,d} | "
        f"mean={stats['mean_empirical']:.3f} (tgt 2.0) | "
        f"q50={stats['q50_empirical']:.2f} (tgt {stats['q50_target']:.2f}) | "
        f"q95={stats['q95_empirical']:.2f} (tgt {stats['q95_target']:.2f}) | "
        f"ECE={stats['ECE']:.3f}"
    )
