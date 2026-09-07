"""2D diagonal Gaussian NLL on optical-flow residual + β-NLL re-weighting.

Per-pixel NLL of residual r_uv = (r_u, r_v) ~ N(0, diag(σ_u², σ_v²)):

    NLL_pixel = r_u² / σ_u² + r_v² / σ_v² + log σ_u² + log σ_v²

β-NLL (Seitzer et al. 2022) re-weights each pixel's loss by `(σ²_eq)^β` with
σ² detached.  For the 2D diagonal case we use the geometric mean variance:
σ²_eq = exp((log σ_u² + log σ_v²) / 2).

Setting β = 0.5 dampens the gradient on low-σ pixels (preventing σ collapse in
flat regions) and amplifies it on high-σ pixels (rare matching-difficulty
outliers we want the head to actually learn).

Mahalanobis² (returned for χ²₂ calibration) is NOT clamped/scaled — it's the
honest per-pixel χ²₂ statistic used to evaluate calibration externally.
"""

from __future__ import annotations

import torch


def flow_nll_2d(
    log_sigma_u_sq: torch.Tensor,
    log_sigma_v_sq: torch.Tensor,
    r_u: torch.Tensor,
    r_v: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    beta_nll: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2D diagonal Gaussian NLL.

    Args:
        log_sigma_u_sq:  [...] — log σ_u² per pixel (already soft-floored upstream)
        log_sigma_v_sq:  [...] — log σ_v²
        r_u:             [...] — u-component of residual (pixel units)
        r_v:             [...] — v-component
        valid_mask:      [...] — bool, loss only counted where True
        beta_nll:        β coefficient for Seitzer-style re-weighting (0 → vanilla NLL)

    Returns:
        loss:            scalar — mean NLL over valid pixels (with β-NLL scaling)
        mahalanobis_sq:  [...] — r_u²/σ_u² + r_v²/σ_v² (detached, for χ²₂ calibration)
    """
    sigma_u_sq = torch.exp(log_sigma_u_sq)
    sigma_v_sq = torch.exp(log_sigma_v_sq)

    mahal_sq = r_u.pow(2) / sigma_u_sq + r_v.pow(2) / sigma_v_sq
    logdet = log_sigma_u_sq + log_sigma_v_sq        # log det Σ = log(σ_u² σ_v²)
    nll = mahal_sq + logdet

    if beta_nll > 0.0:
        with torch.no_grad():
            # σ²_eq = exp(mean(log σ_u², log σ_v²))  → (σ²_eq)^β = exp(β * mean(logs))
            scale = torch.exp(beta_nll * (log_sigma_u_sq + log_sigma_v_sq) / 2.0)
        nll = nll * scale

    if valid_mask is not None:
        m_f = valid_mask.float()
        denom = m_f.sum().clamp(min=1.0)
        loss = (nll * m_f).sum() / denom
    else:
        loss = nll.mean()

    return loss, mahal_sq.detach()
