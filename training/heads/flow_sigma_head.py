"""FlowSigmaHead: per-pixel 2D matching uncertainty for streaming reconstruction.

Architectural intent
--------------------
For each frame t (t > 0 in a sequence), the head predicts (σ_u, σ_v) — the
per-pixel uncertainty of where this pixel re-projects in frame t-1.  By
construction this is **Markov**: σ_t only depends on (t, t-1) tokens, no
sliding-window cache.  The aggregator already accumulates long-range geometry
context implicitly via its 24-layer alternating attention; the σ-head's job is
the *one dimension* that the aggregator doesn't explicitly model: **adjacent-
frame matching difficulty** (texture repeats, fast motion, occlusion-adjacent).

Architecture
------------
Mirrors LoG-VGGT's cross-window pattern but localised to a *single* layer:

      frame t aggregator output           frame t-1 aggregator output
         (last selected_idx, raw)              (last selected_idx, raw)
                  │                                  │
                  └──────► CrossAttnBlock ◄──────────┘
                            Q=t, KV=t-1
                                 │
                  ┌─────────► fused t tokens ──┐
                  │                            │
        agg_list[:-1]                 (replaces agg_list[-1])
                  └────► DPTHead (multi-scale) ─┘
                                 │
                          [B, S, 256, H, W]
                                 │
                          conv 3x3 → relu → conv 1x1 → 2 ch
                                 │
                       (log σ_u², log σ_v²)
                       with softplus floor

We only modify the **deepest** selected_idx block (= block 23 output) — same as
LoG-VGGT putting cross-window attention on the 4 selected blocks (the deepest
of which is block 23).  Other 3 multi-scale layers go through DPT unchanged.

Init: last conv weight = 0, bias = solve_for_target so initial (σ_u, σ_v) ≈ 1px
(matches typical reprojection error magnitude on VKitti scene scale).
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lingbot_map.heads.dpt_head import DPTHead


def _solve_init_bias(log_sigma_init: float, log_sigma_min: float) -> float:
    """Solve b s.t. softplus(b) + log_sigma_min == log_sigma_init.

    softplus(b) = target   ⇒   b = log(exp(target) - 1)
    """
    target = log_sigma_init - log_sigma_min
    if target <= 0:
        return -5.0
    return float(torch.log(torch.expm1(torch.tensor(target))).item())


class CrossAttnBlock(nn.Module):
    """Pre-norm cross-attention with residual and LayerScale.

    Q comes from current frame t patch tokens; K/V come from previous frame t-1
    patch tokens.  Both already include the aggregator's long-range context, so
    the cross-attn here learns to *attend selectively* to the past patch most
    relevant for matching the current patch.

    Args:
        dim: token channel dim (= 2 * embed_dim for the GCT aggregator, i.e. 2048)
        num_heads: 8 by default (head_dim = 256 for dim=2048)
        init_values: LayerScale init for residual (small → cross-attn output is gated low at start)
    """

    def __init__(
        self,
        dim: int = 2048,
        num_heads: int = 8,
        init_values: float = 0.01,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
            bias=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)
        # Layer scale on residual — start with tiny modulation so initial behaviour
        # equals "no cross-attn", and head can learn to enable it gradually.
        self.gamma = nn.Parameter(torch.full((dim,), init_values))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        Args:
            q:  [B, N_q, dim]  current frame patch tokens
            kv: [B, N_kv, dim] previous frame patch tokens
        Returns:
            [B, N_q, dim] — q + cross-attn(q, kv) with residual + LayerScale
        """
        q_norm = self.norm_q(q)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q_norm, kv_norm, kv_norm, need_weights=False)
        attn_out = self.proj_drop(attn_out)
        return q + self.gamma * attn_out


class FlowSigmaHead(nn.Module):
    """2-channel diagonal per-pixel σ_uv head for adjacent-frame reprojection.

    Args:
        dim_in:         token channel dim (= 2 * embed_dim of the aggregator).
        patch_size:     ViT patch size (14 for ViT-L).
        num_heads:      cross-attn heads (8).
        log_sigma_min:  soft floor on log σ via softplus (default -2 → σ ≥ ~0.14 px).
        log_sigma_init: starting log σ (default 0 → σ ≈ 1 px, matches typical VKitti
                        adjacent-frame reprojection error).
        feature_dim:    DPT intermediate features (256).
        patch_start_idx: where patch tokens begin in the aggregator output
                        (6 = 1 camera + 4 register + 1 scale for GCT aggregator).

    Forward inputs:
        agg_tokens_list_t:   list of 4 tensors [B, 1, P, dim_in], current frame at
                             selected_idx [4, 11, 17, 23].
        agg_tokens_list_tm1: list of 4 tensors [B, 1, P, dim_in], previous frame.
        images_t:            [B, 1, 3, H, W] current frame image (for DPT pos embed).
        patch_start_idx:     int — patch tokens begin at this index (specials before).

    Outputs:
        log_sigma_u_sq: [B, 1, H, W] — soft floor at log_sigma_min
        log_sigma_v_sq: [B, 1, H, W]
    """

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 14,
        num_heads: int = 8,
        log_sigma_min: float = -2.0,
        log_sigma_init: float = 0.0,
        feature_dim: int = 256,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.log_sigma_min = log_sigma_min
        self.log_sigma_init = log_sigma_init

        # Cross-attention: Q from frame t, KV from frame t-1
        self.cross_attn = CrossAttnBlock(
            dim=dim_in,
            num_heads=num_heads,
            init_values=0.01,
        )

        # DPT trunk in feature-only mode (returns [B, S, feature_dim, H, W])
        self.dpt = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=2,                # ignored under feature_only=True
            activation="linear",
            conf_activation="linear",
            feature_only=True,
            features=feature_dim,
        )

        # Decoder: 256 → 32 → 2 (log σ_u², log σ_v²)
        self.decoder = nn.Sequential(
            nn.Conv2d(feature_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, kernel_size=1),
        )

        # Zero-init last conv weight; bias chosen so softplus(bias) + min = init.
        last_conv = self.decoder[-1]
        nn.init.zeros_(last_conv.weight)
        diag_bias_value = _solve_init_bias(log_sigma_init, log_sigma_min)
        with torch.no_grad():
            last_conv.bias.fill_(diag_bias_value)

    def forward(
        self,
        agg_tokens_list_t: List[torch.Tensor],
        agg_tokens_list_tm1: List[torch.Tensor],
        images_t: torch.Tensor,
        patch_start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the head.

        Args:
            agg_tokens_list_t:   list of 4 tensors, each [B, 1, P, dim_in].
            agg_tokens_list_tm1: list of 4 tensors, each [B, 1, P, dim_in].
            images_t:            [B, 1, 3, H, W]
            patch_start_idx:     int — index where patch tokens begin.

        Returns:
            log_sigma_u_sq: [B, 1, H, W]
            log_sigma_v_sq: [B, 1, H, W]
        """
        assert len(agg_tokens_list_t) == len(agg_tokens_list_tm1), \
            "current and previous aggregator outputs must have same depth"

        # ---- Cross-attn on the deepest selected block (= block 23 output, idx -1) ----
        t_deepest = agg_tokens_list_t[-1]            # [B, 1, P, dim_in]
        tm1_deepest = agg_tokens_list_tm1[-1]        # [B, 1, P, dim_in]

        B, S_t, P, C = t_deepest.shape
        assert S_t == 1, f"FlowSigmaHead expects single-frame query (got S={S_t})"

        # Strip special tokens — cross-attn over patches only (specials are global
        # signal already, won't gain from matching across frames at the patch level).
        t_patch = t_deepest[:, 0, patch_start_idx:]      # [B, P-ns, C]
        tm1_patch = tm1_deepest[:, 0, patch_start_idx:]   # [B, P-ns, C]

        fused_patch = self.cross_attn(t_patch, tm1_patch)  # [B, P-ns, C]

        # Reconstruct full token tensor (specials unchanged)
        t_fused = t_deepest.clone()
        t_fused[:, 0, patch_start_idx:] = fused_patch     # only patch tokens modified

        # Replace the deepest layer in the multi-scale list (other 3 layers unchanged).
        new_list = list(agg_tokens_list_t)
        new_list[-1] = t_fused

        # ---- DPT multi-scale features ----
        feat = self.dpt(new_list, images=images_t, patch_start_idx=patch_start_idx)
        # feat: [B, S=1, feature_dim, H, W]

        B, S, Cf, H, W = feat.shape
        feat_flat = feat.reshape(B * S, Cf, H, W)

        # ---- 2-channel decoder ----
        raw = self.decoder(feat_flat)                     # [B*S, 2, H, W]
        raw = raw.view(B, S, 2, H, W)                     # [B, S, 2, H, W]

        # Soft floor: log_sigma² = min + softplus(raw)
        # raw → -∞: log_sigma² → min (floor)
        # raw → +∞: log_sigma² → raw (linear)
        log_sigma_u_sq = self.log_sigma_min + F.softplus(raw[:, :, 0])  # [B, S, H, W]
        log_sigma_v_sq = self.log_sigma_min + F.softplus(raw[:, :, 1])
        return log_sigma_u_sq, log_sigma_v_sq


class FlowSigmaHeadCached(FlowSigmaHead):
    """FlowSigmaHead with a 1-frame RNN-style state cache.

    Cache holds **only the previous frame's cross-attn-enhanced tokens** (1
    state vector), but because each enhanced_{t-1} itself was computed from
    enhanced_{t-2}, info propagates implicitly along the chain:

        t=1:  KV = aggregator(t-1) (raw, no cache yet)    → enhanced_1
        t=2:  KV = enhanced_1   (from cache)              → enhanced_2
        t=3:  KV = enhanced_2   (chain: also knows e_1)   → enhanced_3
        ...

    The chain length grows with the sequence — early-frame info reaches frame
    t through O(t) steps of cross-attention.  This is the LoG-VGGT pattern
    applied at the σ-head level.

    Cache is detached at insertion → no gradient flows backward through more
    than one step.  Each frame's loss updates cross-attn weights via its own
    forward; the chain provides *context*, not extra gradient signal.

    Caller responsibilities:
      - `clear_cache()` at the start of every independent sequence (each
        training sample / val pass).
      - Call forward(...) for frames in strict temporal order.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Single-frame state: previous frame's enhanced patch tokens (detached)
        # shape: [B, P-num_special, dim_in], or None at sequence start.
        self._prev_enhanced: torch.Tensor | None = None

    def clear_cache(self) -> None:
        self._prev_enhanced = None

    @property
    def has_cache(self) -> bool:
        return self._prev_enhanced is not None

    def forward(
        self,
        agg_tokens_list_t: List[torch.Tensor],
        agg_tokens_list_tm1: List[torch.Tensor],
        images_t: torch.Tensor,
        patch_start_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cached forward.

        Logic:
          - First frame of sequence (cache empty): use raw `agg_tokens_list_tm1`
            as K/V (Markov fallback — no previous enhanced state exists yet).
          - Subsequent frames: use cached `enhanced_{t-1}` as K/V (chain).

        After forward, the *current* frame's enhanced tokens (detached) replace
        the cache (single-frame state).
        """
        assert len(agg_tokens_list_t) == len(agg_tokens_list_tm1), \
            "current and previous aggregator outputs must have same depth"

        t_deepest = agg_tokens_list_t[-1]
        tm1_deepest = agg_tokens_list_tm1[-1]
        B, S_t, P, C = t_deepest.shape
        assert S_t == 1, f"expects single-frame query (got S={S_t})"

        t_patch = t_deepest[:, 0, patch_start_idx:]      # [B, P-ns, C]

        # K/V source: cached enhanced_{t-1} if available, else raw tm1 (first frame).
        if self._prev_enhanced is not None:
            kv = self._prev_enhanced                       # [B, P-ns, C]
        else:
            kv = tm1_deepest[:, 0, patch_start_idx:]       # [B, P-ns, C]

        fused_patch = self.cross_attn(t_patch, kv)         # [B, P-ns, C]

        # Update single-frame state cache (detached — no BPTT)
        self._prev_enhanced = fused_patch.detach()

        # Reconstruct full token tensor and run DPT + decoder (same as parent)
        t_fused = t_deepest.clone()
        t_fused[:, 0, patch_start_idx:] = fused_patch
        new_list = list(agg_tokens_list_t)
        new_list[-1] = t_fused

        feat = self.dpt(new_list, images=images_t, patch_start_idx=patch_start_idx)
        B, S, Cf, H, W = feat.shape
        feat_flat = feat.reshape(B * S, Cf, H, W)
        raw = self.decoder(feat_flat).view(B, S, 2, H, W)
        log_sigma_u_sq = self.log_sigma_min + F.softplus(raw[:, :, 0])
        log_sigma_v_sq = self.log_sigma_min + F.softplus(raw[:, :, 1])
        return log_sigma_u_sq, log_sigma_v_sq
