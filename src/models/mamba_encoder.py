"""
src/models/mamba_encoder.py
============================
MaBoost encoder — designed as a strong feature extractor for XGBoost.

Stage 1: encoder trains jointly with MLP heads → z_T(d) for gradient signal
Stage 2: encoder.extract() returns RICH multi-resolution features for XGBoost

Design principle
----------------
XGBoost is best at splitting on interpretable numeric features. z_T(128) is
a compressed bottleneck — XGBoost can't see "inside" it. We expose 3 layers:

  Layer 1 — Mamba hidden states (temporal context, 4 × d_model)
    h_last    : hidden state at last OBSERVED position  (current patient state)
    h_mean    : mean over observed positions            (average trajectory)
    h_max     : max over positions                      (peak activation)
    h_weighted: attention-weighted summary              (important moments)

  Layer 2 — Raw temporal statistics per feature (5 × d_input)
    last_val  : last observed value per feature         (current vitals/labs)
    mean_val  : mean of observed values                 (average over ICU stay)
    max_val   : max of observed values                  (worst value)
    std_val   : std of observed values                  (variability)
    miss_rate : fraction of timesteps not observed      (data availability)

    *** CRITICAL: features never observed return NaN (not 0.0) so that
        XGBoost can use its sparsity-aware split finder.  Confusing
        "not measured" with "value = 0" destroyed XGBoost-flat performance. ***

  Layer 3 — Static features (42)  [passed through unchanged]

  Layer 4 — Circadian features (4 per timestep, optional)
    sin/cos of hour-of-day and day-of-week — encode absolute clinical context
    (e.g., 08:00 routine blood draw vs 03:00 acute deterioration).
    Requires absolute timestamp tensors; disabled when t_abs is None.

Total Stage 2 features: 4×128 + 5×d_input + 42 [+ optional circadian]

Stability (mamba_simple.py has no fp32 accumulators unlike CUDA kernel):
  • _register_nan_hooks — zeros NaN gradients before Adam m2 accumulation
  • MambaBlock.forward — .float() cast before Mamba, cast back after
  • nan_to_num guard on encoder output
  • Mamba params never touched by weight init
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# ---------------------------------------------------------------------------
# NaN gradient hook — mandatory for mamba_simple.py
# ---------------------------------------------------------------------------

def _register_nan_hooks(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, Mamba):
            for p in m.parameters():
                p.register_hook(
                    lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                    if g is not None else g
                )


# ---------------------------------------------------------------------------
# GRUDImputer — vectorised time-decay imputation (no recurrence, no NaN)
# ---------------------------------------------------------------------------

class GRUDImputer(nn.Module):
    """
    γ_j = exp(−softplus(w_j) × τ_hours)
    x̂_j = mask_j·x_j + (1−mask_j)·(γ_j·x_j + (1−γ_j)·μ_batch_j)

    Provides clean, continuously-valued input to the Mamba encoder.
    Missing values filled with time-decayed batch mean → encoder sees
    meaningful values at every position rather than zeros.

    NOTE: This imputer lives inside MambaEncoder intentionally (Stage 1).
    For fair benchmarking of LSTM / Transformer baselines, a shared
    imputation step must run *before* the DataLoader — see etl/pipeline.py.
    """
    def __init__(self, d_input: int):
        super().__init__()
        self.log_w = nn.Parameter(torch.full((d_input,), -3.0))

    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        rate  = F.softplus(self.log_w)
        tau_h = (tau / 3600.0).clamp(min=1e-4).unsqueeze(-1)
        gamma = torch.exp(-rate * tau_h)
        n_obs = mask.sum(dim=[0, 1]).clamp(min=1.0)
        x_mean = (x * mask).sum(dim=[0, 1]) / n_obs
        return mask * x + (1.0 - mask) * (gamma * x + (1.0 - gamma) * x_mean)


# ---------------------------------------------------------------------------
# TimeEmbedding — gap-based + optional circadian encoding
# ---------------------------------------------------------------------------

class TimeEmbedding(nn.Module):
    """
    Two complementary time signals:

    1. Gap signal (always active):
       log1p(τ_hours) → 2-layer MLP → (B, L, d_out)
       Captures "how long since last measurement".

    2. Circadian signal (optional, requires absolute timestamps):
       [sin(2π·h/24), cos(2π·h/24), sin(2π·dow/7), cos(2π·dow/7)] → Linear → d_out
       Captures "08:00 routine draw vs 03:00 acute event".
       Pass t_abs (B, L) in seconds-since-epoch to activate.

    Both signals are summed when circadian is active.
    """
    def __init__(self, d_out: int, use_circadian: bool = True):
        super().__init__()
        self.use_circadian = use_circadian
        self.gap_net = nn.Sequential(
            nn.Linear(1, d_out), nn.SiLU(), nn.Linear(d_out, d_out)
        )
        if use_circadian:
            # 4 raw circadian features → d_out
            self.circ_proj = nn.Linear(4, d_out, bias=False)

    def forward(self, tau: torch.Tensor,
                t_abs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        tau   : (B, L) gap in seconds since previous observation
        t_abs : (B, L) absolute UNIX timestamp in seconds (optional)
        """
        tau_h = (tau / 3600.0).clamp(min=1e-4)
        out   = self.gap_net(torch.log1p(tau_h).unsqueeze(-1))   # (B, L, d)

        if self.use_circadian and t_abs is not None:
            # Hours-of-day and day-of-week in [0, 2π]
            hour = (t_abs / 3600.0) % 24.0
            dow  = (t_abs / 86400.0) % 7.0
            circ = torch.stack([
                torch.sin(2 * math.pi * hour / 24.0),
                torch.cos(2 * math.pi * hour / 24.0),
                torch.sin(2 * math.pi * dow  /  7.0),
                torch.cos(2 * math.pi * dow  /  7.0),
            ], dim=-1)                                            # (B, L, 4)
            out = out + self.circ_proj(circ)

        return out


# ---------------------------------------------------------------------------
# MambaBlock — pre-norm + float32 + ZOH time-decay gate + residual
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    ZOH-style time-decay gate:
        gate = exp(−softplus(r) × τ_hours)   r ∈ ℝ^d, init −3
        τ→0: gate=1 (dense obs → full Mamba output)
        τ→∞: gate=0 (sparse obs → residual bypass)

    Gradient note: for gaps > ~72h, gate < 0.05.  This is intentional
    (old context should decay) but can weaken gradients through long gaps.
    The residual bypass (x + drop(h * gate)) preserves gradient flow even
    when gate → 0.
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm      = nn.LayerNorm(d_model)
        self.mamba     = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)
        self.log_decay = nn.Parameter(torch.full((d_model,), -3.0))
        self.drop      = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        h     = self.mamba(self.norm(x).float()).to(x.dtype)
        tau_h = (tau / 3600.0).clamp(min=1e-4).unsqueeze(-1)
        gate  = torch.exp(-F.softplus(self.log_decay) * tau_h)
        return x + self.drop(h * gate)


# ---------------------------------------------------------------------------
# MultiResolutionPooling — exposes 4 views of hidden sequence to XGBoost
# ---------------------------------------------------------------------------

class MultiResolutionPooling(nn.Module):
    """
    Stage 1 (training): returns z_T(d) via attention pooling for MLP heads.
    Stage 2 (feature extraction): returns (B, 4*d) multi-resolution features.

    Four pooling modes:
      last     — hidden state at last observed position (current patient state)
      mean     — mean over observed positions (average trajectory)
      max      — max over L dimension (peak activation / worst state)
      weighted — attention-weighted (u^T tanh(W h) softmax, NaN-safe)
    """
    def __init__(self, d: int):
        super().__init__()
        self.W    = nn.Linear(d, d // 2)
        self.u    = nn.Linear(d // 2, 1, bias=False)
        self.proj = nn.Linear(d * 4, d)   # Stage 2 concat → d for Stage 1 heads

    def _valid_mask(self, mask: Optional[torch.Tensor],
                    B: int, L: int, device) -> torch.Tensor:
        if mask is None:
            return torch.ones(B, L, device=device)
        return (mask.max(-1).values > 0).float() if mask.dim() == 3 else (mask > 0).float()

    def forward(self, h: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns z_T (B, d) for Stage 1 training."""
        return self.proj(self._pool_all(h, mask))

    def extract(self, h: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns (B, 4*d) multi-resolution features for Stage 2."""
        return self._pool_all(h, mask)

    def _pool_all(self, h: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, d = h.shape
        valid   = self._valid_mask(mask, B, L, h.device)          # (B, L)

        # 1. Last observed position
        last_idx = (valid * torch.arange(L, device=h.device).float()).argmax(dim=1)
        h_last   = h[torch.arange(B), last_idx]                   # (B, d)

        # 2. Mean over observed positions
        n_obs  = valid.sum(dim=1, keepdim=True).clamp(min=1)
        h_mean = (h * valid.unsqueeze(-1)).sum(dim=1) / n_obs      # (B, d)

        # 3. Max over all positions
        h_max  = h.max(dim=1).values                               # (B, d)

        # 4. Attention-weighted pooling (NaN-safe softmax)
        scores = self.u(torch.tanh(self.W(h)))                     # (B, L, 1)
        scores = scores.masked_fill((valid == 0).unsqueeze(-1), float("-inf"))
        alpha  = torch.softmax(scores, dim=1)
        # Fallback: uniform weights if all positions masked
        if torch.isnan(alpha).any():
            alpha = torch.where(
                torch.isnan(alpha),
                alpha.new_full(alpha.shape, 1.0 / L),
                alpha,
            )
        h_attn = (alpha * h).sum(dim=1)                            # (B, d)

        return torch.cat([h_last, h_mean, h_max, h_attn], dim=-1)  # (B, 4*d)


# ---------------------------------------------------------------------------
# MambaEncoder
# ---------------------------------------------------------------------------

class MambaEncoder(nn.Module):
    """
    ICU time-series encoder: x(B,L,F), tau(B,L) → z_T(B,d)

    Pipeline
    --------
    GRUDImputer  → clean, continuously-valued input
    InputProj    → Linear(F→d) + LayerNorm
    TimeFuse     → concat([proj, TimeEmb(τ, t_abs?)]) → Linear → h
    MambaBlock×n → selective SSM with ZOH time-decay gate
    LayerNorm    → stable activations
    nan_to_num   → guard before pooling
    MultiResPool → z_T for Stage 1 / (B, 4d) for Stage 2

    Also exposes compute_raw_stats() for per-feature temporal statistics
    that Stage 2 passes to XGBoost alongside encoder features.

    Circadian encoding
    ------------------
    Pass t_abs (B, L) absolute UNIX timestamps in seconds to forward() /
    extract_features() to activate hour-of-day / day-of-week embeddings.
    When t_abs is None the TimeEmbedding falls back to gap-only mode.
    """
    def __init__(self, d_input: int, d_model: int = 128, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1, topk: int = 5,
                 use_circadian: bool = True):
        super().__init__()
        self.d_input    = d_input
        self.imputer    = GRUDImputer(d_input)
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model), nn.LayerNorm(d_model)
        )
        self.t_emb  = TimeEmbedding(d_model, use_circadian=use_circadian)
        self.t_fuse = nn.Linear(d_model * 2, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = MultiResolutionPooling(d_model)

    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                t_abs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Stage 1: returns z_T (B, d_model)."""
        h = self._encode(x, tau, mask, t_abs)
        return self.pool(h, mask)

    def extract_features(self, x: torch.Tensor, tau: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         t_abs: Optional[torch.Tensor] = None,
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Stage 2: returns (z_multi, raw_stats).
          z_multi  : (B, 4*d_model) multi-resolution encoder features
          raw_stats: (B, 5*d_input) per-feature temporal statistics

        raw_stats contains NaN for features that were never observed in a stay.
        This is intentional — XGBoost uses its sparsity-aware split finder on
        NaN values; forcing them to 0 conflates "not measured" with "value=0".
        """
        h         = self._encode(x, tau, mask, t_abs)
        z_multi   = self.pool.extract(h, mask)
        raw_stats = self._raw_stats(x, mask)
        return z_multi, raw_stats

    # ------------------------------------------------------------------
    def _encode(self, x: torch.Tensor, tau: torch.Tensor,
                mask: Optional[torch.Tensor],
                t_abs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask_3d = torch.ones(*x.shape, device=x.device)
        elif mask.dim() == 2:
            mask_3d = mask.unsqueeze(-1).expand_as(x)
        else:
            mask_3d = mask

        x_hat = self.imputer(x, tau, mask_3d)
        h     = self.t_fuse(
            torch.cat([self.input_proj(x_hat),
                       self.t_emb(tau, t_abs)], dim=-1)
        )
        for layer in self.layers:
            h = layer(h, tau)
        h = self.norm(h)
        return torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)

    def _raw_stats(self, x: torch.Tensor,
                   mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Per-feature temporal statistics — directly interpretable by XGBoost.

        5 statistics × d_input features = (B, 5 * d_input):
          last_val  — last observed value  (current state of vital/lab)
          mean_val  — mean of observed values (average over ICU stay)
          max_val   — maximum observed value  (worst-case)
          std_val   — std of observed values  (variability / instability)
          miss_rate — fraction of timesteps unobserved (data availability)

        Features that were NEVER observed in a stay return NaN for
        last/mean/max/std so XGBoost can route them via its built-in
        sparsity-aware split finder (not confused with "value = 0").
        miss_rate stays 1.0 — that IS informative signal.
        """
        B, L, F = x.shape
        if mask is None:
            m3 = torch.ones_like(x)
        elif mask.dim() == 2:
            m3 = mask.unsqueeze(-1).expand_as(x)
        else:
            m3 = mask

        # Raw observation count before clamping — used to tag never-observed features
        n_obs_raw = m3.sum(dim=1)                    # (B, F)  ∈ [0, L]
        never_obs = (n_obs_raw == 0)                 # (B, F)  True → feature absent
        n_obs     = n_obs_raw.clamp(min=1.0)         # safe denominator

        # last observed value per feature (vectorised over t dimension)
        last_idx = torch.zeros(B, F, dtype=torch.long, device=x.device)
        for t in range(L):
            obs_t    = (m3[:, t, :] > 0)            # (B, F)
            last_idx = torch.where(obs_t, torch.full_like(last_idx, t), last_idx)
        last_val = x[torch.arange(B).unsqueeze(-1),
                     last_idx,
                     torch.arange(F).unsqueeze(0)]   # (B, F)

        # mean over observed (zero-fill unobserved positions before sum)
        x_obs    = x * m3                            # (B, L, F)
        mean_val = x_obs.sum(dim=1) / n_obs          # (B, F)

        # max over observed positions only (-inf sentinel → nan_to_num)
        x_max   = x.masked_fill(m3 == 0, float("-inf"))
        max_val = x_max.max(dim=1).values            # (B, F)
        max_val = torch.nan_to_num(max_val, nan=0.0, neginf=0.0)

        # std: E[x²] − E[x]² (Welford-equivalent, numerically stable)
        sq_mean  = (x_obs ** 2).sum(dim=1) / n_obs
        variance = (sq_mean - mean_val ** 2).clamp(min=0.0)
        std_val  = variance.sqrt()

        miss_rate = 1.0 - m3.mean(dim=1)            # (B, F)

        # Mark never-observed features as NaN so XGBoost routes them correctly.
        # miss_rate intentionally kept at 1.0 — that IS real signal.
        nan_val = float("nan")
        last_val = last_val.masked_fill(never_obs, nan_val)
        mean_val = mean_val.masked_fill(never_obs, nan_val)
        max_val  = max_val.masked_fill(never_obs,  nan_val)
        std_val  = std_val.masked_fill(never_obs,  nan_val)

        return torch.cat(
            [last_val, mean_val, max_val, std_val, miss_rate], dim=-1
        )                                            # (B, 5*F)


# ---------------------------------------------------------------------------
# DualHeadMamba — Stage 1 training wrapper
# ---------------------------------------------------------------------------

class DualHeadMamba(nn.Module):
    """
    Encoder + static branch + MLP heads for Stage-1 joint training.

    The learned static_emb (Linear→ReLU→Dropout→Linear) captures non-linear
    interactions between demographics/comorbidities and the temporal state z.
    For Stage 2, pass this module as `static_proj` to extract_features() so
    the same learned transformation is applied rather than raw x_static.
    """
    def __init__(self, d_input: int, d_model: int = 128,
                 d_static: int = 0, **enc_kw):
        super().__init__()
        self.encoder    = MambaEncoder(d_input, d_model, **enc_kw)
        drop            = enc_kw.get("dropout", 0.1)
        self.has_static = d_static > 0
        if self.has_static:
            self.static_emb = nn.Sequential(
                nn.Linear(d_static, d_model), nn.ReLU(),
                nn.Dropout(drop), nn.Linear(d_model, d_model),
            )
        self.mort_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(d_model // 2, 2),
        )
        self.los_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                x_static: Optional[torch.Tensor] = None,
                t_abs: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, tau, mask, t_abs)
        if self.has_static and x_static is not None:
            z = z + self.static_emb(
                torch.nan_to_num(x_static, nan=0.0)
            )
        return self.mort_head(z), self.los_head(z).squeeze(-1)