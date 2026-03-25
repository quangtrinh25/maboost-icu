"""
src/models/mamba_encoder.py
============================
MaBoost encoder v7 — GRUDImputer + SCI + CCI + Mamba.
Fix: SCI.forward B,T,F → B,T,n_feat (avoids shadowing F=nn.functional).
All public function names and signatures unchanged.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


def _register_nan_hooks(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, Mamba):
            for p in m.parameters():
                p.register_hook(
                    lambda g: torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                    if g is not None else g
                )


class GRUDImputer(nn.Module):
    def __init__(self, d_input: int):
        super().__init__()
        self.log_w = nn.Parameter(torch.full((d_input,), -3.0))

    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        rate   = F.softplus(self.log_w)
        tau_h  = (tau / 3600.0).clamp(min=1e-4).unsqueeze(-1)
        gamma  = torch.exp(-rate * tau_h)
        n_obs  = mask.sum(dim=[0, 1]).clamp(min=1.0)
        x_mean = (x * mask).sum(dim=[0, 1]) / n_obs
        return mask * x + (1.0 - mask) * (gamma * x + (1.0 - gamma) * x_mean)


class SCI(nn.Module):
    def __init__(self, n_feat: int, ref_points: int = 32,
                 window_hours: float = 48.0):
        super().__init__()
        self.ref_points   = ref_points
        self.window_hours = window_hours
        self.log_kernel   = nn.Parameter(torch.zeros(n_feat))

    def forward(
        self,
        x_hat: torch.Tensor,
        mask:  torch.Tensor,
        tau:   torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FIX: n_feat instead of F to avoid shadowing torch.nn.functional
        B, T, n_feat = x_hat.shape
        device  = x_hat.device
        kappa   = F.softplus(self.log_kernel)   # (n_feat,)

        t_hours = tau.cumsum(dim=1) / 3600.0
        ref_t = torch.linspace(
            0, self.window_hours, self.ref_points, device=device
        ).unsqueeze(0).expand(B, -1)

        dist2  = (t_hours.unsqueeze(2) - ref_t.unsqueeze(1)) ** 2
        dist2F = dist2.unsqueeze(-1).expand(B, T, self.ref_points, n_feat)
        kappa4 = kappa.view(1, 1, 1, n_feat)
        maskF  = mask.unsqueeze(2).expand(B, T, self.ref_points, n_feat)
        x4     = x_hat.unsqueeze(2).expand(B, T, self.ref_points, n_feat)

        w_lp = torch.exp(-kappa4 * dist2F) * maskF
        w_hp = torch.exp(-10.0 * kappa4 * dist2F) * maskF

        lam   = w_lp.sum(dim=1)
        sigma = (w_lp * x4).sum(dim=1) / lam.clamp(min=1)
        gamma = (w_hp * x4).sum(dim=1) / w_hp.sum(dim=1).clamp(min=1)

        return sigma, lam, gamma


class CCI(nn.Module):
    def __init__(self, n_feat: int):
        super().__init__()
        self.rho = nn.Parameter(torch.eye(n_feat).unsqueeze(0).unsqueeze(0))

    def forward(
        self,
        sigma: torch.Tensor,
        lam:   torch.Tensor,
        gamma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sigma4 = sigma.unsqueeze(-1)
        lam4   = lam.unsqueeze(-1)
        num    = (self.rho * lam4 * sigma4).sum(dim=-2)
        den    = (self.rho * lam4).sum(dim=-2).clamp(1)
        chi    = num / den
        tau_res = gamma - chi
        return chi, tau_res


class TimeEmbedding(nn.Module):
    def __init__(self, d_out: int, use_circadian: bool = True):
        super().__init__()
        self.use_circadian = use_circadian
        self.gap_net = nn.Sequential(
            nn.Linear(1, d_out), nn.SiLU(), nn.Linear(d_out, d_out)
        )
        if use_circadian:
            self.circ_proj = nn.Linear(4, d_out, bias=False)

    def forward(self, tau: torch.Tensor,
                t_abs: Optional[torch.Tensor] = None) -> torch.Tensor:
        tau_h = (tau / 3600.0).clamp(min=1e-4)
        out   = self.gap_net(torch.log1p(tau_h).unsqueeze(-1))
        if self.use_circadian and t_abs is not None:
            hour = (t_abs / 3600.0) % 24.0
            dow  = (t_abs / 86400.0) % 7.0
            circ = torch.stack([
                torch.sin(2 * math.pi * hour / 24.0),
                torch.cos(2 * math.pi * hour / 24.0),
                torch.sin(2 * math.pi * dow  /  7.0),
                torch.cos(2 * math.pi * dow  /  7.0),
            ], dim=-1)
            out = out + self.circ_proj(circ)
        return out


class MambaBlock(nn.Module):
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


class MultiResolutionPooling(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.W    = nn.Linear(d, d // 2)
        self.u    = nn.Linear(d // 2, 1, bias=False)
        self.proj = nn.Linear(d * 4, d)

    def _valid_mask(self, mask, B, L, device):
        if mask is None:
            return torch.ones(B, L, device=device)
        return (mask.max(-1).values > 0).float() if mask.dim() == 3 else (mask > 0).float()

    def forward(self, h, mask=None):
        return self.proj(self._pool_all(h, mask))

    def extract(self, h, mask=None):
        return self._pool_all(h, mask)

    def _pool_all(self, h, mask=None):
        B, L, d  = h.shape
        valid    = self._valid_mask(mask, B, L, h.device)
        last_idx = (valid * torch.arange(L, device=h.device).float()).argmax(dim=1)
        h_last   = h[torch.arange(B), last_idx]
        n_obs    = valid.sum(dim=1, keepdim=True).clamp(min=1)
        h_mean   = (h * valid.unsqueeze(-1)).sum(dim=1) / n_obs
        h_masked = h.masked_fill(valid.unsqueeze(-1).expand_as(h) == 0, float("-inf"))
        h_max    = torch.nan_to_num(h_masked.max(dim=1).values, neginf=0.0)
        scores   = self.u(torch.tanh(self.W(h)))
        scores   = scores.masked_fill((valid == 0).unsqueeze(-1), float("-inf"))
        alpha    = torch.softmax(scores, dim=1)
        if torch.isnan(alpha).any():
            alpha = torch.where(torch.isnan(alpha),
                                alpha.new_full(alpha.shape, 1.0 / L), alpha)
        h_attn = (alpha * h).sum(dim=1)
        return torch.cat([h_last, h_mean, h_max, h_attn], dim=-1)


class MambaEncoder(nn.Module):
    """
    ICU encoder v7: x(B,T,F), tau(B,T) → z_T(B,d)
    Public API identical to v6.
    """
    def __init__(self, d_input: int, d_model: int = 128, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, n_layers: int = 3,
                 n_heads: int = 4, dropout: float = 0.1, topk: int = 5,
                 use_circadian: bool = True,
                 ref_points: int = 32, window_hours: float = 48.0):
        super().__init__()
        self.d_input    = d_input
        self.ref_points = ref_points

        self.imputer    = GRUDImputer(d_input)
        self.sci        = SCI(d_input, ref_points, window_hours)
        self.cci        = CCI(d_input)
        self.input_proj = nn.Sequential(
            nn.Linear(d_input * 3, d_model), nn.LayerNorm(d_model)
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
        h = self._encode(x, tau, mask, t_abs)
        return self.pool(h, mask=None)

    def extract_features(self, x: torch.Tensor, tau: torch.Tensor,
                         mask: Optional[torch.Tensor] = None,
                         t_abs: Optional[torch.Tensor] = None,
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        h         = self._encode(x, tau, mask, t_abs)
        z_multi   = self.pool.extract(h, mask=None)
        raw_stats = self._raw_stats(x, mask)
        return z_multi, raw_stats

    def _encode(self, x: torch.Tensor, tau: torch.Tensor,
                mask: Optional[torch.Tensor],
                t_abs: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask_3d = torch.ones_like(x)
        elif mask.dim() == 2:
            mask_3d = mask.unsqueeze(-1).expand_as(x)
        else:
            mask_3d = mask

        x_hat = self.imputer(x, tau, mask_3d)
        sigma, lam, gamma = self.sci(x_hat, mask_3d, tau)
        chi, tau_res = self.cci(sigma, lam, gamma)

        sci_out = torch.cat([lam, chi, tau_res], dim=-1)
        h = self.t_fuse(torch.cat([
            self.input_proj(sci_out),
            self.t_emb(self._ref_tau(tau)),
        ], dim=-1))

        ref_tau = self._ref_tau(tau)
        for layer in self.layers:
            h = layer(h, ref_tau)

        h = self.norm(h)
        return torch.nan_to_num(h, nan=0.0, posinf=1.0, neginf=-1.0)

    def _ref_tau(self, tau: torch.Tensor) -> torch.Tensor:
        B  = tau.shape[0]
        dt = 48.0 * 3600.0 / self.ref_points
        return torch.full((B, self.ref_points), dt,
                          dtype=tau.dtype, device=tau.device)

    def _raw_stats(self, x: torch.Tensor,
                   mask: Optional[torch.Tensor]) -> torch.Tensor:
        B, L, F = x.shape
        if mask is None:
            m3 = torch.ones_like(x)
        elif mask.dim() == 2:
            m3 = mask.unsqueeze(-1).expand_as(x)
        else:
            m3 = mask

        n_obs_raw = m3.sum(dim=1)
        never_obs = (n_obs_raw == 0)
        n_obs     = n_obs_raw.clamp(min=1.0)

        last_idx = torch.zeros(B, F, dtype=torch.long, device=x.device)
        for t in range(L):
            obs_t    = (m3[:, t, :] > 0)
            last_idx = torch.where(obs_t, torch.full_like(last_idx, t), last_idx)
        last_val = x[torch.arange(B).unsqueeze(-1), last_idx,
                     torch.arange(F).unsqueeze(0)]

        x_obs    = x * m3
        mean_val = x_obs.sum(dim=1) / n_obs
        x_max    = x.masked_fill(m3 == 0, float("-inf"))
        max_val  = torch.nan_to_num(x_max.max(dim=1).values, nan=0.0, neginf=0.0)
        sq_mean  = (x_obs ** 2).sum(dim=1) / n_obs
        std_val  = (sq_mean - mean_val ** 2).clamp(min=0.0).sqrt()
        miss_r   = 1.0 - m3.mean(dim=1)

        nan_val  = float("nan")
        last_val = last_val.masked_fill(never_obs, nan_val)
        mean_val = mean_val.masked_fill(never_obs, nan_val)
        max_val  = max_val.masked_fill(never_obs,  nan_val)
        std_val  = std_val.masked_fill(never_obs,  nan_val)

        return torch.cat([last_val, mean_val, max_val, std_val, miss_r], dim=-1)


class DualHeadMamba(nn.Module):
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
            z = z + self.static_emb(torch.nan_to_num(x_static, nan=0.0))
        return self.mort_head(z), self.los_head(z).squeeze(-1)