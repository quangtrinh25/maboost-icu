"""
src/models/baselines.py
=======================
Baseline models for offline comparison.
All share: forward(x, tau, mask) -> (mort_logits, los_pred)

Bugs fixed
----------
- GRU-D:       B, L, F = x.shape shadows  F = torch.nn.functional
               → renamed to n_feat throughout GRUDModel.forward
               → also fixes W_gh(dt) dim mismatch: W_gh now Linear(1→d_model)
- _WPool:      softmax(-inf everywhere) = NaN when all positions masked
               → fallback to uniform weights
- Transformer: nan_to_num guard on output z before _WPool
- LSTM:        nan_to_num guard on output z before _WPool
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _WPool(nn.Module):
    """Attention-weighted pooling with NaN-safe softmax."""
    def __init__(self, d: int):
        super().__init__()
        self.W = nn.Linear(d, d // 2)
        self.u = nn.Linear(d // 2, 1, bias=False)

    def forward(self, h: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        s = self.u(torch.tanh(self.W(h)))
        if mask is not None:
            valid = (mask.max(-1).values > 0) if mask.dim() == 3 else mask.bool()
            s = s.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(s, 1)
        # All positions masked → softmax(-inf) = NaN → replace with uniform
        alpha = torch.where(
            torch.isnan(alpha),
            torch.full_like(alpha, 1.0 / h.size(1)),
            alpha,
        )
        return (alpha * h).sum(1)


class _DualHead(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.mort = nn.Linear(d, 2)
        self.los  = nn.Linear(d, 1)

    def forward(self, z: torch.Tensor):
        return self.mort(z), self.los(z).squeeze(-1)


class GRUDModel(nn.Module):
    """
    GRU-D with input decay and hidden decay for irregular time-series.

    Key fix: renamed loop variable F→n_feat to avoid shadowing the
    module-level  F = torch.nn.functional  import.
    Previously  B, L, F = x.shape  made F an int, so F.relu() crashed
    with "'int' object has no attribute 'relu'".

    Also: W_gh now maps (1 → d_model) not (d_input → d_model) because
    the hidden decay gate takes the scalar time gap dt, not the full input.
    """
    def __init__(self, d_input: int, d_model: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.W_gx = nn.Parameter(torch.randn(d_input) * 0.01)
        # Hidden decay: scalar dt → d_model gate (was d_input→d_model, wrong)
        self.W_gh = nn.Linear(1, d_model, bias=False)
        self.gru  = nn.GRU(d_input * 2, d_model, n_layers,
                           batch_first=True,
                           dropout=dropout if n_layers > 1 else 0.0)
        self.pool = _WPool(d_model)
        self.head = _DualHead(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        B, L, n_feat = x.shape   # renamed F→n_feat to avoid shadowing F module

        m = mask if mask is not None else torch.ones(B, L, n_feat, device=x.device)
        if m.dim() == 2:
            m = m.unsqueeze(-1).expand(B, L, n_feat)

        # Safe batch mean: observed positions only (NaN-safe)
        x_safe = torch.nan_to_num(x, nan=0.0)
        n_obs  = m.sum(dim=1).clamp(min=1.0)
        xm     = (x_safe * m).sum(dim=1) / n_obs   # (B, n_feat)

        h_prev = torch.zeros(self.gru.num_layers, B, self.d_model, device=x.device)
        x_prev = torch.zeros(B, n_feat, device=x.device)
        outputs = []

        for t in range(L):
            dt = tau[:, t].unsqueeze(-1).float()           # (B, 1)
            # Input decay γ_x — F here is torch.nn.functional (not shadowed)
            gx    = torch.exp(-F.relu(self.W_gx) * dt)    # (B, n_feat)
            x_imp = (m[:, t] * x_safe[:, t]
                     + (1 - m[:, t]) * (gx * x_prev + (1 - gx) * xm))

            # Hidden decay γ_h = exp(-relu(W_gh · dt))
            gh        = torch.exp(-F.relu(self.W_gh(dt)))  # (B, d_model)
            h_decayed = h_prev.clone()
            h_decayed[-1] = gh * h_prev[-1]

            inp = torch.cat([x_imp, m[:, t]], dim=-1).unsqueeze(1)  # (B,1,2*n_feat)
            out, h_new = self.gru(inp, h_decayed)
            outputs.append(out.squeeze(1))

            h_prev = h_new.detach()
            x_prev = x_imp.detach()

        seq = torch.stack(outputs, dim=1)     # (B, L, d_model)
        z   = self.pool(self.drop(seq), mask)
        return self.head(z)


class TransformerModel(nn.Module):
    """
    Transformer with continuous-time sinusoidal encoding.
    nan_to_num guard on both input and output to prevent NaN predictions.
    """
    def __init__(self, d_input: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.proj    = nn.Linear(d_input, d_model)
        self.pe_proj = nn.Linear(d_model, d_model)
        freq = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("freq", freq)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.tf   = nn.TransformerEncoder(enc_layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pool = _WPool(d_model)
        self.head = _DualHead(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        t  = tau.unsqueeze(-1).float()
        pe = torch.zeros(B, L, self.freq.numel() * 2, device=x.device)
        pe[:, :, 0::2] = torch.sin(t * self.freq)
        pe[:, :, 1::2] = torch.cos(t * self.freq)

        h = self.proj(x) + self.pe_proj(pe)

        kpm = None
        if mask is not None:
            m   = mask if mask.dim() == 2 else mask.max(-1).values
            kpm = (m == 0)

        h = self.norm(self.tf(h, src_key_padding_mask=kpm))
        # Guard output NaN (e.g. from fully-masked sequences in LayerNorm)
        h = torch.nan_to_num(h, nan=0.0)
        z = self.pool(self.drop(h), mask)
        return self.head(z)


class LSTMModel(nn.Module):
    def __init__(self, d_input: int, d_model: int = 128,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_input, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(d_model)
        self.pool = _WPool(d_model)
        self.head = _DualHead(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, tau: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        x    = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h, _ = self.lstm(self.proj(x))
        h    = torch.nan_to_num(h, nan=0.0)   # guard output
        z    = self.pool(self.drop(self.norm(h)), mask)
        return self.head(z)