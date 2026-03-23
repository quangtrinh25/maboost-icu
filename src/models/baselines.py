"""
src/models/baselines.py
============================
Baseline models for offline comparison.
All inherit from TimeSeriesModel exactly like the 5 source files.

Models
------
1.  GRUDModel        – baselines.py  (bugfixed: n_feat, W_gh 1->d_model, _WPool NaN)
2.  TransformerModel – baselines.py  (bugfixed: nan_to_num on input + output)
3.  LSTMModel        – baselines.py  (bugfixed: nan_to_num on LSTM output)
4.  GRUD_TS          – grud.py       (JIT-scripted GRUDCell, per-gate dropout)
5.  SAND             – sand.py       (local attn mask radius r, DenseInterp)
6.  Strats           – strats.py     (CVE triplet emb, fusion attention)
7.  InterpNetModel   – interp_net.py (SCI + CCI + GRU, recon aux-loss)
8.  TCNModel         – tcn.py        (dilated causal conv, weight_norm)

References
----------
https://arxiv.org/pdf/1606.01865.pdf            GRU-D
https://arxiv.org/pdf/1711.03905.pdf            SAnD
https://openreview.net/pdf?id=r1efr3C9Ym        InterpNet
https://github.com/locuslab/TCN                 TCN
https://github.com/PeterChe1990/GRU-D           GRU-D cell
https://github.com/khirotaka/SAnD               SAnD
https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
"""
from __future__ import annotations
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch.nn.utils import weight_norm

from models import TimeSeriesModel


# ---------------------------------------------------------------------------
# Shared utilities  (from baselines.py — unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 1. GRUDModel  (baselines.py — bugfixed)
# ---------------------------------------------------------------------------

class GRUDModel(TimeSeriesModel):
    """
    GRU-D with input decay and hidden decay for irregular time-series.

    Bugs fixed (from baselines.py):
    - B, L, F = x.shape shadows F = torch.nn.functional
      → renamed to n_feat throughout forward()
    - W_gh(dt) dim mismatch: W_gh now Linear(1 → d_model)
    - _WPool: softmax(-inf) = NaN → fallback to uniform weights

    forward(x, tau, mask, demo, labels=None)
      x    : (B, L, V)
      tau  : (B, L)     elapsed time since last obs
      mask : (B, L, V)  observation mask
      demo : (B, D)
    """
    def __init__(self, args):
        super().__init__(args)
        self.d_model = args.hid_dim
        self.W_gx    = nn.Parameter(torch.randn(args.V) * 0.01)
        self.W_gh    = nn.Linear(1, args.hid_dim, bias=False)
        self.gru     = nn.GRU(args.V * 2, args.hid_dim,
                              getattr(args, 'num_layers', 2),
                              batch_first=True,
                              dropout=args.dropout if getattr(args, 'num_layers', 2) > 1 else 0.0)
        self.pool    = _WPool(args.hid_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, tau, mask, demo, labels=None):
        B, L, n_feat = x.shape   # renamed F→n_feat to avoid shadowing F module

        m = mask if mask is not None else torch.ones(B, L, n_feat, device=x.device)
        if m.dim() == 2:
            m = m.unsqueeze(-1).expand(B, L, n_feat)

        x_safe = torch.nan_to_num(x, nan=0.0)
        n_obs  = m.sum(dim=1).clamp(min=1.0)
        xm     = (x_safe * m).sum(dim=1) / n_obs   # (B, n_feat)

        h_prev = torch.zeros(self.gru.num_layers, B, self.d_model, device=x.device)
        x_prev = torch.zeros(B, n_feat, device=x.device)
        outputs = []

        for t in range(L):
            dt    = tau[:, t].unsqueeze(-1).float()           # (B, 1)
            gx    = torch.exp(-F.relu(self.W_gx) * dt)       # (B, n_feat)
            x_imp = (m[:, t] * x_safe[:, t]
                     + (1 - m[:, t]) * (gx * x_prev + (1 - gx) * xm))

            gh        = torch.exp(-F.relu(self.W_gh(dt)))     # (B, d_model)
            h_decayed = h_prev.clone()
            h_decayed[-1] = gh * h_prev[-1]

            inp = torch.cat([x_imp, m[:, t]], dim=-1).unsqueeze(1)  # (B,1,2*n_feat)
            out, h_new = self.gru(inp, h_decayed)
            outputs.append(out.squeeze(1))
            h_prev = h_new.detach()
            x_prev = x_imp.detach()

        seq      = torch.stack(outputs, dim=1)          # (B, L, d_model)
        ts_emb   = self.pool(self.dropout(seq), mask)
        demo_emb = self.demo_emb(demo)
        logits   = self.binary_head(torch.cat([ts_emb, demo_emb], dim=-1))[:, 0]
        return self.binary_cls_final(logits, labels)


# ---------------------------------------------------------------------------
# 2. TransformerModel  (baselines.py — bugfixed)
# ---------------------------------------------------------------------------

class TransformerModel(TimeSeriesModel):
    """
    Transformer with continuous-time sinusoidal encoding.
    nan_to_num guard on both input and output to prevent NaN predictions.

    forward(x, tau, mask, demo, labels=None)
    """
    def __init__(self, args):
        super().__init__(args)
        self.proj    = nn.Linear(args.V, args.hid_dim)
        self.pe_proj = nn.Linear(args.hid_dim, args.hid_dim)
        freq = torch.exp(
            torch.arange(0, args.hid_dim, 2).float() * (-math.log(10000.0) / args.hid_dim)
        )
        self.register_buffer("freq", freq)
        enc_layer = nn.TransformerEncoderLayer(
            args.hid_dim, getattr(args, 'num_heads', 4),
            dim_feedforward=args.hid_dim * 4,
            dropout=args.dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.tf      = nn.TransformerEncoder(enc_layer, getattr(args, 'num_layers', 3))
        self.norm    = nn.LayerNorm(args.hid_dim)
        self.pool    = _WPool(args.hid_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, tau, mask, demo, labels=None):
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
        h = torch.nan_to_num(h, nan=0.0)
        ts_emb   = self.pool(self.dropout(h), mask)
        demo_emb = self.demo_emb(demo)
        logits   = self.binary_head(torch.cat([ts_emb, demo_emb], dim=-1))[:, 0]
        return self.binary_cls_final(logits, labels)


# ---------------------------------------------------------------------------
# 3. LSTMModel  (baselines.py — bugfixed)
# ---------------------------------------------------------------------------

class LSTMModel(TimeSeriesModel):
    """
    LSTM with LayerNorm + attention-weighted pooling.
    nan_to_num guard on LSTM output.

    forward(x, tau, mask, demo, labels=None)
      tau accepted for API consistency but unused.
    """
    def __init__(self, args):
        super().__init__(args)
        self.proj    = nn.Linear(args.V, args.hid_dim)
        self.lstm    = nn.LSTM(args.hid_dim, args.hid_dim,
                               getattr(args, 'num_layers', 2),
                               batch_first=True,
                               dropout=args.dropout if getattr(args, 'num_layers', 2) > 1 else 0.0)
        self.norm    = nn.LayerNorm(args.hid_dim)
        self.pool    = _WPool(args.hid_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x, tau, mask, demo, labels=None):
        x    = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h, _ = self.lstm(self.proj(x))
        h    = torch.nan_to_num(h, nan=0.0)
        ts_emb   = self.pool(self.dropout(self.norm(h)), mask)
        demo_emb = self.demo_emb(demo)
        logits   = self.binary_head(torch.cat([ts_emb, demo_emb], dim=-1))[:, 0]
        return self.binary_cls_final(logits, labels)


# ---------------------------------------------------------------------------
# 4. GRUD_TS  (grud.py — JIT-scripted cell-level GRU-D)
# ---------------------------------------------------------------------------

class GRUDCell(jit.ScriptModule):
    """
    GRU-D cell with per-gate dropout masks, input decay (gamma_t),
    and hidden-state decay (gamma_ht). JIT-compiled for speed.

    Reference: https://arxiv.org/pdf/1606.01865.pdf
               https://github.com/PeterChe1990/GRU-D
    """
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.dropout     = dropout

        self.W_gamma   = nn.Parameter(torch.zeros(1, input_size),           requires_grad=True)
        self.b_gamma   = nn.Parameter(torch.zeros(1, input_size),           requires_grad=True)
        self.W_gamma_h = nn.Parameter(torch.zeros(input_size, hidden_size), requires_grad=True)
        self.b_gamma_h = nn.Parameter(torch.zeros(1, hidden_size),          requires_grad=True)

        self.W_z = nn.Parameter(torch.empty(input_size,  hidden_size), requires_grad=True)
        self.W_r = nn.Parameter(torch.empty(input_size,  hidden_size), requires_grad=True)
        self.W   = nn.Parameter(torch.empty(input_size,  hidden_size), requires_grad=True)
        self.U_z = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        self.U_r = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        self.U   = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=True)
        self.V_z = nn.Parameter(torch.empty(input_size,  hidden_size), requires_grad=True)
        self.V_r = nn.Parameter(torch.empty(input_size,  hidden_size), requires_grad=True)
        self.V   = nn.Parameter(torch.empty(input_size,  hidden_size), requires_grad=True)
        self.b_z = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        self.b_r = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)
        self.b   = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

        nn.init.xavier_uniform_(self.W_z); nn.init.xavier_uniform_(self.W_r)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.V_z); nn.init.xavier_uniform_(self.V_r)
        nn.init.xavier_uniform_(self.V)
        nn.init.orthogonal_(self.U_z); nn.init.orthogonal_(self.U_r)
        nn.init.orthogonal_(self.U)

        self.reset_dropout_masks()

    def reset_dropout_masks(self):
        self._dropout_mask           = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        self._recurrent_dropout_mask = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]
        self._masking_dropout_mask   = [torch.tensor(0), torch.tensor(0), torch.tensor(0)]

    @jit.script_method
    def forward(self, input, state):
        # type: (Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]
        x_t, m_t, delta_t = input       # (B,V) (B,V) (B,V)
        h_tm1, xprev      = state       # (B,d) (B,V)
        xprev = xprev * (1 - m_t) + m_t * x_t

        if self.dropout > 0 and self._dropout_mask[0].ndim == 0:
            self._dropout_mask = [
                F.dropout(torch.ones_like(x_t), self.dropout, self.training)
                for i in range(3)]
            self._recurrent_dropout_mask = [
                F.dropout(torch.ones_like(h_tm1), self.dropout, self.training)
                for i in range(3)]
            self._masking_dropout_mask = [
                F.dropout(torch.ones_like(m_t), self.dropout, self.training)
                for i in range(3)]

        gamma_t  = torch.exp(-torch.clip(self.W_gamma * delta_t + self.b_gamma, min=0))
        x_t      = m_t * x_t + (1 - m_t) * gamma_t * xprev
        gamma_ht = torch.exp(-torch.clip(
            torch.matmul(delta_t, self.W_gamma_h) + self.b_gamma_h, min=0))
        h_tm1 = gamma_ht * h_tm1

        if self.dropout > 0:
            x_t_z   = self._dropout_mask[0] * x_t
            x_t_r   = self._dropout_mask[1] * x_t
            x_t_h   = self._dropout_mask[2] * x_t
            m_t_z   = self._masking_dropout_mask[0] * m_t
            m_t_r   = self._masking_dropout_mask[1] * m_t
            m_t_h   = self._masking_dropout_mask[2] * m_t
            h_tm1_z = self._recurrent_dropout_mask[0] * h_tm1
            h_tm1_r = self._recurrent_dropout_mask[1] * h_tm1
            h_tm1_h = self._recurrent_dropout_mask[2] * h_tm1
        else:
            x_t_z = x_t_r = x_t_h       = x_t
            m_t_z = m_t_r = m_t_h       = m_t
            h_tm1_z = h_tm1_r = h_tm1_h = h_tm1

        z_t = torch.sigmoid(torch.matmul(x_t_z, self.W_z)
                            + torch.matmul(h_tm1_z, self.U_z)
                            + torch.matmul(m_t_z, self.V_z) + self.b_z)
        r_t = torch.sigmoid(torch.matmul(x_t_r, self.W_r)
                            + torch.matmul(h_tm1_r, self.U_r)
                            + torch.matmul(m_t_r, self.V_r) + self.b_r)
        h_t = torch.tanh(torch.matmul(x_t_h, self.W)
                         + torch.matmul(r_t * h_tm1_h, self.U)
                         + torch.matmul(m_t_h, self.V) + self.b)
        h_t = (1 - z_t) * h_tm1 + z_t * h_t
        return h_t, (h_t, xprev)


class _GRUDSequence(jit.ScriptModule):
    def __init__(self, *cell_args):
        super().__init__()
        self.cell = GRUDCell(*cell_args)

    @jit.script_method
    def forward(self, input, state):
        # type: (Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]) -> Tensor
        x_seq     = input[0].unbind(1)
        m_seq     = input[1].unbind(1)
        delta_seq = input[2].unbind(1)
        outputs   = torch.jit.annotate(List[torch.Tensor], [])
        self.cell.reset_dropout_masks()
        for i in range(len(x_seq)):
            out, state = self.cell((x_seq[i], m_seq[i], delta_seq[i]), state)
            outputs   += [out]
        return torch.stack(outputs).transpose(0, 1)   # (B, T, d)


class GRUD_TS(TimeSeriesModel):
    """
    JIT-scripted cell-level GRU-D with last-step readout.
    Per-gate dropout masks; mask embedding included in GRU gates (V_z/V_r/V).

    Reference: grud.py

    forward(x_t, m_t, delta_t, seq_len, demo, labels=None)
      x_t     : (B, T, V)
      m_t     : (B, T, V)  observation mask
      delta_t : (B, T, V)  time gaps per feature
      seq_len : (B,)       actual sequence lengths
      demo    : (B, D)
    """
    def __init__(self, args):
        super().__init__(args)
        self.grud    = _GRUDSequence(args.V, args.hid_dim, args.dropout)
        self.V       = args.V
        self.hid_dim = args.hid_dim
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x_t, m_t, delta_t, seq_len, demo, labels=None):
        bsz    = x_t.size(0)
        device = x_t.device
        initial_state = (torch.zeros((bsz, self.hid_dim), device=device),
                         torch.zeros((bsz, self.V),       device=device))
        ts_emb = self.grud((x_t, m_t, delta_t), initial_state)  # (B, T, d)
        bsz, max_len, d = ts_emb.size()
        index  = (seq_len - 1)[:, None, None].repeat((1, 1, d))
        ts_emb = torch.gather(ts_emb, 1, index)[:, 0, :]        # (B, d)
        ts_emb = self.dropout(ts_emb)
        demo_emb    = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        logits      = self.binary_head(ts_demo_emb)[:, 0]
        return self.binary_cls_final(logits, labels)


# ---------------------------------------------------------------------------
# 5. SAND  (sand.py — Self-Attention with Dense Interpolation)
# ---------------------------------------------------------------------------

class _SANDMultiHeadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        assert args.hid_dim % args.num_heads == 0
        self.dk        = args.hid_dim // args.num_heads
        self.num_heads = args.num_heads
        self.dropout   = args.dropout
        self.Wq = nn.Parameter(torch.empty(args.hid_dim, args.hid_dim)); nn.init.xavier_uniform_(self.Wq)
        self.Wk = nn.Parameter(torch.empty(args.hid_dim, args.hid_dim)); nn.init.xavier_uniform_(self.Wk)
        self.Wv = nn.Parameter(torch.empty(args.hid_dim, args.hid_dim)); nn.init.xavier_uniform_(self.Wv)
        self.Wo = nn.Linear(args.hid_dim, args.hid_dim, bias=False)

    def forward(self, x, mask):
        bsz, T, d = x.size()
        q = torch.matmul(x, self.Wq).view(bsz, T, self.num_heads, self.dk) / np.sqrt(self.dk)
        k = torch.matmul(x, self.Wk).view(bsz, T, self.num_heads, self.dk)
        v = torch.matmul(x, self.Wv).view(bsz, T, self.num_heads, self.dk)
        A = torch.einsum('bthd,blhd->bhtl', q, k) + mask    # bsz, h, T, T
        A = F.dropout(F.softmax(A, dim=-1), self.dropout)
        x = self.Wo(torch.einsum('bhtl,bthd->bhtd', A, v).reshape(bsz, T, d))
        return x


class _SANDFeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(args.hid_dim, args.hid_dim * 2, 1),
            nn.ReLU(),
            nn.Conv1d(args.hid_dim * 2, args.hid_dim, 1),
        )

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class _SANDTransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mha      = _SANDMultiHeadAttention(args)
        self.ffn      = _SANDFeedForward(args)
        self.norm_mha = nn.LayerNorm(args.hid_dim)
        self.norm_ffn = nn.LayerNorm(args.hid_dim)
        self.dropout  = args.dropout

    def forward(self, x, mask):
        x = self.norm_mha(x + F.dropout(self.mha(x, mask), self.dropout, self.training))
        x = self.norm_ffn(x + F.dropout(self.ffn(x),       self.dropout, self.training))
        return x


class _DenseInterpolation(nn.Module):
    def __init__(self, args):
        super().__init__()
        cols = torch.arange(args.M).reshape(1, args.M).float() / args.M
        rows = torch.arange(args.T).reshape(args.T, 1).float() / args.T
        W    = (1 - torch.abs(rows - cols)) ** 2
        self.W = nn.Parameter(W, requires_grad=False)

    def forward(self, x):
        bsz = x.size(0)
        return torch.matmul(x.transpose(1, 2), self.W).reshape(bsz, -1)


class SAND(TimeSeriesModel):
    """
    Self-Attention with Dense interpolation (SAnD).

    Reference: https://arxiv.org/pdf/1711.03905.pdf
               https://github.com/khirotaka/SAnD

    forward(ts, demo, labels=None)
      ts   : (B, T, V*3)  pre-concatenated [value, mask, delta]
      demo : (B, D)
    """
    def __init__(self, args):
        super().__init__(args)
        self.input_embedding     = nn.Conv1d(args.V * 3, args.hid_dim, 1)
        self.positional_encoding = nn.Parameter(
            torch.empty(1, args.T, args.hid_dim), requires_grad=True)
        nn.init.normal_(self.positional_encoding)

        indices    = torch.arange(args.T)
        local_mask = torch.logical_and(
            indices[None, :] <= indices[:, None],
            indices[None, :] >= indices[:, None] - args.r,
        ).float()
        attn_mask = (1 - local_mask) * torch.finfo(local_mask.dtype).min
        self.mask = nn.Parameter(attn_mask, requires_grad=False)

        self.dropout             = args.dropout
        self.transformer         = nn.ModuleList(
            [_SANDTransformerBlock(args) for _ in range(args.num_layers)])
        self.dense_interpolation = _DenseInterpolation(args)

    def forward(self, ts, demo, labels=None):
        ts_inp_emb = self.input_embedding(ts.permute(0, 2, 1)).permute(0, 2, 1)
        ts_inp_emb = ts_inp_emb + self.positional_encoding
        if self.dropout > 0:
            ts_inp_emb = F.dropout(ts_inp_emb, self.dropout, self.training)
        ts_hid_emb = ts_inp_emb
        for layer in self.transformer:
            ts_hid_emb = layer(ts_hid_emb, self.mask)
        ts_emb      = self.dense_interpolation(ts_hid_emb)
        demo_emb    = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        logits      = self.binary_head(ts_demo_emb)[:, 0]
        return self.binary_cls_final(logits, labels)


# ---------------------------------------------------------------------------
# 6. Strats  (strats.py — Triplet Transformer + Fusion Attention)
# ---------------------------------------------------------------------------

class CVE(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = int(np.sqrt(args.hid_dim))
        self.W1 = nn.Parameter(torch.empty(1, int_dim),           requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros(int_dim))
        self.W2 = nn.Parameter(torch.empty(int_dim, args.hid_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        self.activation = torch.tanh

    def forward(self, x):
        # x: bsz, max_len
        x = torch.unsqueeze(x, -1)
        x = torch.matmul(x, self.W1) + self.b1[None, None, :]  # bsz,max_len,int_dim
        x = self.activation(x)
        x = torch.matmul(x, self.W2)                           # bsz,max_len,hid_dim
        return x


class FusionAtt(nn.Module):
    def __init__(self, args):
        super().__init__()
        int_dim = args.hid_dim
        self.W = nn.Parameter(torch.empty(args.hid_dim, int_dim), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(int_dim))
        self.u = nn.Parameter(torch.empty(int_dim, 1),            requires_grad=True)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.u)
        self.activation = torch.tanh

    def forward(self, x, mask):
        # x: bsz, max_len, hid_dim
        att = torch.matmul(x, self.W) + self.b[None, None, :]  # bsz,max_len,int_dim
        att = self.activation(att)
        att = torch.matmul(att, self.u)[:, :, 0]               # bsz,max_len
        att = att + (1 - mask) * torch.finfo(att.dtype).min
        att = torch.softmax(att, dim=-1)                        # bsz,max_len
        return att


class _StratsTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.N  = args.num_layers
        self.d  = args.hid_dim
        self.h  = args.num_heads
        self.dk = self.d // self.h
        dff     = self.d * 2
        self.attention_dropout = args.attention_dropout
        self.dropout           = args.dropout

        self.Wq = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wk = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wv = nn.Parameter(self.init_proj((self.N, self.h, self.d, self.dk)), requires_grad=True)
        self.Wo = nn.Parameter(self.init_proj((self.N, self.dk * self.h, self.d)), requires_grad=True)
        self.W1 = nn.Parameter(self.init_proj((self.N, self.d, dff)),             requires_grad=True)
        self.b1 = nn.Parameter(torch.zeros((self.N, 1, 1, dff)),                  requires_grad=True)
        self.W2 = nn.Parameter(self.init_proj((self.N, dff, self.d)),             requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros((self.N, 1, 1, self.d)),               requires_grad=True)

    def init_proj(self, shape, gain=1):
        x = torch.rand(shape)
        fan_in_out = shape[-1] + shape[-2]
        scale = gain * np.sqrt(6 / fan_in_out)
        return x * 2 * scale - scale

    def forward(self, x, mask):
        # x: bsz, max_len, d   mask: bsz, max_len
        bsz, max_len, _ = x.size()
        mask       = mask[:, :, None] * mask[:, None, :]
        layer_mask = (1 - mask)[:, None, :, :] * torch.finfo(x.dtype).min
        for i in range(self.N):
            q = torch.einsum('bld,hde->bhle', x, self.Wq[i])
            k = torch.einsum('bld,hde->bhle', x, self.Wk[i])
            v = torch.einsum('bld,hde->bhle', x, self.Wv[i])
            A = torch.einsum('bhle,bhke->bhlk', q, k)
            if self.training:
                dropout_mask = (torch.rand_like(A) < self.attention_dropout
                                ).float() * torch.finfo(x.dtype).min
                layer_mask = layer_mask + dropout_mask
            A  = torch.softmax(A + layer_mask, dim=-1)
            v  = torch.einsum('bhkl,bhle->bkhe', A, v).reshape(bsz, max_len, -1)
            x  = (F.dropout(torch.matmul(v, self.Wo[i]), self.dropout, self.training) + x) / 2
            ff = torch.matmul(x, self.W1[i]) + self.b1[i]
            ff = F.gelu(ff)
            ff = torch.matmul(ff, self.W2[i]) + self.b2[i]
            x  = (F.dropout(ff, self.dropout, self.training) + x) / 2
        return x


class Strats(TimeSeriesModel):
    """
    STraTS: triplet (value + time + variable-id) Transformer.

    Reference: strats.py

    forward(values, times, varis, obs_mask, demo, labels=None)
      values   : (B, N_obs)  observed values
      times    : (B, N_obs)  timestamps
      varis    : (B, N_obs)  variable indices (int)
      obs_mask : (B, N_obs)  1 = valid observation
      demo     : (B, D)
    """
    def __init__(self, args):
        super().__init__(args)
        self.cve_time     = CVE(args)
        self.cve_value    = CVE(args)
        self.variable_emb = nn.Embedding(args.V, args.hid_dim)
        self.transformer  = _StratsTransformer(args)
        self.fusion_att   = FusionAtt(args)
        self.dropout      = args.dropout
        self.V            = args.V

    def forward(self, values, times, varis, obs_mask, demo, labels=None):
        bsz, max_obs = values.size()
        device = values.device

        if self.training:
            with torch.no_grad():
                var_mask = (torch.rand((bsz, self.V), device=device) <= self.dropout).int()
                for v in range(self.V):
                    mask_pos = (varis == v).int() * var_mask[:, v:v + 1]
                    obs_mask = obs_mask * (1 - mask_pos)

        time_emb    = self.cve_time(times)
        value_emb   = self.cve_value(values)
        vari_emb    = self.variable_emb(varis)
        triplet_emb = time_emb + value_emb + vari_emb
        triplet_emb = F.dropout(triplet_emb, self.dropout, self.training)

        contextual_emb    = self.transformer(triplet_emb, obs_mask)
        attention_weights = self.fusion_att(contextual_emb, obs_mask)[:, :, None]
        ts_emb            = (contextual_emb * attention_weights).sum(dim=1)

        demo_emb    = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        logits      = self.binary_head(ts_demo_emb)[:, 0]
        return self.binary_cls_final(logits, labels)


# ---------------------------------------------------------------------------
# 7. InterpNetModel  (interp_net.py — SCI + CCI + GRU + recon aux-loss)
# ---------------------------------------------------------------------------

class SingleChannelInterp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.kernel           = nn.Parameter(torch.zeros(1, 1, 1, args.V), requires_grad=True)
        self.hours_look_ahead = args.hours_look_ahead
        self.ref_points       = args.ref_points
        self.ref_t            = nn.Parameter(
            torch.linspace(0, self.hours_look_ahead, self.ref_points), requires_grad=False)

    def forward(self, x, m, t, h, reconstruction=False):
        if reconstruction:
            m     = h
            ref_t = t
        else:
            ref_t = self.ref_t[None, :]
        # x,m: bsz, T, V   t: bsz, T   ref_t: bsz(1), T'
        weights    = (t[:, :, None] - ref_t[:, None, :]) ** 2          # bsz,T,T'
        pos_kernel = torch.log(1 + torch.exp(self.kernel))             # 1,1,1,V
        weights    = pos_kernel * weights[:, :, :, None]               # bsz,T,T',V
        weights_lp = torch.exp(-weights)                               # eq (1)
        weights_lp = weights_lp * m[:, :, None, :]                    # bsz,T,T',V
        lambda_    = weights_lp.sum(dim=1)                             # bsz,T',V
        sigma      = (weights_lp * x[:, :, None, :]).sum(dim=1)
        sigma      = sigma / torch.clip(lambda_, min=1)
        if reconstruction:
            return sigma, lambda_
        weights_hp = torch.exp(-10.0 * weights)
        weights_hp = weights_hp * m[:, :, None, :]
        lambda_hp  = weights_hp.sum(dim=1)
        gamma      = (weights_hp * x[:, :, None, :]).sum(dim=1)
        gamma      = gamma / torch.clip(lambda_hp, min=1)
        return sigma, lambda_, gamma


class CrossChannelInterp(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rho = nn.Parameter(torch.eye(args.V)[None, None, :, :], requires_grad=True)

    def forward(self, sigma, lambda_, gamma=None):
        sigma   = sigma[:, :, :, None]             # bsz,T',V,1
        lambda_ = lambda_[:, :, :, None]           # bsz,T',V,1
        chi     = (self.rho * lambda_ * sigma).sum(dim=2)
        chi     = chi / torch.clip(lambda_.sum(dim=2), min=1)
        if gamma is None:
            return chi
        tau = gamma - chi
        return chi, tau


class InterpNetModel(TimeSeriesModel):
    """
    Interpolation Network: SCI + CCI + GRU + reconstruction auxiliary loss.

    Reference: https://openreview.net/pdf?id=r1efr3C9Ym

    forward(x, m, t, h, demo, labels=None)
      x : (B, T, V)   observed values
      m : (B, T, V)   observation mask
      t : (B, T)      timestamps
      h : (B, T, V)   held-out mask for reconstruction aux-loss
      demo : (B, D)
    """
    def __init__(self, args):
        super().__init__(args)
        self.sci     = SingleChannelInterp(args)
        self.cci     = CrossChannelInterp(args)
        self.gru     = nn.GRU(args.V * 3, args.hid_dim, batch_first=True,
                              dropout=args.dropout)
        self.dropout = nn.Dropout(args.dropout)

    def custom_loss(self, x, m, h, aux_output):
        loss      = (x - aux_output) ** 2
        loss_mask = m * (1 - h)
        loss      = loss * loss_mask
        loss      = loss.mean(dim=1) / torch.clip(loss_mask.sum(dim=1), min=1)  # bsz, V
        loss      = loss.sum(dim=1) / x.size()[-1]
        return loss.mean()

    def forward(self, x, m, t, h, demo, labels=None):
        sigma, lambda_, gamma = self.sci(x, m, t, h)
        chi, tau              = self.cci(sigma, lambda_, gamma)
        ts      = torch.cat((lambda_, chi, tau), dim=-1)             # bsz,T',3V
        ts_emb  = self.gru(ts)[1].reshape((ts.size()[0], -1))        # bsz,d
        ts_emb  = self.dropout(ts_emb)
        demo_emb    = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        logits      = self.binary_head(ts_demo_emb)[:, 0]
        if labels is None:
            return F.sigmoid(logits)
        main_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_class_weight)
        sigma, lambda_   = self.sci(x, m, t, h, reconstruction=True)
        aux_output       = self.cci(sigma, lambda_)                  # bsz,T,V
        aux_loss         = self.custom_loss(x, m, h, aux_output)
        return main_loss + aux_loss


# ---------------------------------------------------------------------------
# 8. TCNModel  (tcn.py — Temporal Convolutional Network)
# ---------------------------------------------------------------------------

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1    = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                             stride=stride, padding=padding, dilation=dilation))
        self.chomp1   = Chomp1d(padding)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2    = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                             stride=stride, padding=padding, dilation=dilation))
        self.chomp2   = Chomp1d(padding)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu       = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers     = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels   = num_inputs if i == 0 else num_channels[i - 1]
            out_channels  = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_TS(TimeSeriesModel):
    """
    Dilated causal Temporal Convolutional Network, last-step readout.

    Reference: tcn.py

    forward(ts, demo, labels=None)
      ts   : (B, T, V*3)  pre-concatenated [value, mask, delta]
      demo : (B, D)
    """
    def __init__(self, args):
        super().__init__(args)
        self.tcn = TemporalConvNet(args.V * 3, [args.hid_dim] * args.num_layers,
                                   args.kernel_size, args.dropout)

    def forward(self, ts, demo, labels=None):
        ts      = torch.permute(ts, (0, 2, 1))                       # N,V,T
        ts_emb  = self.tcn(ts)[:, :, -1]                             # N,d
        demo_emb    = self.demo_emb(demo)
        ts_demo_emb = torch.cat((ts_emb, demo_emb), dim=-1)
        logits      = self.binary_head(ts_demo_emb)[:, 0]
        return self.binary_cls_final(logits, labels)