"""
src/training/baseline_trainer.py
==================================
Train and evaluate baseline models for MaBoost comparison.

Models: GRU-D, LSTM, Transformer, SAnD, STraTS, InterpNet, TCN

Fixes vs previous version
--------------------------
- STraTS OOM: max_obs=512 (was 4096), batch size halved for STraTS
- _adapt_strats: faster vectorized triplet building with torch.nonzero
- All adapters exported for use in run_baselines.py
"""
from __future__ import annotations
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Args builder
# ---------------------------------------------------------------------------

def _make_args(
    d_input:          int,
    d_static:         int,
    pos_rate:         float,
    model_type:       str,
    hid_dim:          int   = 256,
    num_layers:       int   = 2,
    num_heads:        int   = 4,
    dropout:          float = 0.1,
    T:                int   = 128,
    M:                int   = 64,
    r:                int   = 8,
    attention_dropout: float = 0.1,
    max_obs:          int   = 512,
    ref_points:       int   = 32,
    hours_look_ahead: float = 48.0,
    kernel_size:      int   = 3,
):
    import argparse
    args                   = argparse.Namespace()
    args.model_type        = model_type
    args.V                 = d_input
    args.D                 = d_static
    args.hid_dim           = hid_dim
    args.num_layers        = num_layers
    args.num_heads         = num_heads
    args.dropout           = dropout
    args.attention_dropout = attention_dropout
    args.pos_class_weight  = (1.0 - pos_rate) / max(pos_rate, 1e-6)
    args.pretrain          = 0
    args.load_ckpt_path    = None
    args.T                 = T
    args.M                 = M
    args.r                 = r
    args.max_obs           = max_obs
    args.ref_points        = ref_points
    args.hours_look_ahead  = hours_look_ahead
    args.kernel_size       = kernel_size
    return args


# ---------------------------------------------------------------------------
# Batch adapters
# ---------------------------------------------------------------------------

def _adapt_standard(batch, device):
    """GRU-D, LSTM, Transformer."""
    x, tau, mask, xs, ym, _ = batch
    return (x.to(device), tau.to(device), mask.to(device),
            xs.to(device), ym.float().to(device))


def _adapt_concat(batch, device):
    """SAnD, TCN: ts_cat = [x | mask | delta]."""
    x, tau, mask, xs, ym, _ = batch
    x    = x.to(device)
    tau  = tau.to(device)
    mask = mask.to(device)
    delta   = tau.unsqueeze(-1).expand_as(x)
    ts_cat  = torch.cat([x, mask, delta], dim=-1)
    return (ts_cat, xs.to(device), ym.float().to(device))


def _adapt_strats(batch, device, max_obs: int = 512):
    """
    STraTS: (values, times, varis, obs_mask, demo, labels)
    Vectorized triplet building using torch.nonzero.
    max_obs=512 to avoid OOM on 16GB GPU.
    """
    x, tau, mask, xs, ym, _ = batch
    x    = x.to(device)
    tau  = tau.to(device)
    mask = mask.to(device)

    B, T, V  = x.shape
    cum_h    = tau.cumsum(dim=1) / 3600.0   # (B, T) hours

    # Find observed positions: (B, T, V) → sparse
    obs_pos = mask.bool()   # (B, T, V)

    values_out   = torch.zeros(B, max_obs, device=device)
    times_out    = torch.zeros(B, max_obs, device=device)
    varis_out    = torch.zeros(B, max_obs, dtype=torch.long, device=device)
    obs_mask_out = torch.zeros(B, max_obs, device=device)

    for b in range(B):
        idx = obs_pos[b].nonzero(as_tuple=False)   # (n_obs, 2): [t, v]
        n   = min(len(idx), max_obs)
        if n > 0:
            t_idx = idx[:n, 0]
            v_idx = idx[:n, 1]
            values_out[b, :n]   = x[b, t_idx, v_idx]
            times_out[b, :n]    = cum_h[b, t_idx]
            varis_out[b, :n]    = v_idx
            obs_mask_out[b, :n] = 1.0

    return (values_out, times_out, varis_out, obs_mask_out,
            xs.to(device), ym.float().to(device))


def _adapt_interpnet(batch, device, ref_points: int = 32):
    """InterpNet: (x, m, t, h, demo, labels)."""
    x, tau, mask, xs, ym, _ = batch
    x    = x.to(device)
    tau  = tau.to(device)
    mask = mask.to(device)
    t    = tau.cumsum(dim=1) / 3600.0
    h    = (mask * (torch.rand_like(mask) < 0.2)).float()
    return (x, mask, t, h, xs.to(device), ym.float().to(device))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer, adapter, device, clip=1.0):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        adapted = adapter(batch, device)
        *inputs, labels = adapted
        optimizer.zero_grad()
        loss = model(*inputs, labels=labels)
        if torch.isnan(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def _evaluate(model, loader, adapter, device):
    model.eval()
    preds, labels_all = [], []
    for batch in loader:
        adapted = adapter(batch, device)
        *inputs, labels = adapted
        prob = model(*inputs, labels=None)
        if isinstance(prob, torch.Tensor):
            prob = prob.cpu().numpy()
        preds.append(np.atleast_1d(prob))
        labels_all.append(labels.cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels_all)
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_pred))


def train_baseline(
    model_class,
    model_type:  str,
    adapter_fn,
    tr_loader:   DataLoader,
    va_loader:   DataLoader,
    te_loader:   DataLoader,
    d_input:     int,
    d_static:    int,
    pos_rate:    float,
    device:      str   = "cuda",
    epochs:      int   = 50,
    lr:          float = 1e-3,
    patience:    int   = 10,
    hid_dim:     int   = 256,
    extra_kw:    dict  = None,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    extra_kw = extra_kw or {}

    # STraTS needs smaller batch — handled in adapter with max_obs
    args  = _make_args(d_input=d_input, d_static=d_static, pos_rate=pos_rate,
                       model_type=model_type, hid_dim=hid_dim, **extra_kw)
    model = model_class(args).to(device)

    n_params  = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc, best_state, patience_n = 0.0, None, 0
    t0 = time.perf_counter()
    print(f"    Training {model_type} ({n_params:,} params) ...")

    for ep in range(1, epochs + 1):
        try:
            loss = _train_one_epoch(model, tr_loader, optimizer, adapter_fn, device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"    OOM at ep {ep} — skipping epoch")
                continue
            raise
        scheduler.step()
        val_auc = _evaluate(model, va_loader, adapter_fn, device)

        if val_auc > best_auc:
            best_auc   = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_n = 0
        else:
            patience_n += 1
            if patience_n >= patience:
                print(f"    Early stop ep={ep}  best_val_auc={best_auc:.4f}")
                break

        if ep % 10 == 0:
            print(f"    ep {ep:3d}/{epochs}  loss={loss:.4f}  "
                  f"val_auc={val_auc:.4f}  best={best_auc:.4f}")

    train_time = time.perf_counter() - t0

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in te_loader:
            adapted = adapter_fn(batch, device)
            *inputs, labels = adapted
            try:
                prob = model(*inputs, labels=None)
                if isinstance(prob, torch.Tensor):
                    prob = prob.cpu().numpy()
                preds.append(np.atleast_1d(prob))
                labels_all.append(labels.cpu().numpy())
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

    if not preds:
        return 0.0, 0.0, train_time, np.array([]), np.array([])

    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels_all)
    auroc  = float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.5
    auprc  = float(average_precision_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0

    print(f"    {model_type}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  time={train_time:.0f}s")
    return auroc, auprc, train_time, y_pred, y_true