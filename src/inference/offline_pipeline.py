"""
src/inference/offline_pipeline.py
==================================
Offline benchmark: MaBoost vs GRU-D, Transformer, LSTM, XGBoost-flat.
Reports AUROC/AUPRC for mortality and MAE/RMSE for LOS.

Bugs fixed
----------
1. eval_maboost() now passes **enc_kw to MambaEncoder so state_dict
   matches the encoder that was actually trained (e.g. use_channel_attn).
2. eval_maboost() passes meta_path to XGBMortality/XGBLos.load() so
   Platt scaling and isotonic calibration are active during benchmark.
3. _flat() returns NaN (not 0.0) for never-observed features so
   XGBoost-flat can use its sparsity-aware splits correctly.
4. _collect() clips log-LOS predictions to ≥0 before expm1 to prevent
   negative LOS values inflating MAE/RMSE.
"""
from __future__ import annotations
import csv, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              mean_absolute_error)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.models.baselines     import GRUDModel, TransformerModel, LSTMModel
from src.models.mamba_encoder import MambaEncoder
from src.models.xgboost_head  import XGBMortality, XGBLos
from src.training.stage2_train import extract_features


@dataclass
class BenchResult:
    name:     str
    auroc:    float = 0.0
    auprc:    float = 0.0
    los_mae:  float = 0.0
    los_rmse: float = 0.0
    train_s:  float = 0.0


# ---------------------------------------------------------------------------
# Training helper for deep baselines
# ---------------------------------------------------------------------------

def _train_baseline(model, tr_loader, va_loader, device, epochs=50, lr=1e-3):
    model = model.to(device)
    w     = torch.tensor([1.0, 9.0], device=device)
    ce    = nn.CrossEntropyLoss(weight=w)
    mse   = nn.MSELoss()
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    best_auc, best_state, no_imp = 0.0, None, 0
    t0 = time.perf_counter()

    for _ in range(epochs):
        model.train()
        for batch in tr_loader:
            x, tau, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            ym, yl       = batch[-2].to(device), batch[-1].to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                lm, ll = model(x, tau, mask)
                loss   = ce(lm, ym) + 0.5 * mse(ll, torch.log1p(yl))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        # Quick val AUC
        model.eval()
        probs, lbs = [], []
        with torch.no_grad():
            for batch in va_loader:
                x, tau, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                lm, _ = model(x, tau, mask)
                probs.append(torch.softmax(lm, -1)[:, 1].cpu())
                lbs.append(batch[-2].cpu())
        auc = roc_auc_score(torch.cat(lbs).numpy(), torch.cat(probs).numpy())
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= 10:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, time.perf_counter() - t0


@torch.no_grad()
def _collect(model, loader, device):
    """
    Collect mortality probs and LOS predictions from a trained baseline.
    FIX 4: clip log-LOS predictions to ≥ 0 before expm1.
    Negative predictions from ll head give expm1(neg) < 0, inflating MAE.
    """
    model.eval()
    p_m, p_l, y_m, y_l = [], [], [], []
    for batch in loader:
        x, tau, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        lm, ll = model(x, tau, mask)
        p_m.append(torch.softmax(lm, -1)[:, 1].cpu())
        # FIX 4: clip before expm1 to prevent negative LOS predictions
        p_l.append(np.expm1(np.maximum(ll.cpu().numpy(), 0.0)))
        y_m.append(batch[-2].cpu())
        y_l.append(batch[-1].cpu())
    return (torch.cat(p_m).numpy(), np.concatenate(p_l),
            torch.cat(y_m).numpy(), torch.cat(y_l).numpy())


def _flat(sequences: dict, ids: list) -> np.ndarray:
    """
    Build flat feature matrix for XGBoost-flat baseline.

    FIX 3: never-observed features return NaN (not 0.0).
    Original used np.array([0.]) as fallback — this zeros out
    "feature was never measured", destroying XGBoost sparsity-aware splits
    and causing AUROC ~0.51 (same bug as the main pipeline had before).
    """
    rows = []
    for s in ids:
        seq, _, mask = sequences[s]
        r = []
        for j in range(seq.shape[1]):
            v   = seq[:, j]
            m   = mask[:, j].astype(bool)
            obs = v[m] if m.any() else None
            if obs is not None:
                r.extend([obs.mean(), obs.std() if len(obs) > 1 else 0.0,
                           obs.min(), obs.max(), obs[-1], m.mean()])
            else:
                # NaN for never-observed → XGBoost routes via sparsity-aware split
                r.extend([np.nan, np.nan, np.nan, np.nan, np.nan, 1.0])
        rows.append(r)
    return np.array(rows, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main benchmark class
# ---------------------------------------------------------------------------

class OfflineBenchmark:
    def __init__(self, sequences, mortality_labels, los_labels,
                 train_ids, val_ids, test_ids,
                 d_input, d_model=128, static_features=None,
                 encoder_path=None, mort_path=None, los_path=None,
                 batch_size=64, device="cuda",
                 enc_kw: Optional[dict] = None):
        """
        enc_kw : keyword args passed to MambaEncoder when loading the trained
                 encoder (e.g. {"use_channel_attn": True}).  Must match the
                 params used during Stage 1 training or load_state_dict fails.
        """
        from src.data.dataset import make_loaders
        self.seqs   = sequences
        self.y_mort = mortality_labels
        self.y_los  = los_labels
        self.tr, self.va, self.te = train_ids, val_ids, test_ids
        self.d_input  = d_input
        self.d_model  = d_model
        self.device   = device
        self.enc_kw   = enc_kw or {}

        self.tr_loader, self.va_loader, self.te_loader = make_loaders(
            sequences, mortality_labels, los_labels,
            train_ids, val_ids, test_ids,
            static_features=static_features, batch_size=batch_size,
        )

        # Flat features for XGBoost-flat baseline (NaN-aware)
        self.X_tr = _flat(sequences, train_ids)
        self.X_va = _flat(sequences, val_ids)
        self.X_te = _flat(sequences, test_ids)

        # Labels
        self.ym_tr = np.array([mortality_labels[s] for s in train_ids])
        self.ym_va = np.array([mortality_labels[s] for s in val_ids])
        self.ym_te = np.array([mortality_labels[s] for s in test_ids])
        self.yl_tr = np.array([los_labels[s] for s in train_ids])
        self.yl_va = np.array([los_labels[s] for s in val_ids])
        self.yl_te = np.array([los_labels[s] for s in test_ids])

        self.enc_path  = encoder_path
        self.mort_path = mort_path
        self.los_path  = los_path

    def _result(self, name, ym, yp_m, yl, yp_l, t=0.0):
        return BenchResult(
            name=name,
            auroc=roc_auc_score(ym, yp_m),
            auprc=average_precision_score(ym, yp_m),
            los_mae=mean_absolute_error(yl, yp_l),
            los_rmse=float(np.sqrt(((yl - yp_l) ** 2).mean())),
            train_s=t,
        )

    def eval_maboost(self):
        """
        FIX 1: pass **self.enc_kw to MambaEncoder so architecture matches
                the checkpoint (e.g. use_channel_attn=True adds channel_attn
                weights — without it, load_state_dict raises RuntimeError).
        FIX 2: pass meta_path to both XGBMortality.load and XGBLos.load so
                Platt scaling and isotonic calibration are active. Without
                this, benchmark measures uncalibrated sigmoid scores.
        """
        enc = MambaEncoder(self.d_input, self.d_model, **self.enc_kw)
        enc.load_state_dict(
            torch.load(self.enc_path, map_location="cpu", weights_only=True)
        )
        for p in enc.parameters():
            p.requires_grad = False

        F, ym, yl = extract_features(enc, self.te_loader, self.device)

        # FIX 2: meta_path loads Platt scaler + keep_idx / isotonic regression
        ckpt_dir  = str(Path(self.mort_path).parent)
        xgb_m = XGBMortality.load(
            self.mort_path,
            meta_path=str(Path(ckpt_dir) / "mort_meta.pkl"),
        )
        xgb_l = XGBLos.load(
            self.los_path,
            meta_path=str(Path(ckpt_dir) / "los_meta.pkl"),
        )
        return self._result(
            "MaBoost (ours)", ym,
            xgb_m.predict(F), yl, xgb_l.predict_days(F),
        )

    def _eval_deep(self, name, model):
        model, t = _train_baseline(model, self.tr_loader, self.va_loader, self.device)
        pm, pl, ym, yl = _collect(model, self.te_loader, self.device)
        return self._result(name, ym, pm, yl, pl, t)

    def eval_grud(self):
        return self._eval_deep("GRU-D", GRUDModel(self.d_input, self.d_model))

    def eval_transformer(self):
        return self._eval_deep("Transformer", TransformerModel(self.d_input, self.d_model))

    def eval_lstm(self):
        return self._eval_deep("LSTM", LSTMModel(self.d_input, self.d_model))

    def eval_xgb_flat(self):
        """XGBoost on hand-crafted flat features (no encoder)."""
        t0  = time.perf_counter()
        m   = XGBMortality()
        m.fit(self.X_tr, self.ym_tr, self.X_va, self.ym_va)
        l   = XGBLos()
        l.fit(self.X_tr, self.yl_tr, self.X_va, self.yl_va)
        return self._result(
            "XGBoost-flat", self.ym_te, m.predict(self.X_te),
            self.yl_te, l.predict_days(self.X_te),
            time.perf_counter() - t0,
        )

    def run_all(self, skip: Optional[List[str]] = None) -> List[BenchResult]:
        skip    = skip or []
        runners = [
            ("MaBoost (ours)", self.eval_maboost),
            ("GRU-D",          self.eval_grud),
            ("Transformer",    self.eval_transformer),
            ("LSTM",           self.eval_lstm),
            ("XGBoost-flat",   self.eval_xgb_flat),
        ]
        results = []
        for name, fn in runners:
            if name in skip:
                continue
            print(f"\n[Benchmark] {name} …")
            try:
                results.append(fn())
            except Exception as e:
                print(f"  ERROR: {e}")
        return sorted(results, key=lambda r: r.auroc, reverse=True)

    @staticmethod
    def print_table(results: List[BenchResult]):
        h = (f"{'Model':<22}  {'AUROC':>6}  {'AUPRC':>6}  "
             f"{'MAE(d)':>7}  {'RMSE(d)':>8}  {'Train(s)':>9}")
        sep = "─" * len(h)
        print(f"\n{sep}\n{h}\n{sep}")
        for r in results:
            print(f"{r.name:<22}  {r.auroc:>6.4f}  {r.auprc:>6.4f}  "
                  f"{r.los_mae:>7.3f}  {r.los_rmse:>8.3f}  {r.train_s:>9.1f}")
        print(sep)

    @staticmethod
    def save_csv(results: List[BenchResult], path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["model", "auroc", "auprc",
                               "los_mae", "los_rmse", "train_s"]
            )
            w.writeheader()
            for r in results:
                w.writerow({"model": r.name, "auroc": f"{r.auroc:.6f}",
                            "auprc": f"{r.auprc:.6f}", "los_mae": f"{r.los_mae:.4f}",
                            "los_rmse": f"{r.los_rmse:.4f}", "train_s": f"{r.train_s:.1f}"})
        print(f"[Benchmark] Results saved → {path}")