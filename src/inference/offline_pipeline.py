"""
src/inference/offline_pipeline.py
==================================
Offline benchmark: MaBoost vs XGBoost-flat.

Two separate evaluation tables
-------------------------------
Table 1 — Full timestep (seq_len timestamps = up to 48h)
Table 2 — Irregular time series (early/sparse/rolling/gap_variance)

Fair comparison design
-----------------------
For time-windowed tests (early_prediction, rolling_prediction):
  MaBoost:     encoder(frozen) + XGBoost RETRAINED on cut features
  XGBoost-flat: RETRAINED on cut flat features

Both models see EXACTLY the same data at each time window.
The encoder weights are frozen (trained on full stays) but the
XGBoost HEAD is retrained at each cutoff — this is fair because:
  - Encoder extracts temporal features from cut sequences correctly
  - XGBoost HEAD adapts its decision boundary to the available data
  - XGBoost-flat also retrains from scratch at each cutoff

For full-stay tests (eval_maboost, eval_xgb_flat):
  Both use original trained weights — no retraining needed.

For group tests (sparse_vs_dense, irregular_gaps):
  Both use original trained weights on subsets — fair because
  both models were trained on the same full dataset.
"""
from __future__ import annotations
import csv, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              mean_absolute_error)

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
    extra:    dict  = field(default_factory=dict)


def _flat(sequences: dict, ids: list) -> np.ndarray:
    rows = []
    for s in ids:
        seq, _, mask = sequences[s]
        r = []
        for j in range(seq.shape[1]):
            v   = seq[:, j]
            m   = mask[:, j].astype(bool)
            obs = v[m] if m.any() else None
            if obs is not None:
                obs = obs[~np.isnan(obs)]
            if obs is not None and len(obs) > 0:
                r.extend([float(np.nanmean(obs)),
                           float(np.nanstd(obs)) if len(obs) > 1 else 0.0,
                           float(np.nanmin(obs)), float(np.nanmax(obs)),
                           float(obs[-1]), float(1.0 - m.mean())])
            else:
                r.extend([np.nan, np.nan, np.nan, np.nan, np.nan, 1.0])
        rows.append(r)
    return np.array(rows, dtype=np.float32)


def _cut_sequences(sequences: dict, ids: list,
                   max_hours: float, seq_len: int = 128) -> dict:
    """
    Cut each stay to only include timestamps within first max_hours.
    Cumulative tau determines elapsed time since first observation.
    """
    PAD_TAU = 60.0
    cut = {}
    for sid in ids:
        seq, tau, mask = sequences[sid]
        valid_rows = (mask.sum(axis=1) > 0)
        if not valid_rows.any():
            cut[sid] = (seq.copy(), tau.copy(), mask.copy())
            continue
        first_valid = int(np.argmax(valid_rows))
        cum_seconds = np.zeros(len(tau), dtype=np.float64)
        for i in range(first_valid + 1, len(tau)):
            if valid_rows[i]:
                cum_seconds[i] = cum_seconds[i-1] + float(tau[i])
            else:
                cum_seconds[i] = cum_seconds[i-1]
        max_seconds = max_hours * 3600.0
        within = (cum_seconds <= max_seconds) & valid_rows
        cutoff  = int(np.where(within)[0][-1]) + 1 if within.any() else first_valid + 1
        cutoff  = min(max(cutoff, 1), seq_len)
        s_cut = seq[:cutoff]; t_cut = tau[:cutoff]; m_cut = mask[:cutoff]
        T = s_cut.shape[0]; pad = seq_len - T
        if pad > 0:
            s_cut = np.vstack([np.full((pad, seq.shape[1]), np.nan, dtype=np.float32), s_cut])
            t_cut = np.concatenate([np.full(pad, PAD_TAU, dtype=np.float32), t_cut])
            m_cut = np.vstack([np.zeros((pad, mask.shape[1]), dtype=np.float32), m_cut])
        cut[sid] = (s_cut, t_cut, m_cut)
    return cut


def _gap_variance(sequences: dict, ids: list) -> np.ndarray:
    variances = []
    for sid in ids:
        _, tau, mask = sequences[sid]
        valid = mask.sum(axis=1) > 0
        variances.append(float(tau[valid].std()) if valid.sum() > 1 else 0.0)
    return np.array(variances)


def _load_encoder(enc_path, d_input, d_model, enc_kw, device):
    enc = MambaEncoder(d_input, d_model, **enc_kw)
    enc.load_state_dict(torch.load(enc_path, map_location="cpu", weights_only=True))
    for p in enc.parameters():
        p.requires_grad = False
    return enc.eval().to(device)


def _load_xgb(mort_path, los_path):
    ckpt_dir = str(Path(mort_path).parent)
    xgb_m = XGBMortality.load(mort_path, meta_path=str(Path(ckpt_dir) / "mort_meta.pkl"))
    xgb_l = XGBLos.load(los_path,        meta_path=str(Path(ckpt_dir) / "los_meta.pkl"))
    return xgb_m, xgb_l


def _train_xgb_flat(X_tr, ym_tr, yl_tr, X_va, ym_va, yl_va):
    m = XGBMortality(); m.fit(X_tr, ym_tr, X_va, ym_va)
    l = XGBLos();       l.fit(X_tr, yl_tr, X_va, yl_va)
    return m, l


def _make_loaders_from_seqs(seqs_tr, seqs_va, seqs_te,
                             tr_ids, va_ids, te_ids,
                             y_mort, y_los, static_features, batch_size):
    from src.data.dataset import make_loaders
    merged = {**seqs_tr, **seqs_va, **seqs_te}
    tr_l, va_l, te_l = make_loaders(merged, y_mort, y_los, tr_ids, va_ids, te_ids,
                                     static_features=static_features, batch_size=batch_size)
    return tr_l, va_l, te_l


class OfflineBenchmark:
    def __init__(self, sequences, mortality_labels, los_labels,
                 train_ids, val_ids, test_ids, d_input, d_model=128,
                 static_features=None, encoder_path=None,
                 mort_path=None, los_path=None, batch_size=64,
                 device="cuda", enc_kw=None, baseline_kw=None):
        from src.data.dataset import make_loaders
        self.seqs = sequences; self.y_mort = mortality_labels
        self.y_los = los_labels; self.tr = train_ids
        self.va = val_ids; self.te = test_ids
        self.d_input = d_input; self.d_model = d_model
        self.device = device; self.enc_kw = enc_kw or {}
        self.batch_sz = batch_size; self._static = static_features
        self.tr_loader, self.va_loader, self.te_loader = make_loaders(
            sequences, mortality_labels, los_labels, train_ids, val_ids, test_ids,
            static_features=static_features, batch_size=batch_size)
        self.X_tr = _flat(sequences, train_ids)
        self.X_va = _flat(sequences, val_ids)
        self.X_te = _flat(sequences, test_ids)
        self.ym_tr = np.array([mortality_labels[s] for s in train_ids])
        self.ym_va = np.array([mortality_labels[s] for s in val_ids])
        self.ym_te = np.array([mortality_labels[s] for s in test_ids])
        self.yl_tr = np.array([los_labels[s] for s in train_ids])
        self.yl_va = np.array([los_labels[s] for s in val_ids])
        self.yl_te = np.array([los_labels[s] for s in test_ids])
        self.enc_path = encoder_path; self.mort_path = mort_path; self.los_path = los_path
        self._seq_len = next(iter(sequences.values()))[0].shape[0]

    def _result(self, name, ym, yp_m, yl=None, yp_l=None, t=0.0, extra=None):
        return BenchResult(
            name=name,
            auroc=float(roc_auc_score(ym, yp_m)),
            auprc=float(average_precision_score(ym, yp_m)),
            los_mae=float(mean_absolute_error(yl, yp_l)) if yl is not None else 0.0,
            los_rmse=float(np.sqrt(((yl-yp_l)**2).mean())) if yl is not None else 0.0,
            train_s=t, extra=extra or {})

    # ------------------------------------------------------------------
    # TABLE 1 — Full timestep (original trained weights)
    # ------------------------------------------------------------------

    def eval_maboost(self):
        enc = _load_encoder(self.enc_path, self.d_input, self.d_model, self.enc_kw, self.device)
        xgb_m, xgb_l = _load_xgb(self.mort_path, self.los_path)
        F, ym, yl = extract_features(enc, self.te_loader, self.device)
        return self._result("MaBoost (ours)", ym, xgb_m.predict(F), yl,
                            xgb_l.predict_days(F), extra={"table": "full"})

    def eval_xgb_flat(self):
        t0 = time.perf_counter()
        xgb_m, xgb_l = _train_xgb_flat(self.X_tr, self.ym_tr, self.yl_tr,
                                         self.X_va, self.ym_va, self.yl_va)
        return self._result("XGBoost-flat", self.ym_te, xgb_m.predict(self.X_te),
                            self.yl_te, xgb_l.predict_days(self.X_te),
                            time.perf_counter()-t0, extra={"table": "full"})

    # ------------------------------------------------------------------
    # TABLE 2 — Irregular time series (FAIR: both retrain at each window)
    # ------------------------------------------------------------------

    def _extract_cut_features(self, enc, h):
        """
        Extract features from encoder using sequences cut at h hours.
        Returns (F_tr, ym_tr, yl_tr, F_va, ym_va, yl_va, F_te, ym_te, yl_te).
        """
        seqs_tr = _cut_sequences(self.seqs, self.tr, h, self._seq_len)
        seqs_va = _cut_sequences(self.seqs, self.va, h, self._seq_len)
        seqs_te = _cut_sequences(self.seqs, self.te, h, self._seq_len)

        tr_l, va_l, te_l = _make_loaders_from_seqs(
            seqs_tr, seqs_va, seqs_te,
            self.tr, self.va, self.te,
            self.y_mort, self.y_los,
            self._static, self.batch_sz)

        F_tr, ym_tr, yl_tr = extract_features(enc, tr_l, self.device)
        F_va, ym_va, yl_va = extract_features(enc, va_l, self.device)
        F_te, ym_te, yl_te = extract_features(enc, te_l, self.device)

        # Flat features for XGBoost-flat
        X_tr = _flat(seqs_tr, self.tr)
        X_va = _flat(seqs_va, self.va)
        X_te = _flat(seqs_te, self.te)

        n_ts_avg = float(np.mean([
            int((seqs_te[s][2].sum(axis=1) > 0).sum()) for s in self.te
        ]))

        return (F_tr, ym_tr, yl_tr,
                F_va, ym_va, yl_va,
                F_te, ym_te, yl_te,
                X_tr, X_va, X_te,
                n_ts_avg)

    def eval_early_prediction(self, hours=[6.0, 12.0, 24.0]):
        """
        FAIR early prediction test.

        Both MaBoost and XGBoost-flat are evaluated at each time window h.

        MaBoost:
          - Encoder: frozen (trained on full stays) — extracts temporal
            features from cut sequences using ZOH time-decay correctly
          - XGBoost HEAD: RETRAINED on encoder features from cut sequences
            → adapts decision boundary to data available at hour h

        XGBoost-flat:
          - RETRAINED from scratch on flat statistics from cut sequences
            → same information budget as MaBoost

        This is the CORRECT fair comparison:
          Both models see exactly [0, h] hours of data.
          MaBoost advantage: temporal context from ZOH decay.
          XGBoost-flat: only sees aggregated stats, no temporal structure.
        """
        enc = _load_encoder(self.enc_path, self.d_input, self.d_model,
                            self.enc_kw, self.device)
        results = []
        print(f"\n  [Early Prediction] — FAIR: both models retrain at each window")
        print(f"  {'Hours':>5}  {'MaBoost':>8}  {'XGB-flat':>9}  "
              f"{'Delta':>7}  {'n_timestamps':>13}")
        print(f"  {'─'*56}")

        for h in hours:
            (F_tr, ym_tr, yl_tr,
             F_va, ym_va, yl_va,
             F_te, ym_te, yl_te,
             X_tr, X_va, X_te,
             n_ts_avg) = self._extract_cut_features(enc, h)

            # MaBoost — retrain XGBoost HEAD on cut encoder features
            xgb_m_cut = XGBMortality()
            xgb_m_cut.fit(F_tr, ym_tr, F_va, ym_va)
            xgb_l_cut = XGBLos()
            xgb_l_cut.fit(F_tr, yl_tr, F_va, yl_va)
            yp_mb  = xgb_m_cut.predict(F_te)
            yp_los = xgb_l_cut.predict_days(F_te)

            # XGBoost-flat — retrain on cut flat features
            m_flat, l_flat = _train_xgb_flat(X_tr, ym_tr, yl_tr, X_va, ym_va, yl_va)
            yp_flat     = m_flat.predict(X_te)
            yp_los_flat = l_flat.predict_days(X_te)

            auroc_mb   = float(roc_auc_score(ym_te, yp_mb))
            auroc_flat = float(roc_auc_score(ym_te, yp_flat))
            delta      = auroc_mb - auroc_flat

            print(f"  {h:>4.0f}h  {auroc_mb:>8.4f}  {auroc_flat:>9.4f}  "
                  f"{delta:>+7.4f}  {n_ts_avg:>13.1f}")

            results.append(self._result(
                f"MaBoost@{h:.0f}h", ym_te, yp_mb, yl_te, yp_los,
                extra={"hours": h, "n_ts_avg": n_ts_avg,
                       "table": "irregular", "test": "early"}))
            results.append(self._result(
                f"XGB-flat@{h:.0f}h", ym_te, yp_flat, yl_te, yp_los_flat,
                extra={"hours": h, "n_ts_avg": n_ts_avg,
                       "table": "irregular", "test": "early"}))

        return results

    def eval_sparse_vs_dense(self, sparse_threshold=10, dense_threshold=40):
        """
        FAIR sparse/dense test.
        Both use original trained weights — evaluated on subsets.
        Fair because both were trained on same full dataset.
        MaBoost advantage: encoder learned temporal patterns from sparse stays.
        """
        from src.data.dataset import make_loaders
        enc = _load_encoder(self.enc_path, self.d_input, self.d_model,
                            self.enc_kw, self.device)
        xgb_m, xgb_l = _load_xgb(self.mort_path, self.los_path)

        def n_ts(sid): return int((self.seqs[sid][2].sum(axis=1) > 0).sum())
        sparse_ids = [s for s in self.te if n_ts(s) < sparse_threshold]
        dense_ids  = [s for s in self.te if n_ts(s) > dense_threshold]

        print(f"\n  [Sparse vs Dense]  "
              f"sparse(<{sparse_threshold})={len(sparse_ids):,}  "
              f"dense(>{dense_threshold})={len(dense_ids):,}")
        print(f"  {'─'*65}")

        results = []
        for group_name, group_ids, label in [
            (f"sparse<{sparse_threshold}", sparse_ids, "Sparse"),
            (f"dense>{dense_threshold}",   dense_ids,  "Dense")]:
            if len(group_ids) < 10:
                print(f"  Skipping {group_name}: only {len(group_ids)} stays")
                continue

            _, _, te_grp = make_loaders(
                self.seqs, self.y_mort, self.y_los,
                self.tr, self.va, group_ids,
                static_features=self._static, batch_size=self.batch_sz)
            F_grp, ym_grp, yl_grp = extract_features(enc, te_grp, self.device)
            yp_mb  = xgb_m.predict(F_grp)
            yp_los = xgb_l.predict_days(F_grp)

            X_grp  = _flat(self.seqs, group_ids)
            m_flat, l_flat = _train_xgb_flat(
                self.X_tr, self.ym_tr, self.yl_tr,
                self.X_va, self.ym_va, self.yl_va)
            yp_flat     = m_flat.predict(X_grp)
            yp_los_flat = l_flat.predict_days(X_grp)

            auroc_mb   = float(roc_auc_score(ym_grp, yp_mb))
            auroc_flat = float(roc_auc_score(ym_grp, yp_flat))
            avg_ts     = float(np.mean([n_ts(s) for s in group_ids]))

            print(f"  {label:<8}  MaBoost={auroc_mb:.4f}  "
                  f"XGB-flat={auroc_flat:.4f}  Delta={auroc_mb-auroc_flat:+.4f}  "
                  f"avg_ts={avg_ts:.1f}  n={len(group_ids):,}")

            results.append(self._result(
                f"MaBoost [{group_name}]", ym_grp, yp_mb, yl_grp, yp_los,
                extra={"group": group_name, "n": len(group_ids),
                       "avg_timestamps": avg_ts,
                       "table": "irregular", "test": "sparse_dense"}))
            results.append(self._result(
                f"XGB-flat [{group_name}]", ym_grp, yp_flat, yl_grp, yp_los_flat,
                extra={"group": group_name, "n": len(group_ids),
                       "avg_timestamps": avg_ts,
                       "table": "irregular", "test": "sparse_dense"}))

        return results

    def eval_rolling_prediction(self, hours=[6.0, 12.0, 18.0, 24.0, 36.0, 48.0]):
        """
        FAIR rolling prediction — both retrain at each window.
        Shows AUC improvement curve as more data becomes available.
        MaBoost should improve faster early on due to temporal context.
        """
        enc = _load_encoder(self.enc_path, self.d_input, self.d_model,
                            self.enc_kw, self.device)
        results = []
        print(f"\n  [Rolling Prediction] — FAIR: both models retrain at each window")
        print(f"  {'Hours':>5}  {'MaBoost':>8}  {'XGB-flat':>9}  "
              f"{'Delta':>7}  {'n_timestamps':>13}")
        print(f"  {'─'*56}")

        for h in hours:
            (F_tr, ym_tr, yl_tr,
             F_va, ym_va, yl_va,
             F_te, ym_te, yl_te,
             X_tr, X_va, X_te,
             n_ts_avg) = self._extract_cut_features(enc, h)

            # MaBoost — retrain XGBoost HEAD
            xgb_m_cut = XGBMortality()
            xgb_m_cut.fit(F_tr, ym_tr, F_va, ym_va)
            xgb_l_cut = XGBLos()
            xgb_l_cut.fit(F_tr, yl_tr, F_va, yl_va)
            yp_mb  = xgb_m_cut.predict(F_te)
            yp_los = xgb_l_cut.predict_days(F_te)

            # XGBoost-flat — retrain
            m_flat, l_flat = _train_xgb_flat(X_tr, ym_tr, yl_tr, X_va, ym_va, yl_va)
            yp_flat     = m_flat.predict(X_te)
            yp_los_flat = l_flat.predict_days(X_te)

            auroc_mb   = float(roc_auc_score(ym_te, yp_mb))
            auroc_flat = float(roc_auc_score(ym_te, yp_flat))
            delta      = auroc_mb - auroc_flat

            print(f"  {h:>4.0f}h  {auroc_mb:>8.4f}  {auroc_flat:>9.4f}  "
                  f"{delta:>+7.4f}  {n_ts_avg:>13.1f}")

            results.append(self._result(
                f"MaBoost@{h:.0f}h", ym_te, yp_mb, yl_te, yp_los,
                extra={"hours": h, "n_ts_avg": n_ts_avg,
                       "table": "irregular", "test": "rolling"}))
            results.append(self._result(
                f"XGB-flat@{h:.0f}h", ym_te, yp_flat, yl_te, yp_los_flat,
                extra={"hours": h, "n_ts_avg": n_ts_avg,
                       "table": "irregular", "test": "rolling"}))

        mb_aucs   = [r.auroc for r in results if "MaBoost" in r.name]
        flat_aucs = [r.auroc for r in results if "XGB"     in r.name]
        avg_delta = float(np.mean([m-f for m,f in zip(mb_aucs, flat_aucs)]))
        print(f"\n  Avg Delta = {avg_delta:+.4f}  "
              f"-> {'MaBoost wins' if avg_delta > 0 else 'XGB-flat wins'} "
              f"on irregular time series")

        return results

    def eval_irregular_gaps(self, top_frac=0.25):
        """
        FAIR irregular gap test.
        Both use original trained weights on high/low variance subsets.
        """
        from src.data.dataset import make_loaders
        enc = _load_encoder(self.enc_path, self.d_input, self.d_model,
                            self.enc_kw, self.device)
        xgb_m, xgb_l = _load_xgb(self.mort_path, self.los_path)

        variances = _gap_variance(self.seqs, self.te)
        threshold = np.percentile(variances, (1-top_frac)*100)
        irreg_ids = [s for s,v in zip(self.te, variances) if v >= threshold]
        reg_ids   = [s for s,v in zip(self.te, variances) if v <  threshold]
        avg_irreg = float(np.mean([v for v in variances if v >= threshold]))
        avg_reg   = float(np.mean([v for v in variances if v <  threshold]))

        print(f"\n  [Irregular Gap Variance]  "
              f"high(top {top_frac:.0%})={len(irreg_ids):,}  "
              f"low(bot {1-top_frac:.0%})={len(reg_ids):,}")
        print(f"  Avg gap std — high: {avg_irreg/60:.1f}min  low: {avg_reg/60:.1f}min")
        print(f"  {'─'*65}")

        results = []
        for group_name, group_ids, label, avg_var in [
            ("high_var", irreg_ids, f"High gap var (top {top_frac:.0%})", avg_irreg),
            ("low_var",  reg_ids,   f"Low gap var  (bot {1-top_frac:.0%})", avg_reg)]:
            if len(group_ids) < 10:
                print(f"  Skipping {group_name}: only {len(group_ids)}")
                continue

            _, _, te_grp = make_loaders(
                self.seqs, self.y_mort, self.y_los,
                self.tr, self.va, group_ids,
                static_features=self._static, batch_size=self.batch_sz)
            F_grp, ym_grp, yl_grp = extract_features(enc, te_grp, self.device)
            yp_mb  = xgb_m.predict(F_grp)
            yp_los = xgb_l.predict_days(F_grp)

            X_grp  = _flat(self.seqs, group_ids)
            m_flat, l_flat = _train_xgb_flat(
                self.X_tr, self.ym_tr, self.yl_tr,
                self.X_va, self.ym_va, self.yl_va)
            yp_flat     = m_flat.predict(X_grp)
            yp_los_flat = l_flat.predict_days(X_grp)

            auroc_mb   = float(roc_auc_score(ym_grp, yp_mb))
            auroc_flat = float(roc_auc_score(ym_grp, yp_flat))

            print(f"  {label:<40}  MaBoost={auroc_mb:.4f}  "
                  f"XGB-flat={auroc_flat:.4f}  Delta={auroc_mb-auroc_flat:+.4f}")

            results.append(self._result(
                f"MaBoost [{group_name}]", ym_grp, yp_mb, yl_grp, yp_los,
                extra={"group": group_name, "n": len(group_ids),
                       "avg_gap_std_s": avg_var,
                       "table": "irregular", "test": "gap_variance"}))
            results.append(self._result(
                f"XGB-flat [{group_name}]", ym_grp, yp_flat, yl_grp, yp_los_flat,
                extra={"group": group_name, "n": len(group_ids),
                       "avg_gap_std_s": avg_var,
                       "table": "irregular", "test": "gap_variance"}))

        return results

    def run_all(self, skip=None):
        skip = skip or []; results = []
        print("\n" + "="*62 + "\n  TABLE 1 — Full Timestep\n" + "="*62)
        for name, fn in [("MaBoost (ours)", self.eval_maboost),
                          ("XGBoost-flat",   self.eval_xgb_flat)]:
            if name in skip: continue
            print(f"\n[Benchmark] {name} ...")
            try: results.append(fn())
            except Exception as e: print(f"  ERROR: {e}")
        print("\n" + "="*62 + "\n  TABLE 2 — Irregular Time Series\n" + "="*62)
        for name, fn in [("early_prediction",   self.eval_early_prediction),
                          ("sparse_vs_dense",    self.eval_sparse_vs_dense),
                          ("rolling_prediction", self.eval_rolling_prediction),
                          ("irregular_gaps",     self.eval_irregular_gaps)]:
            if name in skip: continue
            try: results.extend(fn())
            except Exception as e: print(f"\n  ERROR {name}: {e}")
        return results

    @staticmethod
    def print_table(results):
        full = [r for r in results if r.extra.get("table","full") == "full"]
        if full:
            h = (f"{'Model':<24}  {'AUROC':>6}  {'AUPRC':>6}  "
                 f"{'MAE(d)':>7}  {'RMSE(d)':>8}  {'Train(s)':>9}")
            sep = "─"*len(h)
            print(f"\n{'='*len(h)}\n  TABLE 1 — Full Timestep\n{'='*len(h)}\n{h}\n{sep}")
            for r in sorted(full, key=lambda x: x.auroc, reverse=True):
                print(f"{r.name:<24}  {r.auroc:>6.4f}  {r.auprc:>6.4f}  "
                      f"{r.los_mae:>7.3f}  {r.los_rmse:>8.3f}  {r.train_s:>9.1f}")
            print(sep)

        irregular = [r for r in results if r.extra.get("table") == "irregular"]
        if not irregular: return

        type_labels = {
            "early":        "Early Prediction (6h/12h/24h) — FAIR retrain",
            "sparse_dense": "Sparse vs Dense Stays",
            "rolling":      "Rolling Prediction (6h->48h) — FAIR retrain",
            "gap_variance": "Gap Variance Groups",
        }
        print(f"\n{'='*72}\n  TABLE 2 — Irregular Time Series Evaluation\n{'='*72}")
        by_type = {}
        for r in irregular:
            by_type.setdefault(r.extra.get("test","other"), []).append(r)

        h = (f"  {'Model':<36}  {'AUROC':>6}  {'AUPRC':>6}  "
             f"{'MAE(d)':>7}  {'Delta':>8}")
        sep = "  " + "─"*(len(h)-2)

        for test_type, label in type_labels.items():
            group = by_type.get(test_type, [])
            if not group: continue
            print(f"\n  -- {label} --\n{h}\n{sep}")
            mb_rows   = [r for r in group if "MaBoost" in r.name]
            flat_rows = [r for r in group if "XGB"     in r.name]
            for mb in mb_rows:
                flat = next((f for f in flat_rows
                             if f.extra.get("hours") == mb.extra.get("hours")
                             and f.extra.get("group") == mb.extra.get("group")), None)
                dstr = f"{mb.auroc-flat.auroc:>+8.4f}" if flat else "        "
                print(f"  {mb.name:<36}  {mb.auroc:>6.4f}  {mb.auprc:>6.4f}  "
                      f"{mb.los_mae:>7.3f}  {dstr}")
                if flat:
                    print(f"  {flat.name:<36}  {flat.auroc:>6.4f}  "
                          f"{flat.auprc:>6.4f}  {flat.los_mae:>7.3f}")
                print()
            print(sep)

    @staticmethod
    def save_csv(results, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["model","auroc","auprc","los_mae","los_rmse","train_s",
                      "table","test","hours","group","n","n_ts_avg","avg_gap_std_s"]
        def _row(r):
            return {"model": r.name, "auroc": f"{r.auroc:.6f}",
                    "auprc": f"{r.auprc:.6f}", "los_mae": f"{r.los_mae:.4f}",
                    "los_rmse": f"{r.los_rmse:.4f}", "train_s": f"{r.train_s:.1f}",
                    "table": r.extra.get("table","full"),
                    "test":  r.extra.get("test", "full"),
                    "hours": r.extra.get("hours",""),
                    "group": r.extra.get("group",""),
                    "n":     r.extra.get("n",""),
                    "n_ts_avg":      r.extra.get("n_ts_avg",""),
                    "avg_gap_std_s": r.extra.get("avg_gap_std_s","")}

        full_res = [r for r in results if r.extra.get("table","full") != "irregular"]
        if full_res:
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in full_res: w.writerow(_row(r))
            print(f"[Benchmark] Table 1 saved -> {path}")

        stem = Path(path).stem
        irr_path = str(Path(path).parent / f"{stem}_irregular.csv")
        irr_res = [r for r in results if r.extra.get("table") == "irregular"]
        if irr_res:
            with open(irr_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in irr_res: w.writerow(_row(r))
            print(f"[Benchmark] Table 2 saved -> {irr_path}")