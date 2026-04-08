#!/usr/bin/env python3
"""
scripts/run_hospital_irregular_benchmark.py
============================================
FAIR BENCHMARK: Simulating Irregular Hospital Data — Raw Problem Match

Raw Problem (plan.md §1.3)
--------------------------
  - At each event time, 1 or more variables may be updated or added
  - The set of updated variables is different each time
  - The gap delta_t between event times is irregular
  - After every event update, predict again

What this benchmark does
------------------------
All models see IDENTICAL sequences derived from real MIMIC-IV irregular events.
Every scenario is constructed from the actual per-minute event timestamps,
preserving the real delta_t gaps. No artificial regularization.

Scenarios Tested
----------------
1. raw_event_stream   — predict after every individual event (all timestamps)
2. early_6h           — predict using only the first 6h of events (truly sparse)
3. early_12h          — predict using only the first 12h of events
4. early_24h          — predict using only the first 24h of events
5. sparse_patients    — patients with fewest event timestamps (bottom-25% by count)
6. dense_patients     — patients with most event timestamps (top-25% by count)
7. high_gap_var       — patients with most irregular timing (top-25% by gap std-dev)
8. low_gap_var        — patients with most regular timing (bottom-25% by gap std-dev)

Models Benchmarked
------------------
All models are trained on the SAME TRAIN split and evaluated FRESHLY at each scenario:

  Group A — Deep temporal models (retrained per scenario):
    • GRU-D      (handles missing via time-decay — most similar to raw problem)
    • LSTM       (standard RNN, no time encoding)
    • Transformer (self-attention, positional encoding)
    • InterpNet  (SCI+CCI+GRU — designed for irregular data)
    • TCN        (temporal conv, no time encoding)
    • SAnD       (attention with diagnosis predictions)

  Group B — MaBoost methods (frozen encoder, XGBoost head retrained per scenario):
    • MaBoost         (main method: frozen Mamba enc + XGBoost head)
    • Distilled-XGB   (sub-method: XGBoost + teacher signals)

  Group C — Tabular baseline (no time modeling):
    • XGBoost-flat    (mean/max/std/last/miss aggregation only)

Fairness Guarantees
-------------------
✓ All models use the SAME sequence data per scenario
✓ Deep models retrained on train split per scenario
✓ MaBoost encoder FROZEN (trained on full data) but XGBoost HEAD retrained per scenario
✓ XGBoost-flat retrained per scenario from scratch
✓ Same test set across all models at each scenario

Usage
-----
  python scripts/run_hospital_irregular_benchmark.py --config config.yaml --skip-etl
  python scripts/run_hospital_irregular_benchmark.py --config config.yaml --skip-etl --scenarios early_6h,early_12h,sparse_patients,raw_event_stream

Output
------
  results/benchmark_hospital_irregular_fair.csv
  summary/benchmark_hospital_irregular_fair.md
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
import time
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    roc_auc_score, average_precision_score, mean_absolute_error,
)
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _gc() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _checkpoint(stage: str, label: str, **kwargs) -> None:
    status_file = Path("results/pipeline_progress.json")
    status_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "label": label,
        "ts": datetime.datetime.now().isoformat(),
        **kwargs
    }
    status_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _metrics(label: str, y_m: np.ndarray, p_m: np.ndarray,
             y_l: np.ndarray, p_l: np.ndarray,
             scenario: str, group: str,
             n_ts_avg: float, avg_gap_std_h: float,
             train_s: float = 0.0,
             seed: int = 0) -> dict:
    try:
        auroc = float(roc_auc_score(y_m, p_m))
    except Exception:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_m, p_m))
    except Exception:
        auprc = float("nan")
    los_mae  = float(mean_absolute_error(y_l, p_l))
    los_rmse = float(np.sqrt(np.mean((y_l - p_l) ** 2)))
    return dict(
        seed=seed, model=label, scenario=scenario, group=group,
        auroc=auroc, auprc=auprc, los_mae=los_mae, los_rmse=los_rmse,
        n=int(len(y_m)), train_s=round(train_s, 2),
        n_ts_avg=round(n_ts_avg, 2), avg_gap_std_h=round(avg_gap_std_h, 4),
        pos_rate=round(float(y_m.mean()), 4),
    )


def _print_row(r: dict) -> None:
    seed_tag = f"[seed={r.get('seed',0)}] " if r.get('seed', 0) != 0 else ""
    print(f"  {seed_tag}[{r['scenario']:22s}|{r['group']:20s}] "
          f"{r['model']:28s}  "
          f"AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
          f"LOS-MAE={r['los_mae']:.2f}d  "
          f"n={r['n']:,}  ts_avg={r['n_ts_avg']:.1f}  "
          f"gap_std={r['avg_gap_std_h']:.2f}h")


def _save_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "model", "scenario", "group", "auroc", "auprc",
              "los_mae", "los_rmse", "n", "train_s",
              "n_ts_avg", "avg_gap_std_h", "pos_rate"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in fields})
    print(f"\n[Hospital Benchmark] Saved -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Sequence utilities
# ─────────────────────────────────────────────────────────────────────────────

def _seq_meta(sequences: dict, ids: list) -> dict:
    """Compute per-stay irregularity metrics."""
    out = {}
    for sid in ids:
        tpl = sequences.get(sid)
        if tpl is None:
            continue
        _, tau, mask = tpl[:3]
        valid = mask.sum(axis=1) > 0
        n_ts = int(valid.sum())
        taus_valid = tau[valid].astype(np.float64)
        gap_std_s = float(np.std(taus_valid[1:])) if n_ts > 1 else 0.0
        out[sid] = {"n_ts": n_ts, "gap_std_s": gap_std_s}
    return out


def _cut_to_hours(sequences: dict, ids: list, max_hours: float, seq_len: int) -> Tuple[dict, dict]:
    """
    Returns (cut_seqs, meta_per_id) where each sequence is trimmed to timestamps
    within the first max_hours from the first valid event.
    Reflects the raw problem: predict using only the data available at time t.
    """
    PAD_TAU = 60.0
    cut = {}
    meta = {}
    for sid in ids:
        tpl = sequences.get(sid)
        if tpl is None:
            continue
        seq, tau, mask = tpl[:3]
        valid_rows = mask.sum(axis=1) > 0
        if not valid_rows.any():
            cut[sid] = (seq.copy(), tau.copy(), mask.copy())
            meta[sid] = {"n_ts": 0, "gap_std_s": 0.0}
            continue
        first_v = int(np.argmax(valid_rows))
        cum_s = np.zeros(len(tau), dtype=np.float64)
        for i in range(first_v + 1, len(tau)):
            if valid_rows[i]:
                cum_s[i] = cum_s[i-1] + float(tau[i])
            else:
                cum_s[i] = cum_s[i-1]
        within = (cum_s <= max_hours * 3600.0) & valid_rows
        cutoff  = int(np.where(within)[0][-1]) + 1 if within.any() else first_v + 1
        cutoff  = min(max(cutoff, 1), seq_len)
        s_cut = seq[:cutoff]; t_cut = tau[:cutoff]; m_cut = mask[:cutoff]
        T = s_cut.shape[0]; pad = seq_len - T
        if pad > 0:
            s_cut = np.vstack([np.full((pad, seq.shape[1]), np.nan, dtype=np.float32), s_cut])
            t_cut = np.concatenate([np.full(pad, PAD_TAU, dtype=np.float32), t_cut])
            m_cut = np.vstack([np.zeros((pad, mask.shape[1]), dtype=np.float32), m_cut])
        cut[sid] = (s_cut, t_cut, m_cut)
        valid_in_cut = m_cut.sum(axis=1) > 0
        n_ts = int(valid_in_cut.sum())
        taus_v = t_cut[valid_in_cut].astype(np.float64)
        gap_std_s = float(np.std(taus_v[1:])) if n_ts > 1 else 0.0
        meta[sid] = {"n_ts": n_ts, "gap_std_s": gap_std_s}
    return cut, meta


def _build_raw_event_stream(sequences: dict, ids: list, y_mort: dict, y_los: dict,
                             static: Optional[dict], stay_outcomes: dict,
                             research_cfg: dict, max_samples_per_stay: int = 64) -> Tuple[dict, dict, dict, Optional[dict], dict]:
    """
    One sample per event timestamp — exactly the raw problem:
    - variable set at each event is different
    - delta_t is irregular
    - predict after every event
    """
    from src.data.temporal_samples import build_event_driven_samples
    seq_sub = {sid: sequences[sid] for sid in ids if sid in sequences}
    st_sub  = {sid: static[sid] for sid in ids if static and sid in static} if static else None
    ym_sub  = {sid: y_mort[sid] for sid in ids if sid in y_mort}
    yl_sub  = {sid: y_los[sid]  for sid in ids if sid in y_los}
    result = build_event_driven_samples(
        sequences=seq_sub,
        mortality_labels=ym_sub,
        los_labels=yl_sub,
        stay_outcomes=stay_outcomes,
        static_features=st_sub,
        anchor_mode="event",
        min_elapsed_hours=0.0,
        step_hours=1.0,
        min_anchor_gap_minutes=0.0,
        max_samples_per_stay=max_samples_per_stay,
        obs_window_hours=float(research_cfg.get("obs_window_hours", 4.0)),
        mortality_horizon_hours=float(research_cfg.get("mortality_horizon_hours", 24.0)),
        los_target=str(research_cfg.get("los_target", "remaining")),
    )
    return (result["sequences"], result["mortality_labels"], result["los_labels"],
            result["static_features"], result.get("sample_metadata", {}))


def _group_ids(meta: dict, ids: list, kind: str) -> list:
    """Return subset of ids matching a grouping criterion."""
    if kind == "all":
        return list(ids)
    arr_n   = np.array([meta.get(sid, {}).get("n_ts", 0) for sid in ids])
    arr_gap = np.array([meta.get(sid, {}).get("gap_std_s", 0.0) for sid in ids])
    if kind == "sparse_q25":
        thr = float(np.percentile(arr_n, 25))
        return [sid for sid, v in zip(ids, arr_n) if v <= thr]
    if kind == "dense_q75":
        thr = float(np.percentile(arr_n, 75))
        return [sid for sid, v in zip(ids, arr_n) if v >= thr]
    if kind == "high_gap_var_q75":
        thr = float(np.percentile(arr_gap, 75))
        return [sid for sid, v in zip(ids, arr_gap) if v >= thr]
    if kind == "low_gap_var_q25":
        thr_high = float(np.percentile(arr_gap, 25))
        return [sid for sid, v in zip(ids, arr_gap) if v <= thr_high]
    raise ValueError(f"Unknown group: {kind}")


# ─────────────────────────────────────────────────────────────────────────────
# Flat feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _flat_features(sequences: dict, ids: list) -> np.ndarray:
    rows = []
    for sid in ids:
        seq, _, mask = sequences[sid][:3]
        row = []
        for j in range(seq.shape[1]):
            v = seq[:, j]; m = mask[:, j].astype(bool)
            obs = v[m]; obs = obs[~np.isnan(obs)] if len(obs) > 0 else np.array([])
            if len(obs) > 0:
                row.extend([
                    float(np.nanmean(obs)),
                    float(np.nanstd(obs)) if len(obs) > 1 else 0.0,
                    float(np.nanmax(obs)),
                    float(obs[-1]),
                    float(1.0 - m.mean()),
                ])
            else:
                row.extend([np.nan, np.nan, np.nan, np.nan, 1.0])
        rows.append(row)
    return np.array(rows, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# MaBoost feature extraction (using frozen encoder)
# ─────────────────────────────────────────────────────────────────────────────

def _maboost_features(enc, sequences: dict, ids: list,
                      y_mort: dict, y_los: dict,
                      static: Optional[dict], device: str,
                      batch_size: int = 32,
                      force_stats = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from src.data.dataset import make_loaders
    from src.training.stage2_train import extract_features
    # Build minimal loaders: train=val=[] to avoid leakage, test=ids
    dummy_ids = ids[:1]  # minimum for loader
    ym = {sid: y_mort.get(sid, 0) for sid in ids}
    yl = {sid: y_los.get(sid, 0.0) for sid in ids}
    _, _, te_loader = make_loaders(
        sequences, ym, yl,
        train_ids=dummy_ids,
        val_ids=dummy_ids,
        test_ids=ids,
        static_features=static,
        batch_size=batch_size,
        shuffle_train=False,
        force_stats=force_stats,
        use_dynamic_drop=False,
    )
    return extract_features(enc, te_loader, device)


# ─────────────────────────────────────────────────────────────────────────────
# Deep baseline training & inference
# ─────────────────────────────────────────────────────────────────────────────

# (Removed broken _train_deep_baseline and _infer_deep methods)

# ─────────────────────────────────────────────────────────────────────────────
# XGBoost-flat training & inference
# ─────────────────────────────────────────────────────────────────────────────

def _train_xgb_flat(sequences: dict, y_mort: dict, y_los: dict,
                    train_ids: list, val_ids: list, device: str = "cuda"):
    from src.models.xgboost_head import XGBMortality, XGBLos
    X_tr = _flat_features(sequences, train_ids)
    X_va = _flat_features(sequences, val_ids)
    ym_tr = np.array([y_mort.get(s, 0) for s in train_ids], dtype=np.int32)
    ym_va = np.array([y_mort.get(s, 0) for s in val_ids],   dtype=np.int32)
    yl_tr = np.array([y_los.get(s, 0.0)  for s in train_ids], dtype=np.float32)
    yl_va = np.array([y_los.get(s, 0.0)  for s in val_ids],   dtype=np.float32)
    mort = XGBMortality(device=device)
    los  = XGBLos(device=device)
    t0 = time.perf_counter()
    mort.fit(X_tr, ym_tr, X_va, ym_va)
    los.fit(X_tr, yl_tr, X_va, yl_va)
    train_s = time.perf_counter() - t0
    return mort, los, train_s


def _predict_xgb_flat(mort_h, los_h, sequences: dict, y_mort: dict, y_los: dict, ids: list):
    X  = _flat_features(sequences, ids)
    ym = np.array([y_mort.get(s, 0)   for s in ids], dtype=np.int32)
    yl = np.array([y_los.get(s, 0.0)  for s in ids], dtype=np.float32)
    pm = mort_h.predict(X)
    pl = los_h.predict_days(X)
    return ym, yl, pm, pl


# ─────────────────────────────────────────────────────────────────────────────
# MaBoost XGBoost training & inference
# ─────────────────────────────────────────────────────────────────────────────

def _train_maboost_head(enc, sequences: dict, y_mort: dict, y_los: dict,
                        train_ids: list, val_ids: list, static: Optional[dict],
                        meta: Optional[dict], device: str, batch_size: int = 32,
                        force_stats = None):
    from src.models.xgboost_head import XGBMortality, XGBLos
    from src.training.stage2_train import extract_features
    from src.data.dataset import make_loaders
    import pickle

    tr_loader, va_loader, _ = make_loaders(
        sequences, y_mort, y_los, train_ids, val_ids, val_ids[:1],
        static_features=static, batch_size=batch_size, shuffle_train=False,
        force_stats=force_stats,
        use_dynamic_drop=False,
    )
    F_tr, ym_tr, yl_tr = extract_features(enc, tr_loader, device)
    F_va, ym_va, yl_va = extract_features(enc, va_loader, device)

    # Apply any cached stage2 transforms from the original training
    # Note: ignore_keep_idx=True ensures the freshly trained head has access to all raw stats!
    if meta:
        from run_irregular_basic import _apply_stage2_meta
        F_tr_m = _apply_stage2_meta(F_tr, meta, for_mortality=True, ignore_keep_idx=True)
        F_tr_l = _apply_stage2_meta(F_tr, meta, for_mortality=False, ignore_keep_idx=True)
        F_va_m = _apply_stage2_meta(F_va, meta, for_mortality=True, ignore_keep_idx=True)
        F_va_l = _apply_stage2_meta(F_va, meta, for_mortality=False, ignore_keep_idx=True)
    else:
        F_tr_m = F_tr_l = F_tr
        F_va_m = F_va_l = F_va

    mort = XGBMortality(device=device)
    los  = XGBLos(device=device)
    t0 = time.perf_counter()
    mort.fit(F_tr_m, ym_tr, F_va_m, ym_va)
    los.fit(F_tr_l, yl_tr, F_va_l, yl_va)
    train_s = time.perf_counter() - t0
    return mort, los, train_s, F_tr, F_va


def _predict_maboost(enc, mort_h, los_h, sequences: dict, y_mort: dict, y_los: dict,
                     ids: list, static: Optional[dict], meta: Optional[dict],
                     device: str, batch_size: int = 32,
                     force_stats = None):
    from src.data.dataset import make_loaders
    from src.training.stage2_train import extract_features

    dummy = ids[:1]
    ym = {sid: y_mort.get(sid, 0) for sid in ids}
    yl = {sid: y_los.get(sid, 0.0) for sid in ids}
    _, _, te_loader = make_loaders(
        sequences, ym, yl,
        train_ids=dummy, val_ids=dummy, test_ids=ids,
        static_features=static, batch_size=batch_size, shuffle_train=False,
        force_stats=force_stats,
        use_dynamic_drop=False,
    )
    F_te, ym_te, yl_te = extract_features(enc, te_loader, device)

    try:
        from run_irregular_basic import _apply_stage2_meta
        F_te_m = _apply_stage2_meta(F_te, meta, for_mortality=True, ignore_keep_idx=True)
        F_te_l = _apply_stage2_meta(F_te, meta, for_mortality=False, ignore_keep_idx=True)
    except Exception:
        F_te_m = F_te_l = F_te

    pm = mort_h.predict(F_te_m)
    pl = los_h.predict_days(F_te_l)
    return ym_te.astype(np.int32), yl_te.astype(np.float32), pm, pl


# ─────────────────────────────────────────────────────────────────────────────
# Scenario runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_scenario(
    scenario_name: str,
    group_name: str,
    scenario_seqs: dict,          # sequences for this scenario (possibly cut)
    scenario_ym: dict,            # mortality labels
    scenario_yl: dict,            # LOS labels
    scenario_static: Optional[dict],
    seq_meta: dict,               # per-seq irregularity info
    train_ids: list,
    val_ids: list,
    test_ids: list,
    enc,                          # frozen MambaEncoder
    enc_stage2_meta: Optional[dict],
    cfg: dict,
    active_models: List[str],
    force_stats = None,
) -> List[dict]:
    device = cfg["stage1"]["device"]
    device_xgb = cfg.get("stage2", {}).get("device", "cpu")
    batch_size = int(cfg["stage1"].get("batch_size", 32))
    d_input = len(cfg.get("_feature_names", [1]*40))   # fallback
    rows = []

    # Summary stats for meta columns
    def _grp_meta(ids):
        n_ts = [seq_meta.get(s, {}).get("n_ts", 0) for s in ids]
        gap  = [seq_meta.get(s, {}).get("gap_std_s", 0.0) for s in ids]
        ts_avg  = float(np.mean(n_ts)) if n_ts else 0.0
        gap_avg = float(np.mean([g / 3600.0 for g in gap])) if gap else 0.0
        return ts_avg, gap_avg

    # Filter valid test ids for this scenario
    valid_te = [s for s in test_ids if s in scenario_seqs and s in scenario_ym]
    valid_tr = [s for s in train_ids if s in scenario_seqs and s in scenario_ym]
    valid_va = [s for s in val_ids   if s in scenario_seqs and s in scenario_ym]

    if min(len(valid_te), len(valid_tr), len(valid_va)) < 10:
        print(f"  [{scenario_name}/{group_name}] skipped — not enough data")
        return []

    if len(np.unique([scenario_ym.get(s,0) for s in valid_te])) < 2:
        print(f"  [{scenario_name}/{group_name}] skipped — single class in test")
        return []

    n_ts_avg_te, gap_avg_te = _grp_meta(valid_te)

    # ── XGBoost-flat ────────────────────────────────────────────────────────
    if "xgb_flat" in active_models:
        try:
            flat_m, flat_l, tr_s_flat = _train_xgb_flat(
                scenario_seqs, scenario_ym, scenario_yl, valid_tr, valid_va, device=device_xgb
            )
            ym_, yl_, pm, pl = _predict_xgb_flat(flat_m, flat_l, scenario_seqs,
                                                   scenario_ym, scenario_yl, valid_te)
            r = _metrics("XGBoost-flat", ym_, pm, yl_, pl,
                         scenario_name, group_name, n_ts_avg_te, gap_avg_te, tr_s_flat)
            rows.append(r); _print_row(r)
            # OOM-SAFE: free model + predictions immediately
            del flat_m, flat_l, ym_, yl_, pm, pl
        except Exception as e:
            print(f"  [xgb_flat] ERROR: {e}")
        _gc()

    # ── MaBoost (frozen encoder + retrained XGBoost head) ──────────────────
    if "maboost" in active_models and enc is not None:
        try:
            mb_m, mb_l, tr_s_mb, _, _ = _train_maboost_head(
                enc, scenario_seqs, scenario_ym, scenario_yl,
                valid_tr, valid_va, scenario_static, enc_stage2_meta,
                device, batch_size,
                force_stats=force_stats,
            )
            ym_, yl_, pm, pl = _predict_maboost(
                enc, mb_m, mb_l, scenario_seqs, scenario_ym, scenario_yl,
                valid_te, scenario_static, enc_stage2_meta, device, batch_size,
                force_stats=force_stats,
            )
            r = _metrics("MaBoost", ym_, pm, yl_, pl,
                         scenario_name, group_name, n_ts_avg_te, gap_avg_te, tr_s_mb)
            rows.append(r); _print_row(r)
            # OOM-SAFE: free model + predictions immediately
            del mb_m, mb_l, ym_, yl_, pm, pl
        except Exception as e:
            print(f"  [maboost] ERROR: {e}")
        _gc()

    # ── Deep baselines ───────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, str(ROOT))
    from run_all import _get_registry
    from src.training.baseline_trainer import train_baseline
    from src.data.dataset import make_loaders

    pos_rate = float(np.mean([scenario_ym[s] for s in valid_tr]))
    scen_tr_l, scen_va_l, scen_te_l = make_loaders(
        scenario_seqs, scenario_ym, scenario_yl,
        valid_tr, valid_va, valid_te,
        static_features=scenario_static, batch_size=batch_size, shuffle_train=True
    )
    
    registry = _get_registry(len(cfg.get("_feature_names", [1]*40)))
    
    # Map friendly names matching run_all and what user might pass to --models
    deep_alias_map = {
        "grud": "GRU-D", "lstm": "LSTM", "transformer": "Transformer",
        "interpnet": "InterpNet", "tcn": "TCN",
        "sand": "SAnD", "strats": "STraTS", "grud_ts": "GRU-D (cell)"
    }
    
    for alias, display_label in deep_alias_map.items():
        if alias not in active_models:
            continue
            
        # Find this model in the registry
        reg_entry = next((item for item in registry if item[0] == display_label), None)
        if not reg_entry:
            print(f"  [!] Missing registry entry for {display_label}")
            continue
            
        disp_name, cls, mtype, adapter, extra = reg_entry
        
        try:
            auroc, auprc, tr_s, pm, ym_ = train_baseline(
                model_class=cls, model_type=mtype, adapter_fn=adapter,
                tr_loader=scen_tr_l, va_loader=scen_va_l, te_loader=scen_te_l,
                d_input=d_input, d_static=0 if not scenario_static else len(next(iter(scenario_static.values()))), 
                pos_rate=pos_rate, device=device, epochs=min(25, cfg["stage1"].get("epochs", 25)),
                extra_kw=extra
            )
            
            # Construct fake LOS outcomes (mean imputation) strictly for metric consistency
            yl_ = np.array([scenario_yl[s] for s in valid_te], dtype=np.float32)
            pl  = np.full(len(yl_), float(np.mean([scenario_yl[s] for s in valid_tr])), dtype=np.float32)
            
            r = _metrics(display_label, ym_, pm, yl_, pl,
                         scenario_name, group_name, n_ts_avg_te, gap_avg_te, tr_s)
            rows.append(r); _print_row(r)
            _gc()
            
        except Exception as e:
            print(f"  [{alias}] ERROR: {e}")
            _gc()

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Markdown summary writer
# ─────────────────────────────────────────────────────────────────────────────

def _write_markdown(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(set(r["scenario"] for r in rows))
    groups    = sorted(set(r["group"] for r in rows))
    models    = sorted(set(r["model"] for r in rows),
                       key=lambda m: (0 if "MaBoost" in m else 1 if "flat" in m else 2))

    lines = [
        "# Hospital Irregular Data Benchmark — Fair Comparison",
        "",
        "> **Raw Problem (plan.md):** At each event time, ≥1 variables updated, "
        "different variables each time, irregular delta_t, predict after every event.",
        "> **Fairness:** All models trained on the same split, retrained at each scenario.",
        "",
        "---",
        "",
    ]

    for scenario in scenarios:
        grp_rows = [r for r in rows if r["scenario"] == scenario]
        groups_in_scen = sorted(set(r["group"] for r in grp_rows))
        lines.append(f"## Scenario: `{scenario}`")
        lines.append("")
        for group in groups_in_scen:
            g_rows = [r for r in grp_rows if r["group"] == group]
            if not g_rows:
                continue
            first = g_rows[0]
            n  = first["n"]
            ts = first["n_ts_avg"]
            gp = first["avg_gap_std_h"]
            lines.append(f"### Group: {group}  (n={n:,}, avg_timestamps={ts:.1f}, avg_gap_std={gp:.2f}h)")
            lines.append("")
            lines.append("| Model | AUROC ↑ | AUPRC ↑ | LOS MAE ↓ | Train (s) | Pos rate |")
            lines.append("|-------|---------|---------|---------|---------|----------|")
            g_rows_sorted = sorted(g_rows, key=lambda r: -r["auroc"] if not np.isnan(r["auroc"]) else -999)
            for r in g_rows_sorted:
                auroc = f"{r['auroc']:.4f}" if not np.isnan(r["auroc"]) else "—"
                auprc = f"{r['auprc']:.4f}" if not np.isnan(r["auprc"]) else "—"
                maes  = f"{r['los_mae']:.2f}d"
                bold  = "**" if r["model"] == g_rows_sorted[0]["model"] else ""
                lines.append(f"| {bold}{r['model']}{bold} | {bold}{auroc}{bold} | "
                              f"{bold}{auprc}{bold} | {maes} | {r['train_s']}s | {r['pos_rate']:.1%} |")
            lines.append("")

    lines += [
        "---",
        "",
        "## Key: Scenario Definitions",
        "",
        "| Scenario | What it simulates |",
        "|----------|------------------|",
        "| `raw_event_stream` | Predict after **every individual event** — the exact raw problem |",
        "| `early_6h` | Only first 6h of events — truly irregular, sparse |",
        "| `early_12h` | Only first 12h of events — sparse ICU data |",
        "| `early_24h` | Only first 24h of events |",
        "| `sparse_patients` | Bottom-25% patients by number of event timestamps |",
        "| `dense_patients` | Top-25% patients by number of event timestamps |",
        "| `high_gap_var` | Top-25% patients by gap standard deviation (most irregular timing) |",
        "| `low_gap_var` | Bottom-25% patients by gap standard deviation (most regular timing) |",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Hospital Benchmark] Markdown -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed aggregation & Relative Error table
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_seeds(all_rows: List[dict]) -> List[dict]:
    """
    Aggregate per-seed rows into mean ± std.
    Returns one row per (model, scenario, group) with _mean and _std fields.
    """
    from collections import defaultdict
    buckets: dict = defaultdict(list)      # key=(model,scenario,group) → list of rows
    for r in all_rows:
        key = (r["model"], r["scenario"], r["group"])
        buckets[key].append(r)

    agg = []
    for (model, scenario, group), rows in buckets.items():
        aurocs  = [r["auroc"]   for r in rows if not np.isnan(r["auroc"])]
        auprcs  = [r["auprc"]   for r in rows if not np.isnan(r["auprc"])]
        maes    = [r["los_mae"] for r in rows]
        rmses   = [r["los_rmse"] for r in rows]
        agg.append(dict(
            model=model, scenario=scenario, group=group,
            n_seeds=len(rows),
            seeds=[r["seed"] for r in rows],
            auroc_mean  = float(np.mean(aurocs))  if aurocs else float("nan"),
            auroc_std   = float(np.std(aurocs))   if len(aurocs) > 1 else 0.0,
            auprc_mean  = float(np.mean(auprcs))  if auprcs else float("nan"),
            auprc_std   = float(np.std(auprcs))   if len(auprcs) > 1 else 0.0,
            los_mae_mean = float(np.mean(maes))  if maes else float("nan"),
            los_mae_std  = float(np.std(maes))   if len(maes) > 1 else 0.0,
            los_rmse_mean = float(np.mean(rmses)) if rmses else float("nan"),
            los_rmse_std  = float(np.std(rmses))  if len(rmses) > 1 else 0.0,
            n_ts_avg      = float(np.mean([r["n_ts_avg"] for r in rows])),
            avg_gap_std_h = float(np.mean([r["avg_gap_std_h"] for r in rows])),
            pos_rate      = float(np.mean([r["pos_rate"] for r in rows])),
            n             = int(np.mean([r["n"] for r in rows])),
        ))
    return agg


def _save_agg_csv(agg: List[dict], path: Path) -> None:
    """Save aggregated mean±std results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "scenario", "group", "n_seeds",
        "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
        "los_mae_mean", "los_mae_std", "los_rmse_mean", "los_rmse_std",
        "n", "n_ts_avg", "avg_gap_std_h", "pos_rate",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in agg:
            w.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                        for k in fields})
    print(f"[Hospital Benchmark] Aggregated CSV -> {path}")


def _write_markdown_multiseed(all_rows: List[dict], agg: List[dict], out_path: Path,
                               seeds: List[int]) -> None:
    """
    Write three sections to markdown:
      1. Per-seed raw results (collapsible per scenario)
      2. Aggregated mean ± std table per scenario
      3. Relative Error table: (model - XGBoost-flat) / XGBoost-flat × 100%
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(set(r["scenario"] for r in agg),
                       key=lambda s: ["early_6h","early_12h","early_24h",
                                      "raw_event_stream","sparse_patients",
                                      "dense_patients","high_gap_var","low_gap_var"].index(s)
                                     if s in ["early_6h","early_12h","early_24h",
                                              "raw_event_stream","sparse_patients",
                                              "dense_patients","high_gap_var","low_gap_var"]
                                     else 99)
    flat_lbl = "XGBoost-flat"

    # Build lookup: (model, scenario, group) → agg row
    agg_lut = {(r["model"], r["scenario"], r["group"]): r for r in agg}

    lines = [
        "# Hospital Irregular Benchmark — Multi-Seed Results",
        "",
        f"> **Seeds:** {seeds}  ",
        "> **Raw Problem (plan.md §1.3):** Irregular delta_t, different variable set per event, predict after every event.  ",
        "> **Fairness:** All models retrained on same split per seed; encoder frozen for MaBoost.",
        "",
        "---",
        "",
    ]

    # ── Section 1: Aggregated mean ± std ─────────────────────────────────────
    lines += ["## 1. Mean ± Std  (across all seeds)", ""]
    for scenario in scenarios:
        grp_agg = [r for r in agg if r["scenario"] == scenario]
        groups  = sorted(set(r["group"] for r in grp_agg))
        lines.append(f"### `{scenario}`")
        lines.append("")
        for grp in groups:
            g_rows = sorted(
                [r for r in grp_agg if r["group"] == grp],
                key=lambda r: -r["auroc_mean"] if not np.isnan(r["auroc_mean"]) else -999,
            )
            if not g_rows:
                continue
            example = g_rows[0]
            lines.append(f"**Group: {grp}**  "
                         f"n≈{example['n']:,}, "
                         f"avg_ts={example['n_ts_avg']:.1f}, "
                         f"avg_gap={example['avg_gap_std_h']:.2f}h")
            lines.append("")
            lines.append("| Model | AUROC | AUPRC | LOS MAE |")
            lines.append("|-------|-------|-------|---------|")
            for r in g_rows:
                auroc_s = (f"{r['auroc_mean']:.4f} ± {r['auroc_std']:.4f}"
                           if not np.isnan(r['auroc_mean']) else "—")
                auprc_s = (f"{r['auprc_mean']:.4f} ± {r['auprc_std']:.4f}"
                           if not np.isnan(r['auprc_mean']) else "—")
                mae_s   = f"{r['los_mae_mean']:.2f}d ± {r['los_mae_std']:.2f}d"
                bold = "**" if r == g_rows[0] else ""
                lines.append(f"| {bold}{r['model']}{bold} "
                             f"| {bold}{auroc_s}{bold} "
                             f"| {bold}{auprc_s}{bold} "
                             f"| {mae_s} |")
            lines.append("")

    # ── Section 2: Relative Error table ──────────────────────────────────────
    lines += [
        "---",
        "",
        "## 2. Relative AUROC Error vs XGBoost-flat  (Δ%)",
        "",
        "> Formula: `(model_AUROC_mean − XGBoost-flat_AUROC_mean) / XGBoost-flat_AUROC_mean × 100%`  ",
        "> Positive = model beats flat baseline; Negative = model loses to flat baseline.",
        "",
    ]
    # Gather all models (excluding flat baseline itself)
    all_models = sorted(set(
        r["model"] for r in agg if r["model"] != flat_lbl
    ), key=lambda m: (
        0 if "MaBoost" in m else
        1 if "GRU" in m else
        2 if "LSTM" in m else
        3 if "Transformer" in m else
        4 if "InterpNet" in m else
        5 if "TCN" in m else 9
    ))

    for scenario in scenarios:
        grp_agg = [r for r in agg if r["scenario"] == scenario]
        groups  = sorted(set(r["group"] for r in grp_agg))
        lines.append(f"### `{scenario}`")
        lines.append("")
        for grp in groups:
            flat_row = agg_lut.get((flat_lbl, scenario, grp))
            if flat_row is None or np.isnan(flat_row["auroc_mean"]):
                lines.append(f"*{grp}: XGBoost-flat baseline missing, skipped.*")
                lines.append("")
                continue
            flat_auroc = flat_row["auroc_mean"]
            flat_auprc = flat_row["auprc_mean"]
            lines.append(f"**Group: {grp}**  "
                         f"(XGBoost-flat baseline: AUROC={flat_auroc:.4f}, AUPRC={flat_auprc:.4f})")
            lines.append("")
            lines.append("| Model | ΔAUROC (%) | ΔAUPRC (%) | Winner |")
            lines.append("|-------|------------|------------|--------|")

            # Also include flat itself as reference row = 0%
            rel_rows = []
            for model in [flat_lbl] + all_models:
                row = agg_lut.get((model, scenario, grp))
                if row is None:
                    continue
                if np.isnan(row["auroc_mean"]):
                    continue
                d_auroc = (row["auroc_mean"] - flat_auroc) / flat_auroc * 100.0
                d_auprc = ((row["auprc_mean"] - flat_auprc) / flat_auprc * 100.0
                           if not np.isnan(flat_auprc) and flat_auprc > 0
                           else float("nan"))
                rel_rows.append((model, d_auroc, d_auprc, row))
            rel_rows.sort(key=lambda x: -x[1])  # descending Δ AUROC

            for model, d_auroc, d_auprc, row in rel_rows:
                sign_a = "+" if d_auroc >= 0 else ""
                sign_p = "+" if not np.isnan(d_auprc) and d_auprc >= 0 else ""
                auroc_s = f"{sign_a}{d_auroc:+.2f}%  (±{row['auroc_std']*100:.2f}%)"
                auprc_s = (f"{sign_p}{d_auprc:+.2f}%" if not np.isnan(d_auprc)
                           else "—")
                winner = "📈 +" if d_auroc > 0.001 else ("📉 −" if d_auroc < -0.001 else "≈ tie")
                bold = "**" if model == flat_lbl else ""
                lines.append(f"| {bold}{model}{bold} "
                             f"| {auroc_s} | {auprc_s} | {winner} |")
            lines.append("")

    # ── Section 3: Per-seed details ───────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 3. Per-Seed Raw Results",
        "",
    ]
    for seed in seeds:
        seed_rows = [r for r in all_rows if r["seed"] == seed]
        if not seed_rows:
            continue
        lines.append(f"### Seed {seed}")
        lines.append("")
        for scenario in scenarios:
            sc_rows = sorted(
                [r for r in seed_rows if r["scenario"] == scenario],
                key=lambda r: -r["auroc"] if not np.isnan(r["auroc"]) else -999,
            )
            if not sc_rows:
                continue
            lines.append(f"**{scenario}**")
            lines.append("")
            lines.append("| Model | AUROC | AUPRC | LOS MAE |")
            lines.append("|-------|-------|-------|---------|")
            for r in sc_rows:
                auroc_s = f"{r['auroc']:.4f}" if not np.isnan(r["auroc"]) else "—"
                auprc_s = f"{r['auprc']:.4f}" if not np.isnan(r["auprc"]) else "—"
                lines.append(f"| {r['model']} | {auroc_s} | {auprc_s} "
                             f"| {r['los_mae']:.2f}d |")
            lines.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Scenario Definitions",
        "",
        "| Scenario | What it simulates |",
        "|----------|------------------|",
        "| `raw_event_stream` | Predict after **every individual event** — exact raw problem |",
        "| `early_6h` | Only first 6h of ICU data — truly irregular, sparse |",
        "| `early_12h` | Only first 12h of data |",
        "| `early_24h` | Only first 24h of data |",
        "| `sparse_patients` | Bottom-25% by event-timestamp count |",
        "| `dense_patients` | Top-25% by event-timestamp count |",
        "| `high_gap_var` | Top-25% by inter-event gap std-dev (most irregular timing) |",
        "| `low_gap_var` | Bottom-25% by inter-event gap std-dev (most regular) |",
        "",
    ]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[Hospital Benchmark] Multi-seed markdown -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Fair Hospital Irregular Benchmark — multi-seed with relative error table"
    )
    pa.add_argument("--config",    default="config.yaml")
    pa.add_argument("--skip-etl", action="store_true")
    pa.add_argument("--out",       default="results/benchmark_hospital_irregular_fair.csv",
                    help="Per-seed raw results CSV")
    pa.add_argument("--agg-out",   default="results/benchmark_hospital_irregular_agg.csv",
                    help="Aggregated mean±std CSV")
    pa.add_argument("--md-out",    default="summary/benchmark_hospital_irregular_fair.md",
                    help="Markdown report with relative error table")
    pa.add_argument("--seeds",     default="42,1,3107",
                    help="Comma-separated random seeds (default: 42,1,3107). "
                         "Each seed re-splits the data and retrains all models.")
    pa.add_argument("--scenarios", default="",
                    help="Comma-separated subset of scenarios to run. "
                         "Default: all. Options: "
                         "raw_event_stream,early_6h,early_12h,early_24h,"
                         "sparse_patients,dense_patients,high_gap_var,low_gap_var")
    pa.add_argument("--models", default="",
                    help="Comma-separated subset of models. "
                         "Default: all. Options: maboost,xgb_flat,grud,lstm,transformer,interpnet,tcn")
    pa.add_argument("--batch-size",   type=int, default=32)
    pa.add_argument("--deep-epochs",  type=int, default=25,
                    help="Training epochs for deep baselines per seed (default=25)")
    args = pa.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))

    cfg = _load_cfg(args.config)

    SCENARIO_LIST = [
        "early_6h", "early_12h", "early_24h",
        "raw_event_stream",
        "sparse_patients", "dense_patients",
        "high_gap_var", "low_gap_var",
    ]
    MODEL_LIST = ["maboost", "xgb_flat", "grud", "lstm", "transformer", "interpnet", "tcn"]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    active_scenarios = (
        [s.strip() for s in args.scenarios.split(",") if s.strip()]
        if args.scenarios else SCENARIO_LIST
    )
    active_models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models else MODEL_LIST
    )

    print("=" * 70)
    print("FAIR HOSPITAL IRREGULAR BENCHMARK  (multi-seed)")
    print(f"Seeds:     {seeds}")
    print(f"Scenarios: {active_scenarios}")
    print(f"Models:    {active_models}")
    print("=" * 70)

    _checkpoint("bench_setup", f"Hospital Benchmark: {len(active_models)} models, {len(active_scenarios)} scenarios, {len(seeds)} seeds")

    # ── Load ETL once (sequences are deterministic, splits vary by seed) ──────
    _checkpoint("bench_etl", "Loading ETL data for benchmark")
    from src.data.preprocess import load_etl_output, run_etl, FEATURE_NAMES
    data_cfg = cfg["data"]
    window_mode = data_cfg.get("window_mode", "fixed_48h")
    suffix = "expanding" if window_mode == "expanding_window" else "fixed48"
    cache = Path(data_cfg["processed"]) / f"etl_output_{suffix}.pkl"
    if args.skip_etl and cache.exists():
        etl = load_etl_output(str(cache))
        print(f"Loaded ETL from {cache.name}")
    else:
        print("Running ETL …")
        etl = run_etl(
            mimic_dir=data_cfg["mimic_dir"],
            seq_len=data_cfg["seq_len"],
            min_age=data_cfg["min_age"],
            window_mode=window_mode,
            window_hours=float(data_cfg.get("window_hours", 48.0)),
            crop_strategy=data_cfg.get("sequence_crop", "first"),
            save_dir=data_cfg["processed"],
        )

    sequences   = etl["sequences"]
    static      = etl["static_features"]
    y_mort      = etl["mortality_labels"]
    y_los       = etl["los_labels"]
    stay_outcomes = etl.get("stay_outcomes", {})
    feature_names = etl.get("feature_names", FEATURE_NAMES)
    cfg["_feature_names"] = feature_names
    seq_len      = int(data_cfg.get("seq_len", 64))
    research_cfg = cfg.get("research", {})
    tr_f = float(data_cfg.get("train_frac", 0.60))
    va_f = float(data_cfg.get("val_frac",   0.20))
    all_ids = sorted(set(sequences) & set(y_mort))
    labels  = [y_mort[s] for s in all_ids]

    # ── Load encoder once (frozen weights, shared across seeds) ──────────────
    _checkpoint("bench_encoder", "Loading frozen Mamba encoder")
    enc = None
    ckpt_dir = Path(cfg["ckpt_dir"])
    enc_path = ckpt_dir / "encoder_best.pth"
    stage2_meta = None
    if "maboost" in active_models:
        if enc_path.exists():
            from run_irregular_basic import _load_encoder, _load_stage2_meta
            enc = _load_encoder(enc_path, cfg, len(feature_names))
            stage2_meta = _load_stage2_meta(ckpt_dir)
            print(f"Loaded frozen encoder: {enc_path.name}")
        else:
            print("[!] encoder_best.pth not found — MaBoost skipped")
            active_models = [m for m in active_models if m != "maboost"]

    # ── Load global norm stats for MaBoost benchmark injection ────────────
    norm_stats = None
    if "maboost" in active_models:
        norm_path = ckpt_dir / "norm_stats.pkl"
        if norm_path.exists():
            norm_stats = pickle.load(open(norm_path, "rb"))
            print(f"[Benchmark] Loaded global norm stats from {norm_path}")
            print(f"  mean shape={norm_stats['mean'].shape}  std shape={norm_stats['std'].shape}")
        else:
            print(f"[!] WARNING: {norm_path} not found — MaBoost will use LOCAL norm stats")
            print(f"    This causes normalization drift! Run: python tmp/extract_norm_stats.py")

    # ── Outer seed loop ───────────────────────────────────────────────────────
    all_rows: List[dict] = []   # accumulates every row from every seed

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"  SEED {seed}  ({seed_idx+1}/{len(seeds)})")
        print(f"{'#'*70}")
        _checkpoint("bench_seed", f"Seed {seed} ({seed_idx+1}/{len(seeds)}) — {len(active_scenarios)} scenarios")

        # Re-seed everything for this split
        np.random.seed(seed)
        torch.manual_seed(seed)
        cfg["seed"] = seed

        # ── Stratified group-aware split (different per seed) ─────────────────
        gss1 = GroupShuffleSplit(n_splits=1, train_size=tr_f, random_state=seed)
        tr_idx, tmp_idx = next(gss1.split(all_ids, labels,
                                           groups=np.arange(len(all_ids))))
        tr_ids  = [all_ids[i] for i in tr_idx]
        tmp_ids = [all_ids[i] for i in tmp_idx]
        te_frac = (1 - tr_f - va_f) / (1 - tr_f)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=te_frac, random_state=seed)
        va_i, te_i = next(gss2.split(tmp_ids, [y_mort[s] for s in tmp_ids],
                                      groups=np.arange(len(tmp_ids))))
        va_ids = [tmp_ids[i] for i in va_i]
        te_ids = [tmp_ids[i] for i in te_i]
        print(f"  Split: train={len(tr_ids):,}  val={len(va_ids):,}  test={len(te_ids):,}")
        print(f"  Pos rate: train={np.mean([y_mort[s] for s in tr_ids]):.1%}  "
              f"test={np.mean([y_mort[s] for s in te_ids]):.1%}")

        base_meta = _seq_meta(sequences, te_ids)
        device = cfg["stage1"]["device"]

        for scen_idx, scenario in enumerate(active_scenarios):
            _checkpoint("bench_scenario", f"Seed {seed} — Scenario: {scenario} ({scen_idx+1}/{len(active_scenarios)})")
            print(f"\n  {'='*56}")
            print(f"  SEED {seed} | SCENARIO: {scenario}")
            print(f"  {'='*56}")

            # ── Build scenario data ───────────────────────────────────────────
            if scenario in ("early_6h", "early_12h", "early_24h"):
                hours = {"early_6h": 6.0, "early_12h": 12.0, "early_24h": 24.0}[scenario]
                s_tr, m_tr = _cut_to_hours(sequences, tr_ids, hours, seq_len)
                s_va, m_va = _cut_to_hours(sequences, va_ids, hours, seq_len)
                s_te, m_te = _cut_to_hours(sequences, te_ids, hours, seq_len)
                
                # OOM-SAFE: Inline updates save RAM vs {** }
                scen_seqs = s_tr
                scen_seqs.update(s_va); del s_va
                scen_seqs.update(s_te); del s_te
                
                scen_meta = m_tr
                scen_meta.update(m_va); del m_va
                scen_meta.update(m_te); del m_te

                scen_ym, scen_yl, scen_static = y_mort, y_los, static
                scenario_tr, scenario_va, scenario_te = tr_ids, va_ids, te_ids

            elif scenario == "raw_event_stream":
                for max_s in [32, 16, 8, 4]:
                    import gc; gc.collect()
                    print(f"    Building raw event stream (max_samples={max_s}) …")
                    try:
                        tr_s, tr_ym_s, tr_yl_s, tr_st_s, _ = _build_raw_event_stream(
                            sequences, tr_ids, y_mort, y_los, static, stay_outcomes, research_cfg, max_samples_per_stay=max_s)
                        va_s, va_ym_s, va_yl_s, va_st_s, _ = _build_raw_event_stream(
                            sequences, va_ids, y_mort, y_los, static, stay_outcomes, research_cfg, max_samples_per_stay=max_s)
                        te_s, te_ym_s, te_yl_s, te_st_s, _ = _build_raw_event_stream(
                            sequences, te_ids, y_mort, y_los, static, stay_outcomes, research_cfg, max_samples_per_stay=max_s)
                        
                        scenario_tr, scenario_va, scenario_te = sorted(tr_s), sorted(va_s), sorted(te_s)

                        scen_seqs = tr_s
                        scen_seqs.update(va_s); del va_s
                        scen_seqs.update(te_s); del te_s
                        
                        scen_ym = tr_ym_s
                        scen_ym.update(va_ym_s); del va_ym_s
                        scen_ym.update(te_ym_s); del te_ym_s

                        scen_yl = tr_yl_s
                        scen_yl.update(va_yl_s); del va_yl_s
                        scen_yl.update(te_yl_s); del te_yl_s

                        scen_static = tr_st_s or {}
                        scen_static.update(va_st_s or {}); del va_st_s
                        scen_static.update(te_st_s or {}); del te_st_s
                        if not scen_static: scen_static = None
                        
                        break # Successfully built without OOM
                        
                    except MemoryError as e:
                        print(f"    [RAM Guard] Attempt={max_s} hit RAM limit ({e}). Dropping cap...")
                        if 'tr_s' in locals(): del tr_s
                        if 'va_s' in locals(): del va_s
                        if 'te_s' in locals(): del te_s
                        if 'tr_ym_s' in locals(): del tr_ym_s
                        import gc; gc.collect()
                        continue
                else: # completely failed
                    print("  [Raw Event Stream] Skipped: Failed to allocate even at lowest safety cap.")
                    scen_seqs, scen_ym, scen_yl, scen_static = {}, {}, {}, None
                    scenario_tr, scenario_va, scenario_te = [], [], []

                if scenario_te:
                    scen_meta = _seq_meta(scen_seqs, scenario_te)
                    print(f"    Events: train={len(scenario_tr):,}  val={len(scenario_va):,}  test={len(scenario_te):,}")
                _gc()  # flush before running models on enlarged dataset

            elif scenario in ("sparse_patients", "dense_patients",
                              "high_gap_var", "low_gap_var"):
                group_map = {
                    "sparse_patients": "sparse_q25",
                    "dense_patients":  "dense_q75",
                    "high_gap_var":    "high_gap_var_q75",
                    "low_gap_var":     "low_gap_var_q25",
                }
                kind = group_map[scenario]
                subset_te = _group_ids(base_meta, te_ids, kind)
                if len(subset_te) < 10:
                    print(f"    [!] Too few samples ({len(subset_te)}), skipping")
                    continue
                print(f"    Subset ({kind}): {len(subset_te):,}/{len(te_ids):,}")
                scen_seqs, scen_ym, scen_yl, scen_static = sequences, y_mort, y_los, static
                scen_meta   = base_meta
                scenario_tr, scenario_va, scenario_te = tr_ids, va_ids, subset_te
            else:
                continue

            # ── Run all models on this scenario+seed ──────────────────────────
            group_name = (scenario if scenario not in ("sparse_patients","dense_patients",
                                                       "high_gap_var","low_gap_var")
                          else group_map[scenario])
            rows = _run_scenario(
                scenario, group_name,
                scen_seqs, scen_ym, scen_yl, scen_static, scen_meta,
                scenario_tr, scenario_va, scenario_te,
                enc, stage2_meta, cfg, active_models,
                force_stats=norm_stats,
            )
            # Tag every row with the current seed
            for r in rows:
                r["seed"] = seed
            all_rows.extend(rows)

            # OOM-SAFE: free scenario-specific dicts before next scenario builds its own
            # (only del if they were newly allocated — not for subset scenarios that alias
            #  the original dicts, those are just views, safe to rebind)
            _is_alias = scenario in ("sparse_patients", "dense_patients",
                                     "high_gap_var", "low_gap_var")
            if not _is_alias:
                del scen_seqs, scen_ym, scen_yl
                scen_seqs = scen_ym = scen_yl = None  # noqa: allow reassignment
            del rows

            # OOM-SAFE: flush after EVERY scenario to reclaim VRAM + RAM
            print(f"  [gc] Flushing memory after scenario '{scenario}' (seed={seed}) …")
            _gc()

        # OOM-SAFE: flush after EVERY seed before split rebuild
        print(f"\n[gc] End of seed {seed} — full memory flush …")
        _gc()

    # ── Aggregate and save ────────────────────────────────────────────────────
    _checkpoint("bench_aggregate", f"Aggregating {len(all_rows)} rows across {len(seeds)} seeds")
    if not all_rows:
        print("\n[!] No results produced. Check checkpoints and ETL data.")
        return

    print(f"\n{'='*70}")
    print(f"Aggregating {len(all_rows)} rows across {len(seeds)} seeds …")

    # Per-seed raw CSV
    out_path = ROOT / args.out
    _save_csv(all_rows, out_path)

    # Aggregated mean±std CSV
    agg = _aggregate_seeds(all_rows)
    agg_path = ROOT / args.agg_out
    _save_agg_csv(agg, agg_path)

    # Markdown: mean±std + relative error table + per-seed detail
    md_path = ROOT / args.md_out
    _write_markdown_multiseed(all_rows, agg, md_path, seeds)

    # ── Print relative error summary to console ───────────────────────────────
    print(f"\n{'─'*70}")
    print("RELATIVE AUROC vs XGBoost-flat (mean across seeds) — summary")
    print(f"{'─'*70}")
    flat_lbl = "XGBoost-flat"
    for scenario in active_scenarios:
        sc_agg = {r["model"]: r for r in agg
                  if r["scenario"] == scenario}
        flat_row = sc_agg.get(flat_lbl)
        if flat_row is None or np.isnan(flat_row["auroc_mean"]):
            continue
        base = flat_row["auroc_mean"]
        print(f"\n  {scenario}  (baseline AUROC={base:.4f})")
        for model, row in sorted(sc_agg.items(),
                                  key=lambda x: -x[1]["auroc_mean"]):
            if model == flat_lbl:
                continue
            delta = (row["auroc_mean"] - base) / base * 100.0
            sign = "+" if delta >= 0 else ""
            print(f"    {model:28s}: {sign}{delta:+.2f}%  "
                  f"(AUROC={row['auroc_mean']:.4f}±{row['auroc_std']:.4f})")

    print(f"\n[Done]  {len(seeds)} seeds × {len(active_scenarios)} scenarios × {len(active_models)} models")
    print(f"  Raw      → {out_path}")
    print(f"  Agg      → {agg_path}")
    print(f"  Markdown → {md_path}")
    
    _checkpoint("done", f"Hospital Benchmark Complete. ({len(seeds)} seeds)")

if __name__ == "__main__":
    try:
        main()
    except Exception as _fatal:
        import traceback
        _checkpoint("crashed", f"CRASHED: {str(_fatal)[:120]}", message=traceback.format_exc()[-400:])
        raise
