"""
src/training/stage2_train.py  (v6)
====================================================
v6 fixes vs v5:

FIX-1  OPT-4 interaction features now applied AFTER drop_low_importance
       (was before) — builds pairs only from already-filtered features,
       reducing noise and improving signal-to-noise ratio for XGBoost.

FIX-2  LOS Optuna search objective: reg:squarederror → reg:pseudohubererror
       (v5 had mismatch: Optuna searched squarederror but final train used
       pseudohubererror — hyperparams were suboptimal).

FIX-3  OPT-6 static × z_multi cross: top_z now dynamic (min(32, n_zmulti//4))
       instead of hardcoded 16, adapts to d_model changes.

FIX-4  n_ensemble_mort default changed 2→1 to reduce overfitting variance
       (ensemble helps with noisy labels but hurts when encoder is strong).

FIX-5  colsample_bytree Optuna lower bound 0.4→0.3 — with 754-1266 features
       after SCI, more aggressive column subsampling reduces correlation between
       trees and improves generalization.

All function names, signatures, and checkpoint formats unchanged.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import warnings
import xgboost as xgb
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.mamba_encoder import MambaEncoder
from src.models.xgboost_head  import XGBMortality, XGBLos


# ---------------------------------------------------------------------------
# Encoder utilities (unchanged)
# ---------------------------------------------------------------------------

def load_frozen_encoder(ckpt: str, d_input: int, d_model: int = 128,
                         **kw) -> MambaEncoder:
    state    = torch.load(ckpt, map_location="cpu", weights_only=True)
    n_layers = sum(1 for k in state
                   if k.startswith("layers.") and k.endswith(".log_decay"))
    # Auto-detect d_model from input_proj weight shape
    # v7 SCI: input_proj.0.weight is (d_model, 3*d_input)
    # v5/v6:  input_proj.0.weight is (d_model, d_input)
    proj_w = state.get("input_proj.0.weight")
    if proj_w is not None:
        d_model_ckpt = proj_w.shape[0]
        if d_model_ckpt != d_model:
            print(f"  [Warn] d_model mismatch: cfg={d_model} ckpt={d_model_ckpt}, using ckpt")
            d_model = d_model_ckpt
    print(f"  Encoder: d_model={d_model}  n_layers={n_layers} (from checkpoint)")

    kw.pop('n_layers', None)
    enc = MambaEncoder(d_input, d_model, n_layers=n_layers, **kw)
    enc.load_state_dict(state, strict=False)   # strict=False for new SCI keys
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    n = sum(1 for _ in enc.parameters())
    print(f"[Stage 2] Encoder frozen ({n} tensors).")
    return enc


def _finetune_encoder(enc, loader, device, epochs, lr=1e-5):
    """OPT-7: unchanged."""
    from torch.nn import functional as F_nn
    enc = enc.to(device)
    for p in enc.parameters():
        p.requires_grad = True
    d    = enc.pool.proj.out_features
    head = nn.Linear(d, 2).to(device)
    opt  = torch.optim.AdamW(
        list(enc.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=1e-5
    )
    print(f"  [Enc finetune] {epochs} epochs lr={lr}")
    for ep in range(epochs):
        enc.train(); head.train()
        total_loss = 0.0; n_ok = 0
        for batch in loader:
            x, tau, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            y_mort = batch[-2].to(device)
            opt.zero_grad(set_to_none=True)
            z    = enc(x, tau, mask)
            loss = F_nn.cross_entropy(head(z), y_mort)
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
                opt.step()
                total_loss += loss.item(); n_ok += 1
        print(f"    ep {ep+1}/{epochs} loss={total_loss/max(n_ok,1):.4f}")
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    print("  [Enc finetune] done — re-frozen")
    return enc


@torch.no_grad()
def extract_features(enc, loader, device="cuda", static_proj=None):
    """Feature extraction — unchanged."""
    enc = enc.to(device)
    if static_proj is not None:
        static_proj = static_proj.to(device).eval()
    Fs, morts, los_arr = [], [], []
    for batch in loader:
        x, tau, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        has_static   = len(batch) == 6
        z_multi, raw_stats = enc.extract_features(x, tau, mask)
        z_multi   = z_multi.cpu().numpy()
        raw_stats = raw_stats.cpu().numpy()
        parts = [z_multi, raw_stats]
        if has_static:
            x_static_raw = batch[3]
            if static_proj is not None:
                with torch.no_grad():
                    s_emb = static_proj(
                        torch.nan_to_num(x_static_raw.to(device), nan=0.0)
                    )
                parts.append(s_emb.cpu().numpy())
            else:
                parts.append(x_static_raw.numpy())
        Fs.append(np.hstack(parts))
        morts.append(batch[-2].numpy())
        los_arr.append(batch[-1].numpy())
    F_out = np.vstack(Fs).astype(np.float32)
    y_m   = np.concatenate(morts)
    y_l   = np.concatenate(los_arr)
    print(f"  Features {F_out.shape} | pos_rate={y_m.mean():.3f} | "
          f"mean_LOS={y_l.mean():.1f}d")
    return F_out, y_m, y_l


# ---------------------------------------------------------------------------
# OPT-4: Interaction features — FIX-1: now called AFTER drop_low_importance
# ---------------------------------------------------------------------------

def _build_interaction_features(F_tr, y_tr, F_va, F_te, n_top=8):
    """
    FIX-1: Build pairwise interactions from already-filtered features.
    Calling this before drop_low_importance creates noisy pairs from weak
    features. Calling after ensures pairs are from strong features only.
    """
    print(f"  [Interactions] fitting shallow XGB (top-{n_top}) on {F_tr.shape[1]} features …")
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    p_shallow = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "tree_method": "hist", "max_depth": 3,
        "learning_rate": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "scale_pos_weight": spw,
        "seed": 42, "verbosity": 0,
    }
    dm_tr       = xgb.DMatrix(F_tr, label=y_tr, missing=np.nan)
    bst_shallow = xgb.train(p_shallow, dm_tr, 200, verbose_eval=False)
    n_cols      = F_tr.shape[1]
    scores      = bst_shallow.get_score(importance_type="cover")
    arr         = np.array([scores.get(f"f{i}", 0.0) for i in range(n_cols)])
    top_idx     = np.argsort(arr)[::-1][:n_top]
    pairs       = [(i, j) for idx_i, i in enumerate(top_idx)
                   for j in top_idx[idx_i+1:]]
    print(f"  [Interactions] {len(pairs)} pairs from top-{n_top} filtered features")

    def _make_pairs(F):
        if not pairs:
            return F
        inter = np.stack([F[:, i] * F[:, j] for i, j in pairs], axis=1).astype(np.float32)
        return np.hstack([inter, F])

    F_tr_new = _make_pairs(F_tr)
    F_va_new = _make_pairs(F_va)
    F_te_new = _make_pairs(F_te) if F_te is not None else None
    print(f"  [Interactions] {F_tr.shape[1]} → {F_tr_new.shape[1]} features")
    return F_tr_new, F_va_new, F_te_new


# ---------------------------------------------------------------------------
# OPT-6: Static × z_multi — FIX-3: dynamic top_z
# ---------------------------------------------------------------------------

def _build_static_zmulti_cross(F, n_zmulti, n_static, top_z=None):
    """
    FIX-3: top_z now dynamic = min(32, n_zmulti//4) instead of hardcoded 16.
    """
    if n_static == 0 or n_zmulti == 0:
        return F
    if top_z is None:
        top_z = min(32, n_zmulti // 4)

    z_part  = F[:, :n_zmulti]
    s_start = F.shape[1] - n_static
    s_part  = F[:, s_start:]

    actual_top_z = min(top_z, n_zmulti)
    z_top        = z_part[:, :actual_top_z]

    s_std   = np.nanstd(s_part, axis=0)
    bin_idx = np.where(s_std < 0.55)[0][:8]
    if len(bin_idx) == 0:
        return F

    cross = np.hstack([
        z_top * s_part[:, si:si+1] for si in bin_idx
    ]).astype(np.float32)
    print(f"  [Static×z cross] {cross.shape[1]} features "
          f"({len(bin_idx)} static × {actual_top_z} z_multi)")
    return np.hstack([F, cross])


# ---------------------------------------------------------------------------
# OPT-2: Feature importance filter — cover-based (unchanged)
# ---------------------------------------------------------------------------

def _drop_low_importance(bst, F_tr, F_va, F_te, drop_frac=0.10):
    scores = bst.get_score(importance_type="cover")
    if not scores:
        return F_tr, F_va, F_te, list(range(F_tr.shape[1]))
    n    = F_tr.shape[1]
    arr  = np.array([scores.get(f"f{i}", 0.0) for i in range(n)])
    thr  = np.percentile(arr, drop_frac * 100)
    keep = np.where(arr >= thr)[0]
    print(f"  [Feature filter] keeping {len(keep)}/{n} features "
          f"(dropped {n - len(keep)} with gain < {thr:.4f})")
    return F_tr[:, keep], F_va[:, keep], F_te[:, keep], keep.tolist()


# ---------------------------------------------------------------------------
# OPT-1: Isotonic calibration (unchanged)
# ---------------------------------------------------------------------------

def _isotonic_calibrate(bst, F_va, y_va):
    raw = bst.predict(xgb.DMatrix(F_va, missing=np.nan))
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(raw, y_va.astype(float))
    print("  [Calibration] Isotonic regression fitted on val set")
    return iso


# ---------------------------------------------------------------------------
# Optuna search — FIX-2 and FIX-5
# ---------------------------------------------------------------------------

def _optuna_search(F_tr, y_tr, F_va, y_va, objective,
                   n_trials=200, booster="gbtree",
                   depth_max=10, n_estimators_max=2000):
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [Optuna] not installed — using default params")
        return {}

    metric   = "auc" if objective == "binary:logistic" else "rmse"
    maximize = metric == "auc"
    use_dart = booster == "dart"

    pbar         = tqdm(total=n_trials, desc=f"  Optuna [{objective[:6]}]",
                        unit="trial", ncols=100)
    best_so_far  = [-np.inf if maximize else np.inf]

    def trial_fn(trial):
        p = {
            "objective":        objective,
            "eval_metric":      metric,
            "tree_method":      "hist",
            "booster":          booster,
            "device":           "cuda",
            "seed":             42,
            "verbosity":        0,
            "max_depth":        trial.suggest_int("max_depth", 3, depth_max),
            "learning_rate":    trial.suggest_float("learning_rate", 5e-4, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            # FIX-5: lower bound 0.4→0.3 for better column diversity with many features
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        }
        if use_dart:
            p["rate_drop"] = trial.suggest_float("rate_drop", 0.05, 0.3)
            p["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)
        if objective == "binary:logistic":
            p["scale_pos_weight"] = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

        n_rounds = trial.suggest_int("n_estimators", 100, n_estimators_max)
        dm_tr    = xgb.DMatrix(F_tr, label=y_tr, missing=np.nan)
        dm_va    = xgb.DMatrix(F_va, label=y_va, missing=np.nan)

        if use_dart:
            bst = xgb.train(p, dm_tr, n_rounds,
                            evals=[(dm_va, "val")], verbose_eval=False)
        else:
            cb  = [xgb.callback.EarlyStopping(20, metric, maximize=maximize, save_best=True)]
            bst = xgb.train(p, dm_tr, n_rounds, evals=[(dm_va, "val")],
                            callbacks=cb, verbose_eval=False)

        pred = bst.predict(dm_va)
        from sklearn.metrics import roc_auc_score, mean_squared_error
        score = roc_auc_score(y_va, pred) if maximize else -np.sqrt(mean_squared_error(y_va, pred))
        improved = score > best_so_far[0] if maximize else score < best_so_far[0]
        if improved:
            best_so_far[0] = score
        pbar.update(1)
        pbar.set_postfix({"best": f"{best_so_far[0]:.4f}", "cur": f"{score:.4f}",
                          "trees": bst.num_boosted_rounds(), "depth": p["max_depth"]})
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(trial_fn, n_trials=n_trials, show_progress_bar=False)
    pbar.close()
    print(f"  [Optuna] best value: {study.best_value:.4f}  trials: {len(study.trials)}")
    return study.best_params


# ---------------------------------------------------------------------------
# Ensemble helper (unchanged)
# ---------------------------------------------------------------------------

def _train_single_mort_booster(full_p, dm_tr, dm_va, n_rounds,
                                early_stopping, use_dart, seed):
    p = {**full_p, "seed": seed}
    callbacks = None
    if not use_dart and early_stopping > 0:
        callbacks = [xgb.callback.EarlyStopping(
            early_stopping, "auc", maximize=True, save_best=True
        )]
    try:
        bst = xgb.train(p, dm_tr, n_rounds,
                        evals=[(dm_tr, "train"), (dm_va, "val")],
                        callbacks=callbacks, verbose_eval=50)
    except AttributeError:
        bst = xgb.train(p, dm_tr, n_rounds,
                        evals=[(dm_tr, "train"), (dm_va, "val")],
                        verbose_eval=50)
    if not hasattr(bst, "best_iteration") or bst.best_iteration is None:
        bi = getattr(bst, "best_ntree_limit", None)
        bst.best_iteration = bi if bi is not None else 0
    return bst


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_stage2(
    F_tr, ym_tr, yl_tr,
    F_va, ym_va, yl_va,
    F_te=None,
    ckpt_dir="checkpoints",
    xgb_device="cuda",
    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
    early_stopping=30,
    use_optuna=True,
    n_optuna_trials=200,
    optuna_depth_max=10,
    optuna_n_max=2000,
    use_dart=False,
    use_calibration=True,
    use_isotonic=True,
    drop_low_importance=True,
    use_interaction_features=True,
    n_interaction_top=8,
    n_ensemble_mort=1,          # FIX-4: default 2→1
    use_static_cross=True,
    n_zmulti_dims=0,
    n_static_dims=0,
):
    """
    Train XGBoost HEAD on frozen encoder features.

    v6 fix order:
      1. Static × z_multi cross (OPT-6)
      2. Mortality: Optuna → full train → drop_low_importance → refit
      3. [FIX-1] Interaction features AFTER drop_low_importance
      4. [FIX-1] Refit mortality again on interactions + filtered features
      5. Isotonic calibration
      6. LOS: Optuna (FIX-2: pseudohubererror) → train → isotonic
    """
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    booster_type = "dart" if use_dart else "gbtree"
    default_p = dict(
        max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
    )

    # ── OPT-6: Static × z_multi cross ────────────────────────────────────
    if use_static_cross and n_zmulti_dims > 0 and n_static_dims > 0:
        F_tr = _build_static_zmulti_cross(F_tr, n_zmulti_dims, n_static_dims)
        F_va = _build_static_zmulti_cross(F_va, n_zmulti_dims, n_static_dims)
        if F_te is not None:
            F_te = _build_static_zmulti_cross(F_te, n_zmulti_dims, n_static_dims)

    # ── MORTALITY MODEL ───────────────────────────────────────────────────
    print("\n[Stage 2] Training mortality XGBoost …")

    if use_optuna:
        print(f"  Running Optuna search ({n_optuna_trials} trials) …")
        best_p = _optuna_search(
            F_tr, ym_tr, F_va, ym_va, "binary:logistic",
            n_optuna_trials, booster_type, optuna_depth_max, optuna_n_max,
        )
        params_m = {**default_p, **best_p}
    else:
        params_m = default_p.copy()

    spw      = float((ym_tr == 0).sum() / max((ym_tr == 1).sum(), 1))
    print(f"  scale_pos_weight={spw:.1f}")
    n_rounds = int(params_m.pop("n_estimators", n_estimators))

    full_p_m = {
        "objective": "binary:logistic", "eval_metric": ["auc", "logloss"],
        "tree_method": "hist", "booster": booster_type,
        "device": xgb_device, "scale_pos_weight": spw, "seed": 42,
        **params_m,
    }
    if use_dart:
        full_p_m.setdefault("rate_drop", 0.1)
        full_p_m.setdefault("skip_drop", 0.5)
        full_p_m.setdefault("sample_type", "uniform")
        full_p_m.setdefault("normalize_type", "tree")

    dm_tr_m = xgb.DMatrix(F_tr, label=ym_tr, missing=np.nan)
    dm_va_m = xgb.DMatrix(F_va, label=ym_va, missing=np.nan)

    ensemble_seeds = list(range(42, 42 + max(n_ensemble_mort, 1)))
    print(f"  Training {len(ensemble_seeds)} mortality booster(s) …")
    bst_ensemble = [
        _train_single_mort_booster(full_p_m, dm_tr_m, dm_va_m, n_rounds,
                                   early_stopping, use_dart, seed)
        for seed in ensemble_seeds
    ]
    bst_m = bst_ensemble[0]

    # ── Drop low-importance features ──────────────────────────────────────
    keep_idx = list(range(F_tr.shape[1]))
    F_tr_filt = F_tr; F_va_filt = F_va; F_te_filt = F_te
    if drop_low_importance and F_te is not None:
        F_tr_filt, F_va_filt, F_te_filt, keep_idx = _drop_low_importance(
            bst_m, F_tr, F_va, F_te
        )

    # ── FIX-1: Interaction features AFTER filter ──────────────────────────
    if use_interaction_features and F_te is not None:
        F_tr_filt, F_va_filt, F_te_filt = _build_interaction_features(
            F_tr_filt, ym_tr, F_va_filt, F_te_filt, n_top=n_interaction_top
        )

    # ── Refit boosters on filtered + interaction features ─────────────────
    dm_tr_m2 = xgb.DMatrix(F_tr_filt, label=ym_tr, missing=np.nan)
    dm_va_m2 = xgb.DMatrix(F_va_filt, label=ym_va, missing=np.nan)
    print(f"  Refitting {len(ensemble_seeds)} booster(s) on {F_tr_filt.shape[1]} features …")
    bst_ensemble = [
        _train_single_mort_booster(full_p_m, dm_tr_m2, dm_va_m2, n_rounds,
                                   early_stopping, use_dart, seed)
        for seed in ensemble_seeds
    ]
    bst_m = bst_ensemble[0]

    # ── OPT-1: Isotonic calibration ───────────────────────────────────────
    calibrator = None
    if use_calibration:
        if len(bst_ensemble) > 1:
            raw_preds = np.mean([
                b.predict(xgb.DMatrix(F_va_filt, missing=np.nan))
                for b in bst_ensemble
            ], axis=0)
            calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
            calibrator.fit(raw_preds, ym_va.astype(float))
            print("  [Calibration] Isotonic fitted on ensemble average")
        else:
            calibrator = _isotonic_calibrate(bst_m, F_va_filt, ym_va)

    # Save mortality model
    xgb_mort            = XGBMortality()
    xgb_mort.booster    = bst_m
    xgb_mort.calibrator = calibrator
    xgb_mort.platt      = None
    xgb_mort.keep_idx   = keep_idx   # NOTE: interaction features added AFTER filter
                                     # F_te_filt already includes interactions
                                     # keep_idx only covers the pre-interaction filter

    _bst_ensemble = bst_ensemble
    if len(_bst_ensemble) > 1:
        def _ensemble_predict(F_input):
            dm  = xgb.DMatrix(F_input, missing=np.nan)
            raw = np.mean([b.predict(dm) for b in _bst_ensemble], axis=0)
            if xgb_mort.calibrator is not None:
                raw = xgb_mort.calibrator.predict(raw)
            return raw.astype(np.float32)
        xgb_mort.predict = _ensemble_predict
        print(f"  [Ensemble] predict() averages {len(_bst_ensemble)} boosters")

    xgb_mort.save(str(Path(ckpt_dir) / "xgb_mortality.ubj"))
    with open(Path(ckpt_dir) / "mort_meta.pkl", "wb") as f:
        pickle.dump({"calibrator": calibrator, "platt": None, "keep_idx": keep_idx}, f)

    for idx, bst_extra in enumerate(bst_ensemble[1:], start=1):
        bst_extra.save_model(str(Path(ckpt_dir) / f"xgb_mortality_ens{idx}.ubj"))

    # ── LOS MODEL ─────────────────────────────────────────────────────────
    print("\n[Stage 2] Training LOS XGBoost …")

    # FIX-2: Optuna now searches pseudohubererror (matches final train objective)
    los_objective = "reg:pseudohubererror"
    if use_optuna:
        print(f"  Running Optuna search ({n_optuna_trials} trials) …")
        best_p_l = _optuna_search(
            F_tr_filt, np.log1p(yl_tr),
            F_va_filt, np.log1p(yl_va),
            los_objective, n_optuna_trials,    # FIX-2: was reg:squarederror
            "gbtree", optuna_depth_max, optuna_n_max,
        )
        params_l = {**default_p, **best_p_l}
    else:
        params_l = default_p.copy()

    n_rounds_l = int(params_l.pop("n_estimators", n_estimators))
    full_p_l   = {
        "objective": los_objective, "eval_metric": "rmse",
        "tree_method": "hist", "booster": "gbtree",
        "device": xgb_device, "seed": 42,
        **params_l,
    }

    dm_tr_l = xgb.DMatrix(F_tr_filt, label=np.log1p(yl_tr), missing=np.nan)
    dm_va_l = xgb.DMatrix(F_va_filt, label=np.log1p(yl_va), missing=np.nan)
    callbacks_l = [xgb.callback.EarlyStopping(
        early_stopping, "rmse", maximize=False, save_best=True
    )] if early_stopping > 0 else None

    try:
        bst_l = xgb.train(full_p_l, dm_tr_l, n_rounds_l,
                           evals=[(dm_tr_l, "train"), (dm_va_l, "val")],
                           callbacks=callbacks_l, verbose_eval=50)
    except AttributeError as e:
        print(f"  [Stage 2] LOS callback error, retrying: {e}")
        bst_l = xgb.train(full_p_l, dm_tr_l, n_rounds_l,
                           evals=[(dm_tr_l, "train"), (dm_va_l, "val")],
                           verbose_eval=50)

    if not hasattr(bst_l, "best_iteration") or bst_l.best_iteration is None:
        bi = getattr(bst_l, "best_ntree_limit", None)
        bst_l.best_iteration = bi if bi is not None else 0

    iso = None
    if use_isotonic:
        raw_va = np.expm1(np.maximum(bst_l.predict(dm_va_l), 0.0))
        iso    = IsotonicRegression(out_of_bounds="clip")
        iso.fit(raw_va, yl_va)
        print("  [Isotonic] LOS correction fitted on val set")

    xgb_los         = XGBLos()
    xgb_los.booster = bst_l
    xgb_los.iso     = iso
    xgb_los.save(str(Path(ckpt_dir) / "xgb_los.ubj"))
    with open(Path(ckpt_dir) / "los_meta.pkl", "wb") as f:
        pickle.dump({"iso": iso}, f)

    print("\n[Stage 2] Both models saved.")
    # Return F_te_filt: fully transformed test matrix (filtered + interactions)
    return xgb_mort, xgb_los, F_te_filt