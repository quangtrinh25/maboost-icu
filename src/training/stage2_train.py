"""
src/training/stage2_train.py
=============================
Advanced Stage 2: frozen Mamba encoder → dual XGBoost with Optuna tuning.

Key fixes vs. original:
  1. NaN PRESERVATION — raw_stats NaN values are no longer forced to 0.0.
     XGBoost's sparsity-aware split finder handles NaN natively; zero-filling
     conflated "feature not measured" with "feature value = 0", wrecking the
     XGBoost-flat baseline (AUROC 0.51 instead of expected 0.70+).

  2. STATIC EMBEDDING PROPAGATION — extract_features() now accepts an optional
     static_proj (the DualHeadMamba.static_emb learned in Stage 1).  When
     provided, x_static is projected through the same non-linear transformation
     that Stage 1 learned, so Stage 2 benefits from the encoder's view of
     static features rather than discarding it with a raw hstack.

  3. DART / OPTUNA CONSISTENCY — Optuna now mirrors the final booster type.
     When use_dart=True, trials also use DART so the hyperparameters found
     (max_depth, learning_rate, etc.) are valid for the same optimiser.
     EarlyStopping is disabled within Optuna trials when DART is active
     (DART and EarlyStopping are incompatible in xgboost).
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

from src.models.mamba_encoder import MambaEncoder
from src.models.xgboost_head  import XGBMortality, XGBLos


# ---------------------------------------------------------------------------
# Encoder utilities
# ---------------------------------------------------------------------------

def load_frozen_encoder(ckpt: str, d_input: int, d_model: int = 128,
                         **kw) -> MambaEncoder:
    enc = MambaEncoder(d_input, d_model, **kw)
    enc.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    n = sum(1 for _ in enc.parameters())
    print(f"[Stage 2] Encoder frozen ({n} tensors).")
    return enc


@torch.no_grad()
def extract_features(
    enc: MambaEncoder,
    loader: DataLoader,
    device: str = "cuda",
    static_proj: Optional[nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2 feature extraction.

    F_final = [z_multi(4*d) ; raw_stats(5*F) ; static(42 | projected)]

    Three complementary feature sets:
      z_multi    — multi-resolution encoder features (last/mean/max/weighted)
      raw_stats  — per-feature temporal stats (last/mean/max/std/missingness)
                   NaN for never-observed features → XGBoost sparsity-aware splits
      x_static   — demographics + comorbidities
                   If static_proj is supplied (DualHeadMamba.static_emb), the
                   LEARNED non-linear projection is applied, preserving the
                   encoder's representation of static context from Stage 1.
                   Otherwise raw x_static is appended unchanged.

    Parameters
    ----------
    enc         : frozen MambaEncoder
    loader      : DataLoader yielding (x, tau, mask[, x_static], y_mort, y_los)
    device      : torch device string
    static_proj : optional nn.Module — pass DualHeadMamba.static_emb to keep
                  the learned static transformation from Stage 1.
                  If None, raw x_static is appended (backward-compatible).
    """
    enc = enc.to(device)
    if static_proj is not None:
        static_proj = static_proj.to(device).eval()

    Fs, morts, los_arr = [], [], []

    for batch in loader:
        x, tau, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        has_static   = len(batch) == 6

        # Multi-resolution encoder features + raw temporal statistics
        z_multi, raw_stats = enc.extract_features(x, tau, mask)
        z_multi   = z_multi.cpu().numpy()
        raw_stats = raw_stats.cpu().numpy()    # may contain NaN — intentional

        parts = [z_multi, raw_stats]

        if has_static:
            x_static_raw = batch[3]
            if static_proj is not None:
                # Apply the same learned projection as Stage 1 so XGBoost
                # sees the encoder's non-linear view of demographics.
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

    # CRITICAL: Do NOT force nan=0.0 here.
    # NaN in raw_stats signals "feature never observed in this stay".
    # XGBoost's sparsity-aware split finder routes NaN values to the
    # optimal child node; coercing to 0 conflates missingness with zero value.
    F   = np.vstack(Fs).astype(np.float32)
    y_m = np.concatenate(morts)
    y_l = np.concatenate(los_arr)

    print(f"  Features {F.shape} | pos_rate={y_m.mean():.3f} | "
          f"mean_LOS={y_l.mean():.1f}d")
    return F, y_m, y_l


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------

def _optuna_search(
    F_tr: np.ndarray, y_tr: np.ndarray,
    F_va: np.ndarray, y_va: np.ndarray,
    objective: str,
    n_trials: int = 50,
    booster: str = "gbtree",
) -> dict:
    """
    Return best XGBoost params via Optuna.

    Parameters
    ----------
    booster : "gbtree" or "dart"
        Must match the booster used in the final train_stage2 call.
        When "dart", EarlyStopping is disabled within trials (incompatible)
        and a fixed n_estimators is used instead.
        Previously Optuna always used gbtree/hist while the final model used
        DART, producing hyperparameters that were inconsistent with the
        actual training algorithm.

    Falls back gracefully to empty dict if Optuna is not installed.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [Optuna] not installed — using default params (pip install optuna)")
        return {}

    metric   = "auc" if objective == "binary:logistic" else "rmse"
    maximize = metric == "auc"
    use_dart = booster == "dart"

    def trial_fn(trial):
        p = {
            "objective":        objective,
            "eval_metric":      metric,
            "tree_method":      "hist",
            "booster":          booster,
            "device":           "cuda",
            "seed":             42,
            "verbosity":        0,
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
        }
        if use_dart:
            p["rate_drop"] = trial.suggest_float("rate_drop", 0.05, 0.3)
            p["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)

        if objective == "binary:logistic":
            spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
            p["scale_pos_weight"] = spw

        n_rounds = trial.suggest_int("n_estimators", 100, 1000)

        dm_tr = xgb.DMatrix(F_tr, label=y_tr, missing=np.nan)
        dm_va = xgb.DMatrix(F_va, label=y_va, missing=np.nan)

        if use_dart:
            # EarlyStopping incompatible with DART — use fixed n_rounds
            bst = xgb.train(p, dm_tr, n_rounds,
                            evals=[(dm_va, "val")], verbose_eval=False)
        else:
            cb  = [xgb.callback.EarlyStopping(20, metric, maximize=maximize,
                                               save_best=True)]
            bst = xgb.train(p, dm_tr, n_rounds, evals=[(dm_va, "val")],
                            callbacks=cb, verbose_eval=False)

        pred = bst.predict(dm_va)
        from sklearn.metrics import roc_auc_score, mean_squared_error
        if objective == "binary:logistic":
            return roc_auc_score(y_va, pred)
        return -np.sqrt(mean_squared_error(y_va, pred))

    study = optuna.create_study(direction="maximize")
    study.optimize(trial_fn, n_trials=n_trials, show_progress_bar=False)
    print(f"  [Optuna] best value: {study.best_value:.4f}  "
          f"trials: {len(study.trials)}")
    return study.best_params


# ---------------------------------------------------------------------------
# Feature importance filter
# ---------------------------------------------------------------------------

def _drop_low_importance(
    bst: xgb.Booster,
    F_tr: np.ndarray,
    F_va: np.ndarray,
    F_te: np.ndarray,
    drop_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Drop bottom `drop_frac` of features by XGBoost gain importance."""
    scores = bst.get_score(importance_type="gain")
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
# Platt scaling for mortality calibration
# ---------------------------------------------------------------------------

def _platt_calibrate(
    bst: xgb.Booster,
    F_va: np.ndarray,
    y_va: np.ndarray,
) -> LogisticRegression:
    """Fit Platt scaling (logistic regression) on validation predictions."""
    raw = bst.predict(xgb.DMatrix(F_va, missing=np.nan)).reshape(-1, 1)
    lr  = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(raw, y_va.astype(int))
    print("  [Calibration] Platt scaling fitted on val set")
    return lr


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_stage2(
    F_tr:  np.ndarray, ym_tr: np.ndarray, yl_tr: np.ndarray,
    F_va:  np.ndarray, ym_va: np.ndarray, yl_va: np.ndarray,
    F_te:  Optional[np.ndarray] = None,
    ckpt_dir: str   = "checkpoints",
    xgb_device: str = "cuda",
    n_estimators: int   = 500,
    max_depth:    int   = 5,
    learning_rate: float = 0.05,
    subsample:     float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    reg_lambda: float = 1.0,
    early_stopping: int = 30,
    use_optuna: bool = True,
    n_optuna_trials: int = 100,
    use_dart: bool = False,
    use_calibration: bool = True,
    use_isotonic: bool = True,
    drop_low_importance: bool = True,
) -> Tuple[XGBMortality, XGBLos]:
    """
    Train two XGBoost models on frozen Mamba features.

    Parameters
    ----------
    use_dart : bool
        When True, Optuna trials also use DART so found hyperparameters
        (depth, lr, etc.) are calibrated to the same booster.  Previously,
        Optuna used gbtree/hist while the final model used DART — producing
        hyperparameters inconsistent with the training algorithm.

    Returns
    -------
    xgb_mort : XGBMortality  (with Platt scaling if use_calibration=True)
    xgb_los  : XGBLos        (with isotonic correction if use_isotonic=True)
    """
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    booster_type = "dart" if use_dart else "gbtree"

    default_p = dict(
        max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
    )

    # ── MORTALITY MODEL ───────────────────────────────────────────────────
    print("\n[Stage 2] Training mortality XGBoost …")

    if use_optuna:
        print(f"  Running Optuna search ({n_optuna_trials} trials) …")
        best_p = _optuna_search(
            F_tr, ym_tr, F_va, ym_va,
            "binary:logistic", n_optuna_trials,
            booster=booster_type,               # FIX: match final booster
        )
        params_m = {**default_p, **best_p}
    else:
        params_m = default_p

    spw = float((ym_tr == 0).sum() / max((ym_tr == 1).sum(), 1))
    print(f"  scale_pos_weight={spw:.1f}")

    # n_estimators may come from Optuna; pop before building full_p_m
    n_rounds = int(params_m.pop("n_estimators", n_estimators))

    full_p_m = {
        "objective":         "binary:logistic",
        "eval_metric":       ["auc", "logloss"],
        "tree_method":       "hist",
        "booster":           booster_type,
        "device":            xgb_device,
        "scale_pos_weight":  spw,
        "seed":              42,
        **params_m,
    }
    if use_dart:
        # Merge DART-specific params from Optuna if present, else use defaults
        full_p_m.setdefault("rate_drop",       0.1)
        full_p_m.setdefault("skip_drop",       0.5)
        full_p_m.setdefault("sample_type",    "uniform")
        full_p_m.setdefault("normalize_type", "tree")

    # EarlyStopping is incompatible with DART; use fixed rounds instead
    callbacks_m = None
    if not use_dart and early_stopping and early_stopping > 0:
        callbacks_m = [xgb.callback.EarlyStopping(
            early_stopping, "auc", maximize=True, save_best=True
        )]

    dm_tr_m = xgb.DMatrix(F_tr, label=ym_tr, missing=np.nan)
    dm_va_m = xgb.DMatrix(F_va, label=ym_va, missing=np.nan)

    try:
        bst_m = xgb.train(full_p_m, dm_tr_m, n_rounds,
                           evals=[(dm_tr_m, "train"), (dm_va_m, "val")],
                           callbacks=callbacks_m, verbose_eval=50)
    except AttributeError as e:
        print(f"  [Stage 2] xgb.train callback error, retrying without callbacks: {e}")
        bst_m = xgb.train(full_p_m, dm_tr_m, n_rounds,
                           evals=[(dm_tr_m, "train"), (dm_va_m, "val")],
                           verbose_eval=50)

    if not hasattr(bst_m, "best_iteration") or bst_m.best_iteration is None:
        bi = getattr(bst_m, "best_ntree_limit", None)
        bst_m.best_iteration = bi if bi is not None else 0

    # Feature importance filter → refit on reduced feature set
    if drop_low_importance and F_te is not None:
        F_tr_m, F_va_m, F_te_m, keep_idx = _drop_low_importance(
            bst_m, F_tr, F_va, F_te
        )
        dm_tr_m = xgb.DMatrix(F_tr_m, label=ym_tr, missing=np.nan)
        dm_va_m = xgb.DMatrix(F_va_m, label=ym_va, missing=np.nan)

        callbacks_m2 = None
        if not use_dart and early_stopping and early_stopping > 0:
            callbacks_m2 = [xgb.callback.EarlyStopping(
                early_stopping, "auc", maximize=True, save_best=True
            )]
        try:
            bst_m = xgb.train(full_p_m, dm_tr_m, n_rounds,
                               evals=[(dm_tr_m, "train"), (dm_va_m, "val")],
                               callbacks=callbacks_m2, verbose_eval=50)
        except AttributeError as e:
            print(f"  [Stage 2] xgb.train (refit) callback error, retrying: {e}")
            bst_m = xgb.train(full_p_m, dm_tr_m, n_rounds,
                               evals=[(dm_tr_m, "train"), (dm_va_m, "val")],
                               verbose_eval=50)

        if not hasattr(bst_m, "best_iteration") or bst_m.best_iteration is None:
            bi = getattr(bst_m, "best_ntree_limit", None)
            bst_m.best_iteration = bi if bi is not None else 0
    else:
        keep_idx = list(range(F_tr.shape[1]))
        F_va_m   = F_va

    platt = None
    if use_calibration:
        val_for_platt = F_va_m if (drop_low_importance and F_te is not None) else F_va
        platt = _platt_calibrate(bst_m, val_for_platt, ym_va)

    xgb_mort          = XGBMortality()
    xgb_mort.booster  = bst_m
    xgb_mort.platt    = platt
    xgb_mort.keep_idx = keep_idx
    xgb_mort.save(str(Path(ckpt_dir) / "xgb_mortality.ubj"))
    with open(Path(ckpt_dir) / "mort_meta.pkl", "wb") as f:
        pickle.dump({"platt": platt, "keep_idx": keep_idx}, f)

    # ── LOS MODEL ─────────────────────────────────────────────────────────
    print("\n[Stage 2] Training LOS XGBoost …")

    if use_optuna:
        print(f"  Running Optuna search ({n_optuna_trials} trials) …")
        best_p_l = _optuna_search(
            F_tr, np.log1p(yl_tr), F_va, np.log1p(yl_va),
            "reg:squarederror", n_optuna_trials,
            booster="gbtree",   # LOS uses gbtree (DART rarely helps for regression)
        )
        params_l = {**default_p, **best_p_l}
    else:
        params_l = default_p

    n_rounds_l = int(params_l.pop("n_estimators", n_estimators))
    full_p_l   = {
        "objective":   "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "booster":     "gbtree",
        "device":      xgb_device,
        "seed":        42,
        **params_l,
    }

    dm_tr_l = xgb.DMatrix(F_tr, label=np.log1p(yl_tr), missing=np.nan)
    dm_va_l = xgb.DMatrix(F_va, label=np.log1p(yl_va), missing=np.nan)

    callbacks_l = None
    if early_stopping and early_stopping > 0:
        callbacks_l = [xgb.callback.EarlyStopping(
            early_stopping, "rmse", maximize=False, save_best=True
        )]

    try:
        bst_l = xgb.train(full_p_l, dm_tr_l, n_rounds_l,
                           evals=[(dm_tr_l, "train"), (dm_va_l, "val")],
                           callbacks=callbacks_l, verbose_eval=50)
    except AttributeError as e:
        print(f"  [Stage 2] xgb.train (LOS) callback error, retrying: {e}")
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

    xgb_los          = XGBLos()
    xgb_los.booster  = bst_l
    xgb_los.iso      = iso
    xgb_los.save(str(Path(ckpt_dir) / "xgb_los.ubj"))
    with open(Path(ckpt_dir) / "los_meta.pkl", "wb") as f:
        pickle.dump({"iso": iso}, f)

    print("\n[Stage 2] Both models saved.")
    return xgb_mort, xgb_los