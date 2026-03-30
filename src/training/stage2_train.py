"""
src/training/stage2_train.py  (v5 — ICU-optimised)
====================================================
Optimizations vs v4 — all function names/signatures unchanged:

OPT-1  Isotonic calibration (replaces Platt/LogisticRegression 1D)
       IsotonicRegression handles non-linear miscalibration in mortality
       tails that matter most in ICU risk stratification.

OPT-2  Feature importance filter: gain → cover
       'gain' importance is numerically unstable across seeds.
       'cover' (avg samples routed through a split) is more stable
       and better reflects generalization importance.

OPT-3  LOS objective: squarederror → pseudohubererror  (in xgboost_head.py)
       MIMIC-IV ICU LOS is heavy-tailed; Pseudo-Huber prevents outlier
       long-stay patients from dominating gradients.

OPT-4  Interaction features: top-K clinical crossproducts prepended to F
       z_multi is L2-normalized embedding; raw_stats are raw clinical values.
       XGBoost tree splits treat all features equally by column sampling,
       so cross-products of high-importance features are engineered explicitly
       rather than relying on deep trees to find them.
       - First pass: shallow XGBoost (depth=3, 200 trees) → top-K features by cover
       - Build K*(K-1)/2 interaction pairs for top-K
       - Prepend to F before final training
       - n_interaction_features=30 default (top 8 features → 28 pairs)

OPT-5  Dual-model mortality ensemble: train mort XGBoost twice with different
       random seeds, average predictions — reduces variance with 47k patients.
       Controlled by n_ensemble_mort (default=2, set 1 to disable).

OPT-6  Static × z_multi cross features: for each binary static feature,
       multiply by the top-d_model/4 z_multi dimensions to give XGBoost
       direct access to the interaction without needing extra depth.

OPT-7  Encoder fine-tune option: if finetune_enc_epochs > 0, unfreeze encoder
       for a few epochs with very low lr before feature extraction.
       Allows z_multi to adapt to Stage 2 label distribution.
       Default 0 (disabled) — enable carefully with large datasets.

All unchanged from v4: NaN handling, DART/gbtree switching, Optuna TPE,
early stopping retry, isotonic LOS correction, checkpoint paths.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import warnings
import gc
import xgboost as xgb
warnings.filterwarnings("ignore", category=FutureWarning, module="xgboost")
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def _finetune_encoder(
    enc: MambaEncoder,
    loader: DataLoader,
    device: str,
    epochs: int,
    lr: float = 1e-5,
) -> MambaEncoder:
    """
    OPT-7: Lightly fine-tune encoder on Stage 2 labels.
    Uses only mortality supervision (binary cross-entropy) with very low lr.
    Unfreeze all parameters, train for `epochs` epochs, re-freeze.
    """
    from torch.nn import functional as F_nn

    enc = enc.to(device)
    for p in enc.parameters():
        p.requires_grad = True

    # Lightweight classification head (not saved, only used for fine-tuning)
    d = enc.pool.proj.out_features
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
            z = enc(x, tau, mask)
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
def extract_features(
    enc: MambaEncoder,
    loader: DataLoader,
    device: str = "cuda",
    static_proj: Optional[nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stage 2 feature extraction.
    F_final = [z_multi(4*d) ; raw_stats(5*F) ; static(42 | projected)]
    NaN in raw_stats intentionally preserved for XGBoost sparsity routing.
    """
    enc = enc.to(device)
    if static_proj is not None:
        static_proj = static_proj.to(device).eval()

    Fs, morts, los_arr = [], [], []

    for batch in loader:
        x, tau, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        has_static   = len(batch) >= 6 and hasattr(batch[3], "dim") and batch[3].dim() > 1

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

    # CRITICAL: Do NOT replace NaN — XGBoost routes NaN via sparsity splits
    F_out = np.vstack(Fs).astype(np.float32)
    y_m   = np.concatenate(morts)
    y_l   = np.concatenate(los_arr)

    print(f"  Features {F_out.shape} | pos_rate={y_m.mean():.3f} | "
          f"mean_LOS={y_l.mean():.1f}d")
    return F_out, y_m, y_l


# ---------------------------------------------------------------------------
# OPT-4: Interaction features
# ---------------------------------------------------------------------------

def _build_interaction_features(
    F_tr: np.ndarray,
    y_tr: np.ndarray,
    F_va: np.ndarray,
    F_te: Optional[np.ndarray],
    n_top: int = 8,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[Tuple[int, int]]]:
    """
    OPT-4: Build pairwise interaction features from top-N important features.

    Steps:
      1. Fit shallow XGBoost (depth=3, 200 rounds) to get feature importance.
      2. Select top-n_top features by 'cover' importance (stable across seeds).
      3. Build all n_top*(n_top-1)/2 pairwise products.
      4. Prepend to F — XGBoost sees engineered interactions directly.

    With n_top=8: 28 new features.
    With n_top=10: 45 new features.
    NaN handling: product of (NaN, x) = NaN, preserved for sparsity routing.
    """
    print(f"  [Interactions] fitting shallow XGB for importance (top-{n_top})...")
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    p_shallow = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "tree_method": "hist", "max_depth": 3,
        "learning_rate": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "scale_pos_weight": spw,
        "seed": 42, "verbosity": 0,
    }
    dm_tr = xgb.DMatrix(F_tr, label=y_tr, missing=np.nan)
    bst_shallow = xgb.train(p_shallow, dm_tr, 200, verbose_eval=False)

    n_cols = F_tr.shape[1]
    scores = bst_shallow.get_score(importance_type="cover")
    arr    = np.array([scores.get(f"f{i}", 0.0) for i in range(n_cols)])
    top_idx = np.argsort(arr)[::-1][:n_top]

    pairs = [(i, j) for idx_i, i in enumerate(top_idx)
             for j in top_idx[idx_i+1:]]
    print(f"  [Interactions] {len(pairs)} pairs from top-{n_top} features")

    def _make_pairs(F):
        cols = []
        for i, j in pairs:
            # NaN * x = NaN, preserved automatically via numpy
            cols.append(F[:, i] * F[:, j])
        if not cols:
            return F
        inter = np.stack(cols, axis=1).astype(np.float32)
        return np.hstack([inter, F])

    F_tr_new = _make_pairs(F_tr)
    F_va_new = _make_pairs(F_va)
    F_te_new = _make_pairs(F_te) if F_te is not None else None

    print(f"  [Interactions] feature dim: {F_tr.shape[1]} → {F_tr_new.shape[1]}")
    return F_tr_new, F_va_new, F_te_new, pairs


# ---------------------------------------------------------------------------
# OPT-6: Static × z_multi cross features
# ---------------------------------------------------------------------------

def _build_static_zmulti_cross(
    F: np.ndarray,
    n_zmulti: int,
    n_static: int,
    top_z: int = 16,
) -> np.ndarray:
    """
    OPT-6: For each binary/categorical static feature, multiply by
    top-`top_z` z_multi dimensions.

    Layout of F: [z_multi(n_zmulti) | raw_stats(5*d_feat) | static(n_static)]
    We only cross binary static features (values near 0/1) with z_multi summary.

    This gives XGBoost direct access to group-specific temporal patterns
    (e.g., gender × z_multi, elective_admission × z_multi) without
    requiring extra tree depth.
    """
    if n_static == 0 or n_zmulti == 0:
        return F

    z_part  = F[:, :n_zmulti]                              # (N, n_zmulti)
    s_start = F.shape[1] - n_static
    s_part  = F[:, s_start:]                               # (N, n_static)

    # Use first top_z z_multi dims (early dims tend to be most informative
    # from MultiResolutionPooling: last, mean, max, attn each d/4 wide)
    actual_top_z = min(top_z, n_zmulti)
    z_top = z_part[:, :actual_top_z]                       # (N, top_z)

    # Only cross binary-ish static features (std < 0.6 of a [0,1] var)
    s_std   = np.nanstd(s_part, axis=0)
    bin_idx = np.where(s_std < 0.55)[0][:8]               # at most 8 binary statics

    if len(bin_idx) == 0:
        return F

    cross_cols = []
    for si in bin_idx:
        col = s_part[:, si:si+1]                           # (N, 1)
        cross_cols.append(z_top * col)                     # (N, top_z)

    cross = np.hstack(cross_cols).astype(np.float32)      # (N, len*top_z)
    print(f"  [Static×z cross] {cross.shape[1]} features "
          f"({len(bin_idx)} static × {actual_top_z} z_multi)")
    return np.hstack([F, cross])


def _static_cross_bin_idx_from_input(F: np.ndarray, n_static: int) -> List[int]:
    if n_static <= 0:
        return []
    s_part = F[:, F.shape[1] - n_static:]
    s_std = np.nanstd(s_part, axis=0)
    return np.where(s_std < 0.55)[0][:8].tolist()


def _apply_static_zmulti_cross_with_meta(
    F: np.ndarray,
    n_zmulti: int,
    n_static: int,
    top_z: int,
    bin_idx: List[int],
) -> np.ndarray:
    if n_static == 0 or n_zmulti == 0 or len(bin_idx) == 0:
        return F
    z_part = F[:, :n_zmulti]
    s_start = F.shape[1] - n_static
    s_part = F[:, s_start:]
    actual_top_z = min(top_z, n_zmulti)
    z_top = z_part[:, :actual_top_z]
    cross_cols = []
    for si in bin_idx:
        if si < 0 or si >= s_part.shape[1]:
            continue
        cross_cols.append(z_top * s_part[:, si:si + 1])
    if not cross_cols:
        return F
    cross = np.hstack(cross_cols).astype(np.float32)
    return np.hstack([F, cross])


def _apply_interaction_pairs(F: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return F
    cols = []
    for i, j in pairs:
        if i < 0 or j < 0 or i >= F.shape[1] or j >= F.shape[1]:
            continue
        cols.append(F[:, i] * F[:, j])
    if not cols:
        return F
    inter = np.stack(cols, axis=1).astype(np.float32)
    return np.hstack([inter, F])


# ---------------------------------------------------------------------------
# Optuna hyperparameter search (v4 unchanged)
# ---------------------------------------------------------------------------

def _optuna_search(
    F_tr: np.ndarray, y_tr: np.ndarray,
    F_va: np.ndarray, y_va: np.ndarray,
    objective: str,
    n_trials: int = 200,
    booster: str = "gbtree",
    depth_max: int = 10,
    n_estimators_max: int = 2000,
    score_metric: str = "auroc",
) -> dict:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("  [Optuna] not installed — using default params")
        return {}

    score_metric = str(score_metric).lower().strip()
    if objective == "binary:logistic":
        metric = "aucpr" if score_metric == "auprc" else "auc"
        maximize = True
    else:
        metric = "rmse"
        maximize = False
    use_dart = booster == "dart"
    depth_hi = min(int(depth_max), 8)   # avoid extreme GPU memory spikes
    n_hi     = min(int(n_estimators_max), 1200)
    if depth_hi < 3:
        depth_hi = 3
    if n_hi < 100:
        n_hi = 100

    pbar = tqdm(total=n_trials, desc=f"  Optuna [{objective[:6]}]",
                unit="trial", ncols=100)
    best_so_far = [-np.inf if maximize else np.inf]

    def trial_fn(trial):
        p = {
            "objective":        objective,
            "eval_metric":      metric,
            "tree_method":      "hist",
            "booster":          booster,
            "device":           "cuda",
            "seed":             42,
            "verbosity":        0,
            "max_depth":        trial.suggest_int("max_depth", 3, depth_hi),
            "learning_rate":    trial.suggest_float("learning_rate", 5e-4, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 5.0),
            "max_bin":          256,
        }
        if use_dart:
            p["rate_drop"] = trial.suggest_float("rate_drop", 0.05, 0.3)
            p["skip_drop"] = trial.suggest_float("skip_drop", 0.3, 0.7)

        if objective == "binary:logistic":
            spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
            p["scale_pos_weight"] = spw

        n_rounds = trial.suggest_int("n_estimators", 100, n_hi)

        dm_tr = xgb.DMatrix(F_tr, label=y_tr, missing=np.nan)
        dm_va = xgb.DMatrix(F_va, label=y_va, missing=np.nan)

        try:
            if use_dart:
                bst = xgb.train(p, dm_tr, n_rounds,
                                evals=[(dm_va, "val")], verbose_eval=False)
            else:
                cb  = [xgb.callback.EarlyStopping(20, metric, maximize=maximize,
                                                   save_best=True)]
                bst = xgb.train(p, dm_tr, n_rounds, evals=[(dm_va, "val")],
                                callbacks=cb, verbose_eval=False)
        except xgb.core.XGBoostError as e:
            msg = str(e).lower()
            # Do not abort whole Optuna run for a single oversized GPU trial.
            if "out of memory" in msg or "cudaerrormemoryallocation" in msg:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned("Pruned trial due to GPU OOM")
            raise

        pred = bst.predict(dm_va)
        from sklearn.metrics import average_precision_score, mean_squared_error, roc_auc_score
        if objective == "binary:logistic":
            score = (
                average_precision_score(y_va, pred)
                if score_metric == "auprc"
                else roc_auc_score(y_va, pred)
            )
        else:
            score = -np.sqrt(mean_squared_error(y_va, pred))

        improved = score > best_so_far[0] if maximize else score < best_so_far[0]
        if improved:
            best_so_far[0] = score
        pbar.update(1)
        pbar.set_postfix({
            "best": f"{best_so_far[0]:.4f}",
            "cur":  f"{score:.4f}",
            "trees": bst.num_boosted_rounds(),
            "depth": p["max_depth"],
            "lr":    f"{p['learning_rate']:.4f}",
        })
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(
        trial_fn,
        n_trials=n_trials,
        show_progress_bar=False,
        catch=(xgb.core.XGBoostError, RuntimeError, MemoryError),
    )
    pbar.close()

    print(f"  [Optuna] best value: {study.best_value:.4f}  "
          f"trials: {len(study.trials)}")
    return study.best_params


# ---------------------------------------------------------------------------
# OPT-2: Feature importance filter — cover instead of gain
# ---------------------------------------------------------------------------

def _drop_low_importance(
    bst: xgb.Booster,
    F_tr: np.ndarray,
    F_va: np.ndarray,
    F_te: np.ndarray,
    drop_frac: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    OPT-2: Use 'cover' importance instead of 'gain'.
    'cover' = average number of samples routed through a split — stable
    across seeds and more reflective of generalization value than 'gain'.
    """
    # OPT-2: importance_type="cover" instead of "gain"
    scores = bst.get_score(importance_type="cover")
    if not scores:
        return F_tr, F_va, F_te, list(range(F_tr.shape[1]))
    n    = F_tr.shape[1]
    arr  = np.array([scores.get(f"f{i}", 0.0) for i in range(n)])
    thr  = np.percentile(arr, drop_frac * 100)
    keep = np.where(arr >= thr)[0]
    print(f"  [Feature filter] keeping {len(keep)}/{n} features "
          f"(cover-based, dropped {n - len(keep)} with cover < {thr:.4f})")
    return F_tr[:, keep], F_va[:, keep], F_te[:, keep], keep.tolist()


# ---------------------------------------------------------------------------
# OPT-1: Isotonic calibration (replaces Platt)
# ---------------------------------------------------------------------------

def _isotonic_calibrate(
    bst: xgb.Booster,
    F_va: np.ndarray,
    y_va: np.ndarray,
) -> IsotonicRegression:
    """
    OPT-1: Isotonic regression calibration.

    Platt scaling (LogisticRegression 1D) is parametric and can only correct
    linear miscalibration. IsotonicRegression is monotone and non-parametric —
    it corrects arbitrary shape miscalibration without assumptions.

    For ICU mortality, calibration of very high risk (>0.8) and very low risk
    (<0.1) patients matters clinically. Platt tends to under-correct these tails.
    IsotonicRegression fits them directly from validation data.
    """
    raw = bst.predict(xgb.DMatrix(F_va, missing=np.nan))
    iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
    iso.fit(raw, y_va.astype(float))
    print("  [Calibration] Isotonic regression fitted on val set")
    return iso


# ---------------------------------------------------------------------------
# OPT-5: Ensemble helper
# ---------------------------------------------------------------------------

def _train_single_mort_booster(
    full_p: dict,
    dm_tr: xgb.DMatrix,
    dm_va: xgb.DMatrix,
    n_rounds: int,
    early_stopping: int,
    use_dart: bool,
    seed: int,
    stop_metric: str = "auc",
) -> xgb.Booster:
    """Train one mortality booster with a given seed."""
    p = {**full_p, "seed": seed}
    callbacks = None
    if not use_dart and early_stopping > 0:
        callbacks = [xgb.callback.EarlyStopping(
            early_stopping, stop_metric, maximize=True, save_best=True
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
    F_tr:  np.ndarray, ym_tr: np.ndarray, yl_tr: np.ndarray,
    F_va:  np.ndarray, ym_va: np.ndarray, yl_va: np.ndarray,
    F_te:  Optional[np.ndarray] = None,
    ckpt_dir: str   = "checkpoints",
    xgb_device: str = "cuda",
    n_estimators: int   = 800,
    max_depth:    int   = 6,
    learning_rate: float = 0.05,
    subsample:     float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 5,
    reg_lambda: float = 1.0,
    early_stopping: int = 30,
    use_optuna: bool = True,
    n_optuna_trials: int = 200,
    optuna_depth_max: int = 10,
    optuna_n_max: int = 2000,
    use_dart: bool = False,
    mortality_opt_metric: str = "auroc",
    use_calibration: bool = True,
    use_isotonic: bool = True,
    drop_low_importance: bool = True,
    # OPT-4: interaction features
    use_interaction_features: bool = True,
    n_interaction_top: int = 8,
    # OPT-5: mortality ensemble
    n_ensemble_mort: int = 2,
    # OPT-6: static × z_multi cross features
    use_static_cross: bool = True,
    n_zmulti_dims: int = 0,      # set to 4*d_model at call site
    n_static_dims: int = 0,      # set to static feature count at call site
) -> Tuple[XGBMortality, XGBLos, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Train two XGBoost models on frozen Mamba encoder features.

    Returns (xgb_mort, xgb_los, F_te_mort, F_te_los).
    - F_te_mort: transformed + keep_idx filtered (for mortality booster)
    - F_te_los : transformed (no keep_idx filter; for LOS booster)

    v5 changes:
      OPT-1  Isotonic calibration (replaces Platt for mortality)
      OPT-2  Feature filter: cover instead of gain
      OPT-3  LOS: pseudohubererror (in xgboost_head.py)
      OPT-4  Interaction features: top-K pairwise products prepended
      OPT-5  Mortality ensemble: average over n_ensemble_mort seeds
      OPT-6  Static × z_multi cross features
    """
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    booster_type = "dart" if use_dart else "gbtree"

    default_p = dict(
        max_depth=max_depth, learning_rate=learning_rate,
        subsample=subsample, colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight, reg_lambda=reg_lambda,
    )

    # ── OPT-6: Static × z_multi cross features ───────────────────────────
    static_cross_meta: Dict = {"enabled": False, "top_z": 16, "bin_idx": []}
    if use_static_cross and n_zmulti_dims > 0 and n_static_dims > 0:
        bin_idx = _static_cross_bin_idx_from_input(F_tr, n_static_dims)
        F_tr = _build_static_zmulti_cross(F_tr, n_zmulti_dims, n_static_dims)
        F_va = _build_static_zmulti_cross(F_va, n_zmulti_dims, n_static_dims)
        if F_te is not None:
            F_te = _build_static_zmulti_cross(F_te, n_zmulti_dims, n_static_dims)
        static_cross_meta = {"enabled": True, "top_z": 16, "bin_idx": bin_idx}

    # ── OPT-4: Interaction features ───────────────────────────────────────
    interaction_pairs: List[Tuple[int, int]] = []
    if use_interaction_features:
        F_tr, F_va, F_te, interaction_pairs = _build_interaction_features(
            F_tr, ym_tr, F_va, F_te, n_top=n_interaction_top
        )

    # ── MORTALITY MODEL ───────────────────────────────────────────────────
    print("\n[Stage 2] Training mortality XGBoost …")

    mortality_opt_metric = str(mortality_opt_metric).lower().strip()
    if mortality_opt_metric not in {"auroc", "auprc"}:
        mortality_opt_metric = "auroc"
    xgb_mort_eval_metric = "aucpr" if mortality_opt_metric == "auprc" else "auc"

    if use_optuna:
        print(f"  Running Optuna search ({n_optuna_trials} trials) …")
        best_p = _optuna_search(
            F_tr, ym_tr, F_va, ym_va,
            "binary:logistic", n_optuna_trials,
            booster=booster_type,
            depth_max=optuna_depth_max,
            n_estimators_max=optuna_n_max,
            score_metric=mortality_opt_metric,
        )
        params_m = {**default_p, **best_p}
    else:
        params_m = default_p

    spw = float((ym_tr == 0).sum() / max((ym_tr == 1).sum(), 1))
    print(f"  scale_pos_weight={spw:.1f}")
    n_rounds = int(params_m.pop("n_estimators", n_estimators))

    full_p_m = {
        "objective":         "binary:logistic",
        "eval_metric":       [xgb_mort_eval_metric, "logloss"],
        "tree_method":       "hist",
        "booster":           booster_type,
        "device":            xgb_device,
        "scale_pos_weight":  spw,
        "seed":              42,
        **params_m,
    }
    if use_dart:
        full_p_m.setdefault("rate_drop",       0.1)
        full_p_m.setdefault("skip_drop",       0.5)
        full_p_m.setdefault("sample_type",    "uniform")
        full_p_m.setdefault("normalize_type", "tree")

    dm_tr_m = xgb.DMatrix(F_tr, label=ym_tr, missing=np.nan)
    dm_va_m = xgb.DMatrix(F_va, label=ym_va, missing=np.nan)

    # OPT-5: Train n_ensemble_mort boosters with different seeds
    ensemble_seeds = list(range(42, 42 + max(n_ensemble_mort, 1)))
    print(f"  [Ensemble] training {len(ensemble_seeds)} mortality booster(s) …")
    bst_ensemble = []
    for seed in ensemble_seeds:
        print(f"  [Ensemble] seed={seed}")
        bst = _train_single_mort_booster(
            full_p_m, dm_tr_m, dm_va_m, n_rounds,
            early_stopping, use_dart, seed, stop_metric=xgb_mort_eval_metric
        )
        bst_ensemble.append(bst)

    # Primary booster is seed=42
    bst_m = bst_ensemble[0]

    if drop_low_importance and F_te is not None:
        F_tr_m, F_va_m, F_te_m, keep_idx = _drop_low_importance(
            bst_m, F_tr, F_va, F_te
        )
        dm_tr_m2 = xgb.DMatrix(F_tr_m, label=ym_tr, missing=np.nan)
        dm_va_m2 = xgb.DMatrix(F_va_m, label=ym_va, missing=np.nan)

        print(f"  [Ensemble] refit {len(ensemble_seeds)} booster(s) on filtered features …")
        bst_ensemble = []
        for seed in ensemble_seeds:
            bst = _train_single_mort_booster(
                full_p_m, dm_tr_m2, dm_va_m2, n_rounds,
                early_stopping, use_dart, seed, stop_metric=xgb_mort_eval_metric
            )
            bst_ensemble.append(bst)
        bst_m = bst_ensemble[0]
        # Mortality sees filtered features; LOS keeps transformed-but-unfiltered.
        F_te_mort = F_te_m
        F_te_los = F_te
    else:
        keep_idx   = list(range(F_tr.shape[1]))
        F_va_m     = F_va
        F_te_mort = F_te
        F_te_los = F_te

    # OPT-1: Isotonic calibration
    calibrator = None
    if use_calibration:
        val_for_calib = F_va_m if (drop_low_importance and F_te is not None) else F_va
        if len(ensemble_seeds) > 1:
            # Average raw predictions from ensemble for calibration
            raw_preds = np.mean([
                b.predict(xgb.DMatrix(val_for_calib, missing=np.nan))
                for b in bst_ensemble
            ], axis=0)
            # Fit isotonic on ensemble average
            from sklearn.isotonic import IsotonicRegression as IR
            calibrator = IR(out_of_bounds="clip", increasing=True)
            calibrator.fit(raw_preds, ym_va.astype(float))
            print("  [Calibration] Isotonic fitted on ensemble average")
        else:
            calibrator = _isotonic_calibrate(bst_m, val_for_calib, ym_va)

    # Store ensemble in XGBMortality (primary booster + ensemble list)
    xgb_mort             = XGBMortality()
    xgb_mort.booster     = bst_m
    xgb_mort.calibrator  = calibrator
    xgb_mort.platt       = None
    xgb_mort.keep_idx    = keep_idx
    # Save ensemble predictions capability via monkey-patching predict
    _bst_ensemble = bst_ensemble  # capture for closure
    _keep_idx     = keep_idx

    if len(_bst_ensemble) > 1:
        def _ensemble_predict(F_input: np.ndarray) -> np.ndarray:
            # F_input MUST already be the transformed+filtered feature matrix.
            # keep_idx was already applied by train_stage2 before returning F_te_transformed.
            # Applying it again here would re-index a 1266-col matrix with indices up to 1266 → OOB.
            dm  = xgb.DMatrix(F_input, missing=np.nan)
            raw = np.mean([b.predict(dm) for b in _bst_ensemble], axis=0)
            if xgb_mort.calibrator is not None:
                raw = xgb_mort.calibrator.predict(raw)
            elif xgb_mort.platt is not None:
                raw = xgb_mort.platt.predict_proba(raw.reshape(-1, 1))[:, 1]
            return raw.astype(np.float32)
        xgb_mort.predict = _ensemble_predict
        print(f"  [Ensemble] predict() will average {len(_bst_ensemble)} boosters")

    xgb_mort.save(str(Path(ckpt_dir) / "xgb_mortality.ubj"))
    with open(Path(ckpt_dir) / "mort_meta.pkl", "wb") as f:
        pickle.dump({
            "calibrator": calibrator,
            "platt":      None,
            "keep_idx":   keep_idx,
        }, f)

    # Save additional ensemble boosters
    for idx, bst_extra in enumerate(bst_ensemble[1:], start=1):
        bst_extra.save_model(str(Path(ckpt_dir) / f"xgb_mortality_ens{idx}.ubj"))

    # ── LOS MODEL ─────────────────────────────────────────────────────────
    print("\n[Stage 2] Training LOS XGBoost …")

    if use_optuna:
        print(f"  Running Optuna search ({n_optuna_trials} trials) …")
        best_p_l = _optuna_search(
            F_tr, np.log1p(yl_tr), F_va, np.log1p(yl_va),
            "reg:squarederror", n_optuna_trials,
            booster="gbtree",
            depth_max=optuna_depth_max,
            n_estimators_max=optuna_n_max,
        )
        params_l = {**default_p, **best_p_l}
    else:
        params_l = default_p

    n_rounds_l = int(params_l.pop("n_estimators", n_estimators))
    full_p_l   = {
        # OPT-3: pseudohubererror is set in XGBLos.params (xgboost_head.py)
        # but also override here for the direct xgb.train call:
        "objective":   "reg:pseudohubererror",
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

    with open(Path(ckpt_dir) / "stage2_meta.pkl", "wb") as f:
        pickle.dump({
            "use_static_cross": bool(use_static_cross and n_zmulti_dims > 0 and n_static_dims > 0),
            "n_zmulti_dims": int(n_zmulti_dims),
            "n_static_dims": int(n_static_dims),
            "static_cross_top_z": int(static_cross_meta["top_z"]),
            "static_cross_bin_idx": list(static_cross_meta["bin_idx"]),
            "use_interaction_features": bool(use_interaction_features),
            "interaction_pairs": list(interaction_pairs),
            "keep_idx": list(keep_idx),
        }, f)

    print("\n[Stage 2] Both models saved.")
    return xgb_mort, xgb_los, F_te_mort, F_te_los


def apply_stage2_transforms(
    F: np.ndarray,
    ckpt_dir: str,
    for_mortality: bool = True,
) -> np.ndarray:
    """
    Apply the same Stage 2 feature transforms used in training.
    Used by online inference/update to keep feature-space parity.
    """
    meta_path = Path(ckpt_dir) / "stage2_meta.pkl"
    if not meta_path.exists():
        return F
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    out = F.astype(np.float32, copy=False)

    if meta.get("use_static_cross", False):
        out = _apply_static_zmulti_cross_with_meta(
            out,
            int(meta.get("n_zmulti_dims", 0)),
            int(meta.get("n_static_dims", 0)),
            int(meta.get("static_cross_top_z", 16)),
            list(meta.get("static_cross_bin_idx", [])),
        )

    if meta.get("use_interaction_features", False):
        out = _apply_interaction_pairs(out, list(meta.get("interaction_pairs", [])))

    if for_mortality:
        keep_idx = list(meta.get("keep_idx", []))
        if keep_idx:
            out = out[:, keep_idx]
    return out
