"""
src/visualization/shap_explain.py
===================================
SHAP analysis for MaBoost — mortality XGBoost + LOS XGBoost.

Fixes vs original
-----------------
1. Feature name length mismatch — auto-trim/extend to match booster
   (happens when drop_low_importance filters features after training)
2. 3D SHAP values (multi-class) handled correctly
3. Beeswarm/waterfall shape validation before plotting
4. Clear, human-readable feature names:
     z_last_0..127   → encoder hidden state (last position)
     z_mean_0..127   → encoder hidden state (mean pooling)
     z_max_0..127    → encoder hidden state (max pooling)
     z_attn_0..127   → encoder hidden state (attention pooling)
     last_heart_rate → last observed heart rate value
     mean_lactate    → mean lactate over stay
     miss_creatinine → fraction of timestamps creatinine not measured
     age, gender_male, chf, ... → static/demographic features

Plots saved
-----------
shap/mortality_beeswarm.png      — top features by SHAP value distribution
shap/mortality_bar.png           — mean |SHAP| importance bar chart
shap/mortality_waterfall_top{N}  — per-patient breakdown for top-risk cases
shap/mortality_clinical.png      — SHAP for clinical features only (no latent z)
shap/los_beeswarm.png
shap/los_bar.png
shap/los_clinical.png
shap/latent_umap.png             — z_T coloured by risk + true label
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_shap():
    try:
        import shap
        return shap
    except ImportError:
        raise ImportError("pip install shap")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SHAP] {path.name}")


def _fix_feat_names(names: Optional[List[str]], n_features: int) -> List[str]:
    """
    Ensure feature names list matches n_features exactly.
    Handles keep_idx filtering that reduces feature count after training.
    """
    if names is None:
        return [f"f{i}" for i in range(n_features)]
    names = list(names)
    if len(names) > n_features:
        names = names[:n_features]
    elif len(names) < n_features:
        names = names + [f"f{i}" for i in range(len(names), n_features)]
    return names


def _compute_sv(booster, F: np.ndarray,
                names: Optional[List[str]],
                positive_class: bool = True):
    """
    Compute SHAP values, returning a clean 2D Explanation object.
    Handles:
    - Multi-class output (3D values) → extract positive class
    - Feature name length mismatch (keep_idx filtering)
    - NaN in F → replace with 0 for SHAP (XGBoost handles internally)
    """
    shap = _require_shap()

    # Replace NaN with 0 for SHAP — XGBoost already learned NaN routing
    # but SHAP explainer can struggle with NaN in input data
    F_safe = np.nan_to_num(F, nan=0.0)

    ex   = shap.TreeExplainer(booster)
    sv   = ex(F_safe)
    vals = sv.values
    base = sv.base_values
    data = sv.data if sv.data is not None else F_safe

    # Handle 3D output (multi-class binary: shape = (n, f, 2))
    if vals.ndim == 3:
        cls_idx = 1 if (positive_class and vals.shape[2] > 1) else 0
        vals = vals[:, :, cls_idx]
        if isinstance(base, np.ndarray):
            if base.ndim == 2:
                base = base[:, cls_idx]
            elif base.ndim == 1 and base.shape[0] == sv.values.shape[2]:
                base = base[cls_idx]

    # Handle 1D edge case
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    n_features = vals.shape[1]
    names_fixed = _fix_feat_names(names, n_features)

    # Also fix data shape
    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = data.reshape(-1, 1)
    if isinstance(data, np.ndarray) and data.shape[1] != n_features:
        data = F_safe[:, :n_features] if F_safe.shape[1] >= n_features else F_safe

    return shap.Explanation(
        values       = vals,
        base_values  = base,
        data         = data,
        feature_names= names_fixed,
    )


def _beeswarm(sv, title: str, path: Path, top_n: int = 25):
    shap = _require_shap()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    plt.sca(ax)
    shap.plots.beeswarm(sv, max_display=top_n, show=False)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    plt.tight_layout()
    _save(fig, path)


def _bar(sv, title: str, path: Path, top_n: int = 25):
    ma    = np.abs(sv.values).mean(0)
    idx   = np.argsort(ma)[-top_n:]
    names = sv.feature_names or [f"f{i}" for i in range(len(ma))]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    colors  = ["#2563EB" if not n.startswith("z_") else "#9CA3AF"
               for n in [names[i] for i in idx]]
    ax.barh([names[i] for i in idx], ma[idx], color=colors, alpha=0.85)
    ax.set(xlabel="Mean |SHAP value|", title=title)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    # Legend: blue = clinical, gray = encoder latent
    legend = [
        mpatches.Patch(color="#2563EB", label="Clinical feature"),
        mpatches.Patch(color="#9CA3AF", label="Encoder latent (z)"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="lower right")
    plt.tight_layout()
    _save(fig, path)


def _waterfall(sv, pidx: int, title: str, path: Path):
    shap = _require_shap()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.waterfall(sv[pidx], show=False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, path)


def _clinical_bar(sv, title: str, path: Path,
                  top_n: int = 30):
    """
    Bar chart showing only CLINICAL features (not encoder latent z_*).
    More interpretable for clinical audience.
    """
    names  = sv.feature_names or []
    ma     = np.abs(sv.values).mean(0)

    # Keep only non-latent features
    clin_idx = [i for i, n in enumerate(names) if not n.startswith("z_")]
    if not clin_idx:
        return  # nothing to plot

    clin_ma    = ma[clin_idx]
    clin_names = [names[i] for i in clin_idx]

    # Top N clinical features
    top_idx = np.argsort(clin_ma)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))

    # Color by feature group
    def _color(name):
        if name.startswith("last_"):   return "#059669"
        if name.startswith("mean_"):   return "#2563EB"
        if name.startswith("max_"):    return "#DC2626"
        if name.startswith("std_"):    return "#D97706"
        if name.startswith("miss_"):   return "#7C3AED"
        return "#374151"  # static/demographic

    colors = [_color(clin_names[i]) for i in top_idx]
    ax.barh([clin_names[i] for i in top_idx], clin_ma[top_idx],
            color=colors, alpha=0.85)
    ax.set(xlabel="Mean |SHAP value|")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    legend = [
        mpatches.Patch(color="#059669", label="Last observed value"),
        mpatches.Patch(color="#2563EB", label="Mean over stay"),
        mpatches.Patch(color="#DC2626", label="Max over stay"),
        mpatches.Patch(color="#D97706", label="Std (variability)"),
        mpatches.Patch(color="#7C3AED", label="Miss rate"),
        mpatches.Patch(color="#374151", label="Static / demographic"),
    ]
    ax.legend(handles=legend, fontsize=8, loc="lower right")
    plt.tight_layout()
    _save(fig, path)


def _plot_umap(z_T: np.ndarray, y_prob: np.ndarray,
               y_true: np.ndarray, path: Path):
    n   = min(3000, len(z_T))
    idx = np.random.default_rng(42).choice(len(z_T), n, replace=False)
    Z, yp, yt = z_T[idx], y_prob[idx], y_true[idx]

    try:
        from umap import UMAP
        Z2  = UMAP(n_components=2, random_state=42, verbose=False).fit_transform(Z)
        lbl = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        Z2  = PCA(n_components=2, random_state=42).fit_transform(Z)
        lbl = "PCA (install umap-learn for UMAP)"

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))

    sc = a1.scatter(Z2[:,0], Z2[:,1], c=yp, cmap="RdBu_r",
                    s=8, alpha=0.55, vmin=0, vmax=1)
    plt.colorbar(sc, ax=a1, label="Predicted mortality risk")
    a1.set(title=f"{lbl} — coloured by predicted risk")
    a1.set_xlabel(f"{lbl} dim 1"); a1.set_ylabel(f"{lbl} dim 2")
    a1.grid(False)

    colors = np.where(yt == 1, "#DC2626", "#2563EB")
    a2.scatter(Z2[:,0], Z2[:,1], c=colors, s=8, alpha=0.5)
    a2.legend(handles=[
        mpatches.Patch(color="#2563EB", label="Survived"),
        mpatches.Patch(color="#DC2626", label="Died"),
    ], fontsize=10)
    a2.set(title=f"{lbl} — coloured by true label")
    a2.set_xlabel(f"{lbl} dim 1"); a2.set_ylabel(f"{lbl} dim 2")
    a2.grid(False)

    plt.suptitle(f"Mamba encoder latent space  (n={n:,})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------------------------
# Feature name builder — call from run_experiment.py
# ---------------------------------------------------------------------------

def build_feature_names(
    ts_names:  List[str],
    st_names:  List[str],
    d_model:   int,
    keep_idx:  Optional[List[int]] = None,
) -> List[str]:
    """
    Build human-readable feature names matching the XGBoost input vector.

    F_final = [z_last(d) | z_mean(d) | z_max(d) | z_attn(d) |
               last(F) | mean(F) | max(F) | std(F) | miss(F) |
               static(42)]

    If keep_idx is provided (from drop_low_importance), only the kept
    features are returned — this MUST match the actual XGBoost input.

    Parameters
    ----------
    ts_names : list of time-series feature names (e.g. ["heart_rate", ...])
    st_names : list of static feature names (e.g. ["age", "gender_male", ...])
    d_model  : encoder hidden dimension
    keep_idx : optional list of column indices kept after importance filtering
    """
    names = (
        # Encoder latent — 4 pooling strategies × d_model
        [f"z_last_{i}"  for i in range(d_model)] +
        [f"z_mean_{i}"  for i in range(d_model)] +
        [f"z_max_{i}"   for i in range(d_model)] +
        [f"z_attn_{i}"  for i in range(d_model)] +
        # Raw temporal statistics — 5 stats × n_features
        [f"last_{n}"    for n in ts_names] +
        [f"mean_{n}"    for n in ts_names] +
        [f"max_{n}"     for n in ts_names] +
        [f"std_{n}"     for n in ts_names] +
        [f"miss_{n}"    for n in ts_names] +
        # Static / demographic features
        list(st_names)
    )

    if keep_idx is not None:
        names = [names[i] for i in keep_idx if i < len(names)]

    return names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_shap(
    xgb_mort,
    xgb_los,
    F_test:        np.ndarray,
    feature_names: List[str],
    y_prob_mort:   np.ndarray,
    y_true_mort:   np.ndarray,
    z_T:           Optional[np.ndarray],
    out_dir:       str,
    top_n:         int = 25,
    n_waterfall:   int = 5,
):
    """
    Compute and save all SHAP plots for mortality and LOS models.

    Feature names are automatically trimmed/extended to match the actual
    number of features in the booster (handles keep_idx filtering).
    """
    _require_shap()
    d = Path(out_dir) / "shap"
    d.mkdir(parents=True, exist_ok=True)
    print("\n[SHAP] Computing SHAP values …")

    # --- Mortality ---
    sv_m = _compute_sv(xgb_mort.booster, F_test, feature_names,
                       positive_class=True)
    _beeswarm(sv_m, "SHAP — Mortality Risk (top features)",
              d / "mortality_beeswarm.png", top_n)
    _bar(sv_m, "Feature importance — Mortality",
         d / "mortality_bar.png", top_n)
    _clinical_bar(sv_m, "Clinical feature importance — Mortality",
                  d / "mortality_clinical.png", top_n=30)

    # Waterfall for top-risk patients
    top_risk_idx = np.argsort(y_prob_mort)[-n_waterfall:][::-1]
    for rank, pidx in enumerate(top_risk_idx):
        _waterfall(
            sv_m, pidx,
            (f"Patient #{rank+1} (rank by risk)  "
             f"risk={y_prob_mort[pidx]:.3f}  "
             f"true={'died' if y_true_mort[pidx] else 'survived'}"),
            d / f"mortality_waterfall_top{rank+1}.png",
        )

    # --- LOS ---
    sv_l = _compute_sv(xgb_los.booster, F_test, feature_names,
                       positive_class=False)
    _beeswarm(sv_l, "SHAP — Length of Stay (top features)",
              d / "los_beeswarm.png", top_n)
    _bar(sv_l, "Feature importance — LOS",
         d / "los_bar.png", top_n)
    _clinical_bar(sv_l, "Clinical feature importance — LOS",
                  d / "los_clinical.png", top_n=30)

    # --- Latent space ---
    if z_T is not None and len(z_T) > 10:
        _plot_umap(z_T, y_prob_mort, y_true_mort, d / "latent_umap.png")

    print(f"[SHAP] All plots saved → {d}/")