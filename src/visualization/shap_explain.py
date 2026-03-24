"""
src/visualization/shap_explain.py
===================================
SHAP analysis for MaBoost.

Fix: run_shap() now accepts F_test_mort and F_test_los separately.
- Mortality booster uses keep_idx filtered features (678 cols)
- LOS booster uses full features (754 cols) — no keep_idx

Plots saved
-----------
shap/mortality_beeswarm.png
shap/mortality_bar.png
shap/mortality_clinical.png
shap/mortality_waterfall_top{N}.png
shap/los_beeswarm.png
shap/los_bar.png
shap/los_clinical.png
shap/latent_umap.png
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


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


def _fix_feat_names(names, n_features):
    if names is None:
        return [f"f{i}" for i in range(n_features)]
    names = list(names)
    if len(names) > n_features:
        return names[:n_features]
    if len(names) < n_features:
        return names + [f"f{i}" for i in range(len(names), n_features)]
    return names


def _compute_sv(booster, F, names, positive_class=True):
    shap  = _require_shap()
    F_safe = np.nan_to_num(F, nan=0.0)
    ex    = shap.TreeExplainer(booster)
    sv    = ex(F_safe)
    vals  = sv.values
    base  = sv.base_values
    data  = sv.data if sv.data is not None else F_safe

    if vals.ndim == 3:
        cls_idx = 1 if (positive_class and vals.shape[2] > 1) else 0
        vals = vals[:, :, cls_idx]
        if isinstance(base, np.ndarray):
            if base.ndim == 2:
                base = base[:, cls_idx]
            elif base.ndim == 1 and base.shape[0] == sv.values.shape[2]:
                base = base[cls_idx]

    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    n_features  = vals.shape[1]
    names_fixed = _fix_feat_names(names, n_features)

    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = data.reshape(-1, 1)
    if isinstance(data, np.ndarray) and data.shape[1] != n_features:
        data = F_safe[:, :n_features] if F_safe.shape[1] >= n_features else F_safe

    return shap.Explanation(
        values=vals, base_values=base,
        data=data, feature_names=names_fixed,
    )


def _beeswarm(sv, title, path, top_n=25):
    shap = _require_shap()
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
    plt.sca(ax)
    shap.plots.beeswarm(sv, max_display=top_n, show=False)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=10)
    plt.tight_layout()
    _save(fig, path)


def _bar(sv, title, path, top_n=25):
    ma    = np.abs(sv.values).mean(0)
    idx   = np.argsort(ma)[-top_n:]
    names = sv.feature_names or [f"f{i}" for i in range(len(ma))]
    colors = ["#2563EB" if not n.startswith("z_") else "#9CA3AF"
              for n in [names[i] for i in idx]]
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    ax.barh([names[i] for i in idx], ma[idx], color=colors, alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(handles=[
        mpatches.Patch(color="#2563EB", label="Clinical feature"),
        mpatches.Patch(color="#9CA3AF", label="Encoder latent (z)"),
    ], fontsize=9, loc="lower right")
    plt.tight_layout()
    _save(fig, path)


def _clinical_bar(sv, title, path, top_n=30):
    names    = sv.feature_names or []
    ma       = np.abs(sv.values).mean(0)
    clin_idx = [i for i, n in enumerate(names) if not n.startswith("z_")]
    if not clin_idx:
        return
    clin_ma    = ma[clin_idx]
    clin_names = [names[i] for i in clin_idx]
    top_idx    = np.argsort(clin_ma)[-top_n:]

    def _color(n):
        if n.startswith("last_"):  return "#059669"
        if n.startswith("mean_"):  return "#2563EB"
        if n.startswith("max_"):   return "#DC2626"
        if n.startswith("std_"):   return "#D97706"
        if n.startswith("miss_"):  return "#7C3AED"
        return "#374151"

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    ax.barh([clin_names[i] for i in top_idx], clin_ma[top_idx],
            color=[_color(clin_names[i]) for i in top_idx], alpha=0.85)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    ax.tick_params(axis="y", labelsize=9)
    ax.legend(handles=[
        mpatches.Patch(color="#059669", label="Last observed value"),
        mpatches.Patch(color="#2563EB", label="Mean over stay"),
        mpatches.Patch(color="#DC2626", label="Max over stay"),
        mpatches.Patch(color="#D97706", label="Std (variability)"),
        mpatches.Patch(color="#7C3AED", label="Miss rate"),
        mpatches.Patch(color="#374151", label="Static / demographic"),
    ], fontsize=8, loc="lower right")
    plt.tight_layout()
    _save(fig, path)


def _waterfall(sv, pidx, title, path):
    shap = _require_shap()
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.plots.waterfall(sv[pidx], show=False)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, path)


def _plot_umap(z_T, y_prob, y_true, path):
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
        lbl = "PCA"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    sc = a1.scatter(Z2[:,0], Z2[:,1], c=yp, cmap="RdBu_r", s=8, alpha=0.55, vmin=0, vmax=1)
    plt.colorbar(sc, ax=a1, label="Predicted mortality risk")
    a1.set_title(f"{lbl} — predicted risk"); a1.grid(False)
    colors = np.where(yt == 1, "#DC2626", "#2563EB")
    a2.scatter(Z2[:,0], Z2[:,1], c=colors, s=8, alpha=0.5)
    a2.legend(handles=[
        mpatches.Patch(color="#2563EB", label="Survived"),
        mpatches.Patch(color="#DC2626", label="Died"),
    ], fontsize=10)
    a2.set_title(f"{lbl} — true label"); a2.grid(False)
    plt.suptitle(f"Mamba encoder latent space  (n={n:,})", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, path)


def build_feature_names(ts_names, st_names, d_model, keep_idx=None):
    """
    Build human-readable feature names for XGBoost input vector.
    If keep_idx provided, returns only kept features (mortality booster).
    For LOS booster (no keep_idx), returns all 754 names.
    """
    names = (
        [f"z_last_{i}" for i in range(d_model)] +
        [f"z_mean_{i}" for i in range(d_model)] +
        [f"z_max_{i}"  for i in range(d_model)] +
        [f"z_attn_{i}" for i in range(d_model)] +
        [f"last_{n}"   for n in ts_names] +
        [f"mean_{n}"   for n in ts_names] +
        [f"max_{n}"    for n in ts_names] +
        [f"std_{n}"    for n in ts_names] +
        [f"miss_{n}"   for n in ts_names] +
        list(st_names)
    )
    if keep_idx is not None:
        names = [names[i] for i in keep_idx if i < len(names)]
    return names


def run_shap(
    xgb_mort,
    xgb_los,
    F_test_mort:   np.ndarray,   # filtered by keep_idx  (e.g. 678 cols)
    F_test_los:    np.ndarray,   # full features          (e.g. 754 cols)
    feature_names: List[str],    # names for mortality (len = F_test_mort cols)
    y_prob_mort:   np.ndarray,
    y_true_mort:   np.ndarray,
    z_T:           Optional[np.ndarray],
    out_dir:       str,
    top_n:         int = 25,
    n_waterfall:   int = 5,
    feature_names_los: Optional[List[str]] = None,  # if None, auto-build from F_test_los
):
    """
    Compute and save all SHAP plots.

    F_test_mort — features for mortality booster (keep_idx filtered)
    F_test_los  — features for LOS booster (full, no filtering)
    feature_names_los — optional separate names for LOS; if None uses
                        _fix_feat_names(feature_names, F_test_los.shape[1])
    """
    _require_shap()
    d = Path(out_dir) / "shap"
    d.mkdir(parents=True, exist_ok=True)
    print("\n[SHAP] Computing SHAP values …")

    # --- Mortality (filtered features) ---
    sv_m = _compute_sv(xgb_mort.booster, F_test_mort, feature_names,
                       positive_class=True)
    _beeswarm(sv_m, "SHAP — Mortality Risk (top features)",
              d / "mortality_beeswarm.png", top_n)
    _bar(sv_m, "Feature importance — Mortality",
         d / "mortality_bar.png", top_n)
    _clinical_bar(sv_m, "Clinical feature importance — Mortality",
                  d / "mortality_clinical.png", top_n=30)
    top_risk_idx = np.argsort(y_prob_mort)[-n_waterfall:][::-1]
    for rank, pidx in enumerate(top_risk_idx):
        _waterfall(
            sv_m, pidx,
            (f"Patient #{rank+1}  risk={y_prob_mort[pidx]:.3f}  "
             f"true={'died' if y_true_mort[pidx] else 'survived'}"),
            d / f"mortality_waterfall_top{rank+1}.png",
        )

    # --- LOS (full features — no keep_idx) ---
    los_names = feature_names_los or _fix_feat_names(
        feature_names, F_test_los.shape[1]
    )
    sv_l = _compute_sv(xgb_los.booster, F_test_los, los_names,
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