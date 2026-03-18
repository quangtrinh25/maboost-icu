"""
src/visualization/shap_explain.py
===================================
SHAP analysis for MaBoost — mortality XGBoost + LOS XGBoost.

Plots saved
-----------
shap/mortality_beeswarm.png
shap/mortality_bar.png
shap/mortality_waterfall_top{N}.png
shap/los_beeswarm.png
shap/los_bar.png
shap/latent_umap.png  (z_T coloured by risk + true label)
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _require_shap():
    try:
        import shap
        return shap
    except ImportError:
        raise ImportError("Install shap first:  pip install shap")


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SHAP] {path.name}")


def _compute_sv(booster, F, names, positive_class=True):
    shap = _require_shap()
    ex = shap.TreeExplainer(booster)
    sv = ex(F)

    # Normalize shapes: ensure values -> 2D (n_samples, n_features)
    vals = sv.values
    base = sv.base_values
    data = sv.data

    # If values is 3D: (n_samples, n_features, n_classes)
    if vals.ndim == 3:
        # Choose class index: prefer positive_class (1) if available, else last axis
        class_axis = 1 if vals.shape[2] > 1 and positive_class else min(1, vals.shape[2]-1)
        try:
            vals2 = vals[:, :, class_axis]
            # base_values may be shape (n_samples, n_classes) or (n_classes,)
            if isinstance(base, np.ndarray) and base.ndim == 2:
                base2 = base[:, class_axis]
            elif isinstance(base, np.ndarray) and base.ndim == 1 and base.shape[0] == vals.shape[2]:
                base2 = base[class_axis]
            else:
                base2 = base
        except Exception:
            # Fallback: sum over class axis (rare)
            vals2 = vals.sum(axis=2)
            base2 = base
        vals = vals2
        base = base2

    # If values is 1D (rare), make it 2D with single feature
    if vals.ndim == 1:
        vals = vals.reshape(-1, 1)

    # Ensure data is 2D and matches feature dim
    if data is None:
        data = F
    else:
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

    n_features = vals.shape[1]
    # Fix feature names length: trim or extend with generic names
    if names is None:
        names = [f"f{i}" for i in range(n_features)]
    else:
        if len(names) > n_features:
            names = names[:n_features]
        elif len(names) < n_features:
            names = list(names) + [f"f{i}" for i in range(len(names), n_features)]

    # Build a new Explanation object with consistent shapes
    sv2 = shap.Explanation(
        values=vals,
        base_values=base,
        data=data,
        feature_names=names
    )
    return sv2


def _beeswarm(sv, title, path, top_n=25):
    shap = _require_shap()
    # Validate sv before plotting
    vals = sv.values
    if vals is None:
        raise ValueError("SHAP values are empty.")
    if vals.ndim != 2:
        # Try to coerce to 2D
        if vals.ndim == 3:
            vals = vals[:, :, 0]
        else:
            vals = np.reshape(vals, (vals.shape[0], -1))
        sv = shap.Explanation(values=vals, base_values=sv.base_values, data=sv.data, feature_names=sv.feature_names)

    # Ensure feature_names length matches values
    n_feats = sv.values.shape[1]
    fnames = sv.feature_names or [f"f{i}" for i in range(n_feats)]
    if len(fnames) != n_feats:
        if len(fnames) > n_feats:
            fnames = fnames[:n_feats]
        else:
            fnames = list(fnames) + [f"f{i}" for i in range(len(fnames), n_feats)]
        sv.feature_names = fnames

    fig, ax = plt.subplots(figsize=(9, max(5, top_n*0.32)))
    plt.sca(ax)
    shap.plots.beeswarm(sv, max_display=top_n, show=False)
    ax.set_title(title)
    _save(fig, path)



def _bar(sv, title, path, top_n=25):
    ma  = np.abs(sv.values).mean(0)
    idx = np.argsort(ma)[-top_n:]
    names = sv.feature_names or [f"f{i}" for i in range(len(ma))]
    fig, ax = plt.subplots(figsize=(8, max(5, top_n*0.32)))
    ax.barh([names[i] for i in idx], ma[idx], color="#2563EB", alpha=0.8)
    ax.set(xlabel="Mean |SHAP|", title=title)
    _save(fig, path)


def _waterfall(sv, pidx, title, path):
    shap = _require_shap()
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.sca(ax)
    shap.plots.waterfall(sv[pidx], show=False)
    ax.set_title(title)
    _save(fig, path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_shap(
    xgb_mort, xgb_los,
    F_test:       np.ndarray,
    feature_names: List[str],
    y_prob_mort:  np.ndarray,
    y_true_mort:  np.ndarray,
    z_T:          Optional[np.ndarray],
    out_dir:      str,
    top_n:        int = 25,
    n_waterfall:  int = 5,
):
    """Compute and save all SHAP plots for both tasks."""
    _require_shap()
    d = Path(out_dir) / "shap"
    d.mkdir(parents=True, exist_ok=True)
    print("\n[SHAP] Computing SHAP values …")

    # Mortality
    sv_m = _compute_sv(xgb_mort.booster, F_test, feature_names, positive_class=True)
    _beeswarm(sv_m, "SHAP beeswarm — mortality", d/"mortality_beeswarm.png", top_n)
    _bar(sv_m, "Feature importance — mortality", d/"mortality_bar.png", top_n)
    for i, pidx in enumerate(np.argsort(y_prob_mort)[-n_waterfall:][::-1]):
        _waterfall(sv_m, pidx,
                   f"Patient {i+1}  risk={y_prob_mort[pidx]:.3f}  true={y_true_mort[pidx]}",
                   d/f"mortality_waterfall_top{i+1}.png")

    # LOS
    sv_l = _compute_sv(xgb_los.booster, F_test, feature_names, positive_class=False)
    _beeswarm(sv_l, "SHAP beeswarm — LOS", d/"los_beeswarm.png", top_n)
    _bar(sv_l, "Feature importance — LOS", d/"los_bar.png", top_n)

    # Latent UMAP
    if z_T is not None:
        _plot_umap(z_T, y_prob_mort, y_true_mort, d/"latent_umap.png")

    print(f"[SHAP] All plots saved → {d}/")


def _plot_umap(z_T, y_prob, y_true, path):
    n   = min(3000, len(z_T))
    idx = np.random.default_rng(42).choice(len(z_T), n, replace=False)
    Z, yp, yt = z_T[idx], y_prob[idx], y_true[idx]
    try:
        from umap import UMAP
        Z2 = UMAP(n_components=2, random_state=42, verbose=False).fit_transform(Z)
        lbl = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        Z2 = PCA(n_components=2, random_state=42).fit_transform(Z)
        lbl = "PCA"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    sc = a1.scatter(Z2[:,0],Z2[:,1],c=yp,cmap="RdBu_r",s=7,alpha=0.55,vmin=0,vmax=1)
    plt.colorbar(sc,ax=a1,label="Predicted mortality risk")
    a1.set(title=f"{lbl} — coloured by predicted risk"); a1.grid(False)
    from matplotlib.patches import Patch
    colors=np.where(yt==1,"#DC2626","#2563EB")
    a2.scatter(Z2[:,0],Z2[:,1],c=colors,s=7,alpha=0.5)
    a2.legend(handles=[Patch(color="#2563EB",label="Survived"),Patch(color="#DC2626",label="Died")])
    a2.set(title=f"{lbl} — coloured by true label"); a2.grid(False)
    plt.suptitle(f"Mamba latent z_T  (n={n:,})",fontweight="bold")
    plt.tight_layout(); _save(fig, path)