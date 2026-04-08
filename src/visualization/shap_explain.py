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
import pickle
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


def _stage2_full_feature_names(
    base_names: List[str],
    ckpt_dir: Optional[str],
) -> List[str]:
    if not ckpt_dir:
        return list(base_names)

    meta_path = Path(ckpt_dir) / "stage2_meta.pkl"
    if not meta_path.exists():
        return list(base_names)

    with meta_path.open("rb") as f:
        meta = pickle.load(f)

    names = list(base_names)

    if meta.get("use_static_cross", False):
        n_zmulti = int(meta.get("n_zmulti_dims", 0))
        n_static = int(meta.get("n_static_dims", 0))
        top_z = min(int(meta.get("static_cross_top_z", 16)), max(n_zmulti, 0))
        bin_idx = list(meta.get("static_cross_bin_idx", []))
        static_start = len(names) - n_static
        cross_names = []
        if n_zmulti > 0 and n_static > 0 and static_start >= 0:
            for si in bin_idx:
                if si < 0 or (static_start + si) >= len(names):
                    continue
                s_name = names[static_start + si]
                for zi in range(top_z):
                    if zi >= len(names):
                        break
                    z_name = names[zi]
                    cross_names.append(f"cross__{s_name}__x__{z_name}")
        names.extend(cross_names)

    if meta.get("use_interaction_features", False):
        pair_names = []
        for i, j in meta.get("interaction_pairs", []):
            if 0 <= i < len(names) and 0 <= j < len(names):
                pair_names.append(f"int__{names[i]}__x__{names[j]}")
            else:
                pair_names.append(f"int__f{i}__x__f{j}")
        names = pair_names + names

    return names


_CLINICAL_DISPLAY: dict = {
    # ── Vital signs ─────────────────────────────────────────────────────────
    "heart_rate":       "Heart Rate (bpm)",
    "sbp":              "Systolic BP (mmHg)",
    "dbp":              "Diastolic BP (mmHg)",
    "map":              "Mean Arterial Pressure",
    "spo2":             "SpO2 (%)",
    "resp_rate":        "Respiratory Rate",
    "temp_c":           "Temperature (°C)",
    "temp_f":           "Temperature (°F)",
    "gcs_total":        "GCS Total",
    "gcs_verbal":       "GCS Verbal",
    "gcs_motor":        "GCS Motor",
    "gcs_eye":          "GCS Eye",
    "glucose":          "Glucose",
    "fio2":             "FiO2 (%)",
    "urine_output":     "Urine Output (mL)",
    # ── Labs ────────────────────────────────────────────────────────────────
    "lactate":          "Lactate",
    "bicarbonate":      "Bicarbonate",
    "ph":               "pH (arterial)",
    "pao2":             "PaO2",
    "paco2":            "PaCO2",
    "creatinine":       "Creatinine",
    "bun":              "BUN",
    "sodium":           "Sodium",
    "potassium":        "Potassium",
    "chloride":         "Chloride",
    "calcium":          "Calcium",
    "magnesium":        "Magnesium",
    "phosphate":        "Phosphate",
    "hemoglobin":       "Hemoglobin",
    "hematocrit":       "Hematocrit",
    "wbc":              "WBC",
    "platelet":         "Platelet Count",
    "bilirubin_total":  "Total Bilirubin",
    "alt":              "ALT",
    "ast":              "AST",
    "inr":              "INR",
    "pt":               "PT",
    "ptt":              "PTT",
    "crp":              "CRP",
    "procalcitonin":    "Procalcitonin",
    # ── Static / Demographics ───────────────────────────────────────────────
    "age":                      "Age (years)",
    "gender_male":              "Male gender",
    "emergency_admit":          "Emergency admission",
    "elective_admit":           "Elective admission",
    "insurance_medicare":       "Medicare insurance",
    "insurance_medicaid":       "Medicaid insurance",
    "insurance_private":        "Private insurance",
    "micu":                     "Medical ICU",
    "sicu":                     "Surgical ICU",
    "cicu":                     "Cardiac ICU",
    "nicu":                     "Neuro ICU",
    "csru":                     "Cardiac Surgery Recovery",
    # ── Elixhauser comorbidities ───────────────────────────────────────────
    "chf":                      "Congestive Heart Failure",
    "cardiac_arrhythmia":       "Cardiac Arrhythmia",
    "valvular_disease":         "Valvular Disease",
    "pulmonary_circulation":    "Pulmonary Circulation Disorder",
    "pvd":                      "Peripheral Vascular Disease",
    "hypertension":             "Hypertension",
    "paralysis":                "Paralysis",
    "other_neurological":       "Other Neurological Disorder",
    "copd":                     "COPD",
    "diabetes_uncomplicated":   "Diabetes (uncomplicated)",
    "diabetes_complicated":     "Diabetes (complicated)",
    "hypothyroidism":           "Hypothyroidism",
    "renal_failure":            "Renal Failure",
    "liver_disease":            "Liver Disease",
    "peptic_ulcer":             "Peptic Ulcer Disease",
    "aids":                     "AIDS/HIV",
    "lymphoma":                 "Lymphoma",
    "metastatic_cancer":        "Metastatic Cancer",
    "solid_tumor":              "Solid Tumor",
    "rheumatoid_arthritis":     "Rheumatoid Arthritis",
    "coagulopathy":             "Coagulopathy",
    "obesity":                  "Obesity",
    "weight_loss":              "Weight Loss",
    "fluid_electrolyte":        "Fluid/Electrolyte Disorder",
    "blood_loss_anemia":        "Blood Loss Anemia",
    "deficiency_anemia":        "Deficiency Anemia",
    "alcohol_abuse":            "Alcohol Abuse",
    "drug_abuse":               "Drug Abuse",
    "psychoses":                "Psychoses",
    "depression":               "Depression",
}

_STAT_SUFFIX_DISPLAY: dict = {
    "last_":  " (last obs)",
    "mean_":  " (stay mean)",
    "max_":   " (stay max)",
    "std_":   " (variability)",
    "miss_":  " missing rate",
}


def hospital_feature_names(raw_names: list) -> list:
    """
    Convert internal feature names to human-readable hospital display names.

    Mapping:
        last_heart_rate   → "Heart Rate (bpm) (last obs)"
        mean_lactate      → "Lactate (stay mean)"
        miss_creatinine   → "Creatinine missing rate"
        age               → "Age (years)"
        chf               → "Congestive Heart Failure"
        z_last_0          → "Mamba Latent·last [dim 0]"
        z_mean_42         → "Mamba Latent·mean [dim 42]"
        cross__gender_male__x__z_last_0 → "cross: Male × Mamba-last[0]"
        int__f0__x__f1    → "interaction: f0 × f1"

    Parameters
    ----------
    raw_names : list of str
        Feature names as produced by build_feature_names().

    Returns
    -------
    list of str
        Human-readable display names, same length as raw_names.
    """
    out = []
    for name in raw_names:
        # ── Mamba latent dims ──────────────────────────────────────────────
        if name.startswith("z_last_"):
            out.append(f"Mamba Latent·last [dim {name[7:]}]")
        elif name.startswith("z_mean_"):
            out.append(f"Mamba Latent·mean [dim {name[7:]}]")
        elif name.startswith("z_max_"):
            out.append(f"Mamba Latent·max [dim {name[6:]}]")
        elif name.startswith("z_attn_"):
            out.append(f"Mamba Latent·attn [dim {name[7:]}]")
        # ── Cross / interaction features ───────────────────────────────────
        elif name.startswith("cross__"):
            parts = name[7:].split("__x__")
            if len(parts) == 2:
                a = hospital_feature_names([parts[0]])[0]
                b = hospital_feature_names([parts[1]])[0]
                out.append(f"Cross: {a} × {b}")
            else:
                out.append(name)
        elif name.startswith("int__"):
            parts = name[5:].split("__x__")
            if len(parts) == 2:
                out.append(f"Interact: {parts[0]} × {parts[1]}")
            else:
                out.append(name)
        else:
            # ── stat prefix + clinical name ────────────────────────────────
            matched = False
            for prefix, suffix in _STAT_SUFFIX_DISPLAY.items():
                if name.startswith(prefix):
                    feat_code = name[len(prefix):]
                    clinical = _CLINICAL_DISPLAY.get(feat_code, feat_code.replace("_", " ").title())
                    out.append(clinical + suffix)
                    matched = True
                    break
            if not matched:
                # ── static / demographic ───────────────────────────────────
                out.append(_CLINICAL_DISPLAY.get(name, name.replace("_", " ").title()))
    return out


def build_feature_names(ts_names, st_names, d_model, keep_idx=None, ckpt_dir=None,
                        hospital_readable: bool = False):
    """
    Build human-readable feature names for the Stage 2 XGBoost input vector.

    If `ckpt_dir` contains `stage2_meta.pkl`, reconstructs the exact transformed
    feature space used by Stage 2:
      1. base features [z_multi | raw_stats | static]
      2. optional static x z_multi cross features (appended)
      3. optional interaction features (prepended)

    If keep_idx is provided, returns only the kept mortality features from that
    transformed space. Without ckpt_dir, this falls back to base-feature names.

    Parameters
    ----------
    hospital_readable : bool
        If True, applies hospital_feature_names() to convert internal codes
        to human-readable clinical display names (e.g. for SHAP plot axes).
    """
    base_names = (
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
    names = _stage2_full_feature_names(base_names, ckpt_dir=ckpt_dir)
    if keep_idx is not None:
        names = [names[i] for i in keep_idx if i < len(names)]
    if hospital_readable:
        names = hospital_feature_names(names)
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
