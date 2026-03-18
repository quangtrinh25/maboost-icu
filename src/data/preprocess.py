"""
src/data/preprocess.py
======================
8-stage Polars ETL for MaBoost (CSV.gz edition).

Stages
------
1. infer_age          DOB → anchor_age; clamp MIMIC-III 300-yr artefact → 91.4
2. filter_by_age      drop stays where age < min_age
3. align_labs         restrict LABEVENTS to each ICU window [intime, outtime]
4. merge_events       chart ∪ labs; °F → °C; itemid → feature name
5. compute_delta_t    τ_t (seconds) via Polars .over() — zero Python loops
6. build_sequences    pivot → (seq_len, 47) float32 arrays
7. build_static       42-dim static vector per stay (demographics + Elixhauser)
8. build_labels       y_mort (0/1)  +  y_los (fractional days)
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------
_CHART: Dict[str, List[int]] = {
    "heart_rate":   [211, 220045],
    "sbp":          [51, 442, 455, 6701, 220179, 220050],
    "dbp":          [8368, 8440, 8441, 8555, 220180, 220051],
    "map":          [52, 456, 6702, 443, 220052, 220181, 225312],
    "spo2":         [646, 220277],
    "resp_rate":    [615, 618, 220210, 224690],
    "temp_c":       [223762, 676],
    "temp_f":       [223761, 678],     # converted to °C
    "gcs_total":    [198, 220739],
    "gcs_verbal":   [723, 223900],
    "gcs_motor":    [454, 223901],
    "gcs_eye":      [184],
    "glucose":      [807, 811, 1529, 3745, 3744, 225664, 220621, 226537],
    "fio2":         [3420, 190, 223835, 3422],
    "urine_output": [40055, 43175, 40069, 40094, 40715, 40473, 40085,
                     40057, 40056, 226559, 226560, 226561, 226584,
                     226563, 226564, 226565, 226557, 226558],
}
_LAB: Dict[str, List[int]] = {
    "lactate":         [816, 225668],
    "bicarbonate":     [227443, 1716, 227686],
    "ph":              [780, 860, 1126, 1673, 3734, 3750, 220274, 220734, 223830],
    "pao2":            [779, 220224],
    "paco2":           [778, 220235],
    "creatinine":      [791, 220615, 1525],
    "bun":             [781, 225624, 1162],
    "sodium":          [837, 220645, 1536],
    "potassium":       [833, 220640, 1535],
    "chloride":        [788, 220602, 1522],
    "calcium":         [786, 220635],
    "magnesium":       [1532, 227442],
    "phosphate":       [789, 1524],
    "hemoglobin":      [814, 220228],
    "hematocrit":      [813, 220545],
    "wbc":             [1542, 220546, 3834, 4200],
    "platelet":        [828, 220505, 227457],
    "bilirubin_total": [848, 225690, 1538],
    "alt":             [770, 220644],
    "ast":             [3702, 220587],
    "inr":             [1530, 227467],
    "pt":              [824, 227465],
    "ptt":             [829, 227466],
    "crp":             [1335],
    "procalcitonin":   [225677],
}
_ITEMID_MAP: Dict[int, str] = {}
for _f, _ids in {**_CHART, **_LAB}.items():
    for _i in _ids:
        _ITEMID_MAP[_i] = _f
_FAHRENHEIT: set = set(_CHART["temp_f"])

FEATURE_NAMES: List[str] = list(dict.fromkeys(_ITEMID_MAP.values()))  # 47
N_FEAT = len(FEATURE_NAMES)

_ELIX: Dict[str, List[str]] = {
    "chf":                    ["4280","4281","42820","42821","42822","42823","42830","42831","42832","42833","4289"],
    "cardiac_arrhythmia":     ["42610","42611","42613","4262","4263","42640","42641","42650","4269","42731","4279"],
    "valvular_disease":       ["3940","3941","3942","3949","3950","3951","3952","3959","4240","4241","4242","4243","4249"],
    "pulmonary_circulation":  ["41511","41519","4160","4161","4168","4169"],
    "pvd":                    ["4431","4432","4433","4434","4439","4442","4443","V434"],
    "hypertension":           ["4011","4019","40200","40210","40290","40300","40310","40390","4372"],
    "paralysis":              ["3341","3440","34410","34411","34412","34419","3442","3443","3444","3449"],
    "other_neurological":     ["3310","3311","3312","332","3320","3321","340","341","342","343"],
    "copd":                   ["4168","4169","5064","5081","5088"],
    "diabetes_uncomplicated": ["2500","2501","2502","2503","2508","2509"],
    "diabetes_complicated":   ["2504","2505","2506","2507"],
    "hypothyroidism":         ["2409","243","2440","2441","2448","2449"],
    "renal_failure":          ["585","5853","5854","5855","5856","5859","586","V420","V451"],
    "liver_disease":          ["4560","4561","4562","5722","5723","5724","5725","5726","5727","5728","V427"],
    "peptic_ulcer":           ["53141","53151","53161","53170","53171","53191"],
    "aids":                   ["042","0431","0432","0433","0434","0435"],
    "lymphoma":               ["20000","20001","20002","20003","20100","20200","20300","20380","20400","20500","2386"],
    "metastatic_cancer":      ["1960","1961","1962","1963","1964","1970","1971","1972","1973","1990","1991","1992"],
    "solid_tumor":            ["1400","1401","1410","1411","1420","1421","1429"],
    "rheumatoid_arthritis":   ["7100","7101","7102","7103","7104","7140","7141","7142","725"],
    "coagulopathy":           ["2860","2861","2862","2863","2864","2865","2866","2867","2868","2869"],
    "obesity":                ["2780","27800","27801","27803"],
    "weight_loss":            ["260","261","262","2630","2631","2632","2638","2639"],
    "fluid_electrolyte":      ["2536","276"],
    "blood_loss_anemia":      ["2800"],
    "deficiency_anemia":      ["2801","2808","2809","281"],
    "alcohol_abuse":          ["2911","2912","2913","3030","3039","5710","5711","5712","5713"],
    "drug_abuse":             ["292","304","3050","3051","3052","3053","3054","3055"],
    "psychoses":              ["2938","295","2950","2952","2953","2954","2955","2958","2959"],
    "depression":             ["3004","30112","3090","3091","311"],
}
STATIC_NAMES: List[str] = [
    "age", "gender_male", "emergency_admit", "elective_admit",
    "insurance_medicare", "insurance_medicaid", "insurance_private",
    "micu", "sicu", "cicu", "nicu", "csru",
] + list(_ELIX.keys())   # 12 + 30 = 42

PAD_TAU = 60.0
MIN_TAU = 1.0
MIMIC3_MAX_AGE = 91.4


# ---------------------------------------------------------------------------
# Stage 1 — infer age
# ---------------------------------------------------------------------------
def infer_age(patients: pl.DataFrame, admissions: pl.DataFrame) -> pl.DataFrame:
    if patients.is_empty():
        return patients
    cols = {c.lower() for c in patients.columns}
    for col in ("anchor_age", "age"):
        if col in cols:
            return patients.with_columns(
                pl.when(pl.col(col) > 150).then(pl.lit(MIMIC3_MAX_AGE))
                .when(pl.col(col) < 0).then(pl.lit(0.0))
                .otherwise(pl.col(col)).cast(pl.Float32).alias("anchor_age")
            )
    if "dob" in cols and not admissions.is_empty() and "admittime" in {c.lower() for c in admissions.columns}:
        first = (
            admissions.select(["subject_id", "admittime"])
            .with_columns(pl.col("admittime").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False))
            .group_by("subject_id")
            .agg(pl.col("admittime").min().alias("fa"))
        )
        return (
            patients.join(first, on="subject_id", how="left")
            .with_columns(pl.col("dob").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False))
            .with_columns(
                ((pl.col("fa") - pl.col("dob")).dt.total_days() / 365.25)
                .clip(0, 150).fill_null(0.0).cast(pl.Float32).alias("anchor_age")
            ).drop("fa")
        )
    return patients.with_columns(pl.lit(0.0).cast(pl.Float32).alias("anchor_age"))


# ---------------------------------------------------------------------------
# Stage 2 — age filter
# ---------------------------------------------------------------------------
def filter_by_age(icu: pl.DataFrame, patients: pl.DataFrame, min_age: int = 18) -> pl.DataFrame:
    if icu.is_empty() or "anchor_age" not in patients.columns:
        return icu
    eligible = patients.filter(pl.col("anchor_age") >= min_age).select("subject_id")
    n0 = icu.height
    out = icu.join(eligible, on="subject_id", how="semi")
    print(f"  Age filter (≥{min_age}): {out.height:,} / {n0:,} stays kept")
    return out


# ---------------------------------------------------------------------------
# Stage 3 — align lab events to ICU window
# ---------------------------------------------------------------------------
def align_labs(labs: pl.DataFrame, icu: pl.DataFrame) -> pl.DataFrame:
    # Robust: ensure we always return a polars DataFrame (never None)
    if labs is None or labs.is_empty() or icu is None or icu.is_empty():
        return pl.DataFrame()
    icu_cols = {c.lower() for c in icu.columns}
    if not {"hadm_id", "icustay_id", "intime", "outtime"}.issubset(icu_cols):
        return pl.DataFrame()

    win = icu.select(["hadm_id", "icustay_id", "intime", "outtime"]).with_columns([
        pl.col("hadm_id").cast(pl.Int64, strict=False),
        pl.col("intime").cast(pl.Datetime, strict=False),
        pl.col("outtime").cast(pl.Datetime, strict=False),
    ])
    # Convert charttime to Datetime only for the time-window filter,
    # then keep original str charttime so it matches chartevents type
    # and can be safely concatenated in merge_events.
    if "charttime" in labs.columns:
        ct_dtype = labs.schema.get("charttime", None)
    else:
        ct_dtype = None

    if ct_dtype is not None and ct_dtype == pl.Utf8:
        ct_expr = pl.col("charttime").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False).alias("_ct_dt")
    else:
        # already datetime (or unknown) — use as-is (cast defensively)
        ct_expr = pl.col("charttime").cast(pl.Datetime, strict=False).alias("_ct_dt")

    out = (
        labs.with_columns([
            ct_expr,
            pl.col("hadm_id").cast(pl.Int64, strict=False),
        ])
        .join(win, on="hadm_id", how="inner")
        .filter((pl.col("_ct_dt") >= pl.col("intime")) &
                (pl.col("_ct_dt") <= pl.col("outtime")))
        .drop(["intime", "outtime", "hadm_id", "_ct_dt"])
    )
    return out


# ---------------------------------------------------------------------------
# Stage 4 — merge events
# ---------------------------------------------------------------------------
def merge_events(chart: pl.DataFrame, labs: pl.DataFrame, icu: pl.DataFrame) -> pl.DataFrame:
    # Robust: accept None inputs and coerce to empty DataFrame
    if chart is None:
        chart = pl.DataFrame()
    if labs is None:
        labs = pl.DataFrame()
    if icu is None:
        icu = pl.DataFrame()

    # Cast icustay_id in icu to i64 so it matches chartevents (which may load as i64 or str)
    valid  = icu.select(pl.col("icustay_id").cast(pl.Int64, strict=False)) if not icu.is_empty() else pl.DataFrame()
    imap   = pl.DataFrame({"itemid": list(_ITEMID_MAP), "feat": list(_ITEMID_MAP.values())})
    parts  = []

    if not chart.is_empty() and "icustay_id" in chart.columns:
        c = (chart
                  .with_columns(pl.col("icustay_id").cast(pl.Int64, strict=False))
                  .filter(pl.col("valuenum").is_not_null())
                  .join(valid, on="icustay_id", how="semi")
                  .with_columns(
                      pl.when(pl.col("itemid").is_in(list(_FAHRENHEIT)))
                      .then((pl.col("valuenum") - 32.0) * 5.0 / 9.0)
                      .otherwise(pl.col("valuenum")).alias("valuenum")
                  ))
        parts.append(c)

    if not labs.is_empty():
        parts.append(labs.filter(pl.col("valuenum").is_not_null()))

    if not parts:
        raise RuntimeError("No event data found. Check CHARTEVENTS.csv.gz path.")

    combined = pl.concat(parts, how="diagonal") if len(parts) > 1 else parts[0]
    combined = (
        combined.join(imap, on="itemid", how="inner")
        .drop("itemid").rename({"feat": "feature_name"})
    )
    print(f"  Merge       : {combined.height:,} events · "
          f"{combined['icustay_id'].n_unique():,} stays · "
          f"{combined['feature_name'].n_unique()} features")
    return combined


# ---------------------------------------------------------------------------
# Stage 5 — delta-t
# ---------------------------------------------------------------------------
def compute_delta_t(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute tau_t (seconds) between consecutive observations within each icustay.
    Robust to charttime being either string or already Datetime.
    Leaves 'charttime' column intact for downstream code.
    """
    if df is None or df.is_empty():
        return pl.DataFrame()

    # Determine if charttime column is string or already datetime
    ct_dtype = df.schema.get("charttime", None)

    if ct_dtype == pl.Utf8:
        # charttime is string: parse to datetime and produce epoch seconds
        df2 = (
            df.with_columns(
                pl.col("charttime").str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False).alias("_ct_dt")
            )
            .with_columns(pl.col("_ct_dt").dt.epoch("s").alias("_ts_epoch"))
        )
    else:
        # charttime already datetime (or unknown): cast defensively to Datetime then epoch
        df2 = (
            df.with_columns(
                pl.col("charttime").cast(pl.Datetime, strict=False).alias("_ct_dt")
            )
            .with_columns(pl.col("_ct_dt").dt.epoch("s").alias("_ts_epoch"))
        )

    # compute delta per icustay using epoch seconds
    out = (
        df2.sort(["icustay_id", "_ts_epoch"])
        .with_columns(
            (pl.col("_ts_epoch") - pl.col("_ts_epoch").shift(1).over("icustay_id")).alias("_raw")
        )
        .with_columns(
            pl.col("_raw").fill_null(pl.col("_raw").median().over("icustay_id"))
            .clip(lower_bound=MIN_TAU).cast(pl.Float32).alias("tau_t")
        )
        # keep original charttime and other columns; drop helpers
        .drop(["_raw", "_ts_epoch", "_ct_dt"])
    )
    return out


# ---------------------------------------------------------------------------
# Stage 6 — sequences
# ---------------------------------------------------------------------------
def build_sequences(df: pl.DataFrame, seq_len: int) -> Dict:
    """
    Pivot long events -> per-stay (seq_len, N_FEAT) arrays.
    Robust sorting: prefer epoch 'ts' if present, else 'charttime'.
    """
    fidx = {f: i for i, f in enumerate(FEATURE_NAMES)}
    out  = {}
    if df is None or df.is_empty():
        print(f"  Sequences   : 0 stays ({seq_len}×{N_FEAT})")
        return out

    # group_by returns tuples like (sid,), iterate safely
    for (sid,), grp in df.group_by(["icustay_id"]):
        # Robust sort key: prefer numeric epoch-like column 'ts' or 'tau_t' if present,
        # otherwise fall back to 'charttime'.
        if "ts" in grp.columns:
            grp = grp.sort("ts")
        elif "tau_t" in grp.columns:
            # tau_t is per-row delta; prefer sorting by charttime if available
            if "charttime" in grp.columns:
                grp = grp.sort("charttime")
            else:
                grp = grp.sort("tau_t")
        elif "charttime" in grp.columns:
            grp = grp.sort("charttime")
        else:
            # fallback: keep original order
            grp = grp

        T   = grp.height
        seq  = np.zeros((T, N_FEAT), dtype=np.float32)
        mask = np.zeros((T, N_FEAT), dtype=np.float32)

        # iterate features present in this stay
        for (fn,), sub in grp.group_by(["feature_name"]):
            j = fidx.get(fn)
            if j is None:
                continue
            # sort sub by same robust key as above
            if "ts" in sub.columns:
                sub_sorted = sub.sort("ts")
            elif "charttime" in sub.columns:
                sub_sorted = sub.sort("charttime")
            else:
                sub_sorted = sub
            v = sub_sorted["valuenum"].to_numpy().astype(np.float32)
            n = min(len(v), T)
            seq[:n, j] = v[:n]
            mask[:n, j] = 1.0

        # tau: prefer tau_t column if present, else fallback to PAD_TAU for padding
        if "tau_t" in grp.columns:
            tau = grp["tau_t"].to_numpy().astype(np.float32)
        else:
            # if no tau_t, create a default increasing array (or PAD_TAU)
            tau = np.full(T, PAD_TAU, dtype=np.float32)

        if T >= seq_len:
            seq, tau, mask = seq[-seq_len:], tau[-seq_len:], mask[-seq_len:]
        else:
            pad  = seq_len - T
            z    = np.zeros((pad, N_FEAT), dtype=np.float32)
            seq  = np.vstack([z, seq])
            tau  = np.concatenate([np.full(pad, PAD_TAU, dtype=np.float32), tau])
            mask = np.vstack([z, mask])

        # FIX: MIN_OBS_PER_STAY filter removed from here.
        # Filtering on the full dataset before train/val/test split causes data leakage
        # — the decision of which stays to drop is influenced by val/test distribution.
        # Use filter_ids_by_obs() from dataset.py on train_ids only, after splitting.
        out[sid] = (seq, tau, mask)

    print(f"  Sequences   : {len(out):,} stays ({seq_len}×{N_FEAT})")
    return out


# ---------------------------------------------------------------------------
# Stage 7 — static features
# ---------------------------------------------------------------------------
def build_static_features(icu, patients, admissions, diagnoses) -> Dict:
    elix: Dict[int, List[float]] = {}
    if not diagnoses.is_empty() and "icd9_code" in diagnoses.columns:
        for (hid,), g in diagnoses.group_by(["hadm_id"]):
            codes = {c.replace(".", "").strip() for c in g["icd9_code"].drop_nulls().to_list()}
            elix[int(hid)] = [float(bool(codes & set(v))) for v in _ELIX.values()]

    age_sex: Dict[int, Tuple] = {}
    if not patients.is_empty():
        ac = "anchor_age" if "anchor_age" in patients.columns else None
        gc = "gender"     if "gender"     in patients.columns else None
        for r in patients.iter_rows(named=True):
            sid = int(r["subject_id"])
            age_sex[sid] = (
                float(r[ac]) if ac else 0.0,
                1.0 if str(r.get(gc, "")).upper() == "M" else 0.0,
            )

    adm_info: Dict[int, Dict] = {}
    if not admissions.is_empty():
        for r in admissions.iter_rows(named=True):
            adm_info[int(r["hadm_id"])] = {
                "t": str(r.get("admission_type") or "").upper(),
                "i": str(r.get("insurance")      or "").upper(),
            }

    out = {}
    for r in icu.iter_rows(named=True):
        sid  = int(r["subject_id"])
        hid  = int(r["hadm_id"])
        icid = int(r["icustay_id"])
        unit = str(r.get("first_careunit") or "").upper()
        age, gm = age_sex.get(sid, (0.0, 0.0))
        ai = adm_info.get(hid, {})
        t, ins = ai.get("t", ""), ai.get("i", "")
        f = [age, gm,
             float("EMERGENCY" in t), float("ELECTIVE" in t),
             float("MEDICARE"  in ins), float("MEDICAID" in ins), float("PRIVATE" in ins),
             float("MICU" in unit), float("SICU" in unit),
             float("CICU" in unit or "CCU" in unit),
             float("NICU" in unit), float("CSRU" in unit)]
        f.extend(elix.get(hid, [0.0] * 30))
        out[icid] = np.array(f, dtype=np.float32)
    print(f"  Static      : {len(out):,} vectors × {len(STATIC_NAMES)} features")
    return out


# ---------------------------------------------------------------------------
# Stage 8 — labels
# ---------------------------------------------------------------------------
def build_labels(icu, admissions) -> Tuple[Dict, Dict]:
    expire: Dict[int, int] = {}
    if not admissions.is_empty() and "hospital_expire_flag" in admissions.columns:
        for r in admissions.iter_rows(named=True):
            expire[int(r["hadm_id"])] = int(r.get("hospital_expire_flag") or 0)
    mort, los = {}, {}
    has_los = "los" in icu.columns
    for r in icu.iter_rows(named=True):
        icid = int(r["icustay_id"])
        mort[icid] = expire.get(int(r["hadm_id"]), 0)
        los[icid]  = float(r["los"]) if has_los and r["los"] is not None else 0.0
    n_pos = sum(mort.values())
    n     = len(mort)
    print(f"  Labels      : {n:,} stays | "
          f"mortality {n_pos:,} ({100*n_pos/max(n,1):.1f}%) | "
          f"mean LOS {np.mean(list(los.values())):.1f}d")
    return mort, los


# ---------------------------------------------------------------------------
# run_etl — full pipeline
# ---------------------------------------------------------------------------
def run_etl(
    mimic_dir: str,
    seq_len:   int  = 48,
    min_age:   int  = 18,
    save_dir:  Optional[str] = None,
) -> Dict:
    """Execute all 8 stages and return the ETL output dict."""
    from src.data.loader import (load_chartevents, load_labevents, load_icustays,
                                  load_admissions, load_patients, load_diagnoses)
    print("=" * 55)
    print("[ETL] Loading MIMIC-IV CSV.gz tables …")
    chart  = load_chartevents(mimic_dir)
    labs   = load_labevents(mimic_dir)
    icu    = load_icustays(mimic_dir)
    adm    = load_admissions(mimic_dir)
    pat    = load_patients(mimic_dir)
    diag   = load_diagnoses(mimic_dir)
    print("[ETL] Running pipeline …")
    pat    = infer_age(pat, adm)
    icu    = filter_by_age(icu, pat, min_age)
    labs   = align_labs(labs, icu)
        # Defensive: ensure loader outputs are DataFrames (not None)
    if chart is None:
        chart = pl.DataFrame()
    if labs is None:
        labs = pl.DataFrame()
    if icu is None:
        icu = pl.DataFrame()

    events = merge_events(chart, labs, icu)
    events = compute_delta_t(events)
    seqs   = build_sequences(events, seq_len)
    static = build_static_features(icu, pat, adm, diag)
    y_mort, y_los = build_labels(icu, adm)
    n = len(set(seqs) & set(y_mort))
    print(f"[ETL] Complete — {n:,} labelled stays ready.")
    print("=" * 55)

    out = dict(sequences=seqs, static_features=static,
               mortality_labels=y_mort, los_labels=y_los,
               feature_names=FEATURE_NAMES, static_names=STATIC_NAMES)
    if save_dir:
        sp = Path(save_dir)
        sp.mkdir(parents=True, exist_ok=True)
        with (sp / "etl_output.pkl").open("wb") as f:
            pickle.dump(out, f, protocol=5)
        print(f"[ETL] Saved → {sp}/etl_output.pkl")
    return out


def load_etl_output(path: str) -> Dict:
    with open(path, "rb") as f:
        d = pickle.load(f)
    print(f"[ETL] Loaded {len(d['sequences']):,} stays from {path}")
    return d