"""
src/data/loader.py
==================
Stream MIMIC-III and MIMIC-IV CSV.gz tables into Polars DataFrames.

Backward compatible with the original MIMIC-III loader structure.
Added: path fallback (tries lowercase then uppercase filenames) and
column auto-detect (MIMIC-IV uses stay_id/lowercase, MIMIC-III uses
ICUSTAY_ID/uppercase). All outputs normalised to lowercase column
names so preprocess.py works unchanged for both versions.

Type casting at load time:
  charttime  → pl.Datetime   (chartevents + labevents)
  intime     → pl.Datetime   (icustays)
  outtime    → pl.Datetime   (icustays)
  icustay_id → pl.Int64
  hadm_id    → pl.Int64
  itemid     → pl.Int64
"""
from __future__ import annotations
from pathlib import Path
from typing import Union

import polars as pl

_DT_FMT = "%Y-%m-%d %H:%M:%S"


def _resolve(mimic_dir: Path, *names: str) -> Path:
    """Try each candidate filename; return first that exists."""
    for name in names:
        p = mimic_dir / name
        if p.exists():
            return p
    return mimic_dir / names[0]   # return first for error message


def _stream(path: Path, **kw) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.scan_csv(path, null_values=["", "NA"], **kw).collect(engine="streaming")


def _lower(df: pl.DataFrame) -> pl.DataFrame:
    return df.rename({c: c.lower() for c in df.columns}) if not df.is_empty() else df


def load_chartevents(mimic_dir: Union[str, Path]) -> pl.DataFrame:
    """
    Stream CHARTEVENTS — handles both MIMIC-III (uppercase) and MIMIC-IV (lowercase).
    Output: [icustay_id, charttime, itemid, valuenum] normalised to lowercase.
    """
    path = _resolve(
        Path(mimic_dir),
        "chartevents.csv.gz",   # MIMIC-IV
        "CHARTEVENTS.csv.gz",   # MIMIC-III
    )
    if not path.exists():
        print("  chartevents : not found (skipping)")
        return pl.DataFrame()

    # Peek to detect actual column names (case-sensitive)
    peek    = pl.read_csv(path, n_rows=0)
    cols    = peek.columns
    cols_lc = {c.lower() for c in cols}

    # Identify stay column: MIMIC-IV=stay_id, MIMIC-III=ICUSTAY_ID
    if "icustay_id" in cols_lc:
        stay_col = next(c for c in cols if c.lower() == "icustay_id")
    elif "stay_id" in cols_lc:
        stay_col = next(c for c in cols if c.lower() == "stay_id")
    else:
        print("  chartevents : no stay_id / ICUSTAY_ID column (skipping)")
        return pl.DataFrame()

    # Identify required columns using actual casing
    ct_col  = next(c for c in cols if c.lower() == "charttime")
    itm_col = next(c for c in cols if c.lower() == "itemid")
    val_col = next(c for c in cols if c.lower() == "valuenum")

    # Error/warning flag column (optional)
    err_col = None
    for candidate in ("ERROR", "error", "WARNING", "warning"):
        if candidate in cols:
            err_col = candidate
            break

    lf = pl.scan_csv(
        path,
        schema_overrides={
            "VALUE": pl.String,   "value": pl.String,
            "VALUENUM": pl.Float64, "valuenum": pl.Float64,
        },
        null_values=["", "NA"],
    )

    select_cols = [stay_col, ct_col, itm_col, val_col]
    if err_col:
        select_cols.append(err_col)
    lf = lf.select(select_cols)

    if err_col:
        lf = lf.filter(pl.col(err_col).is_null() | (pl.col(err_col) == 0))
        lf = lf.drop(err_col)

    df = (
        lf
        .filter(pl.col(val_col).is_not_null())
        .with_columns([
            pl.col(ct_col).str.strptime(pl.Datetime, _DT_FMT, strict=False).alias("charttime"),
            pl.col(stay_col).cast(pl.Int64, strict=False).alias("icustay_id"),
            pl.col(itm_col).cast(pl.Int64, strict=False).alias("itemid"),
            pl.col(val_col).cast(pl.Float64).alias("valuenum"),
        ])
        .select(["icustay_id", "charttime", "itemid", "valuenum"])
        .collect(engine="streaming")
    )
    print(f"  chartevents : {df.height:>12,} rows")
    return df


def load_labevents(mimic_dir: Union[str, Path]) -> pl.DataFrame:
    """
    Stream LABEVENTS — handles both MIMIC-III and MIMIC-IV.
    Output: [hadm_id, itemid, charttime, valuenum] normalised to lowercase.
    """
    path = _resolve(
        Path(mimic_dir),
        "labevents.csv.gz",   # MIMIC-IV
        "LABEVENTS.csv.gz",   # MIMIC-III
    )
    if not path.exists():
        print("  labevents   : not found (skipping)")
        return pl.DataFrame()

    peek    = pl.read_csv(path, n_rows=0)
    cols    = peek.columns
    cols_lc = {c.lower() for c in cols}

    hadm_col = next((c for c in cols if c.lower() == "hadm_id"),   None)
    itm_col  = next((c for c in cols if c.lower() == "itemid"),    None)
    ct_col   = next((c for c in cols if c.lower() == "charttime"), None)
    val_col  = next((c for c in cols if c.lower() == "valuenum"),  None)

    if not all([hadm_col, itm_col, ct_col, val_col]):
        print(f"  labevents   : missing required columns {[hadm_col,itm_col,ct_col,val_col]} (skipping)")
        return pl.DataFrame()

    df = (
        pl.scan_csv(
            path,
            schema_overrides={
                "VALUE": pl.String,   "value": pl.String,
                "VALUENUM": pl.Float64, "valuenum": pl.Float64,
            },
            null_values=["", "NA"],
        )
        .select([hadm_col, itm_col, ct_col, val_col])
        .filter(pl.col(val_col).is_not_null())
        .with_columns([
            pl.col(ct_col).str.strptime(pl.Datetime, _DT_FMT, strict=False).alias("charttime"),
            pl.col(hadm_col).cast(pl.Int64, strict=False).alias("hadm_id"),
            pl.col(itm_col).cast(pl.Int64, strict=False).alias("itemid"),
            pl.col(val_col).cast(pl.Float64).alias("valuenum"),
        ])
        .select(["hadm_id", "itemid", "charttime", "valuenum"])
        .collect(engine="streaming")
    )
    print(f"  labevents   : {df.height:>12,} rows")
    return df


def load_icustays(mimic_dir: Union[str, Path]) -> pl.DataFrame:
    """
    Load ICUSTAYS — handles both MIMIC-III (ICUSTAY_ID) and MIMIC-IV (stay_id).
    Output: normalised to icustay_id, intime/outtime as Datetime.
    """
    path = _resolve(
        Path(mimic_dir),
        "icustays.csv.gz",   # MIMIC-IV
        "ICUSTAYS.csv.gz",   # MIMIC-III
    )
    df = _stream(path, schema_overrides={
        "INTIME": pl.String, "OUTTIME": pl.String,
        "intime": pl.String, "outtime": pl.String,
    })
    if df.is_empty():
        return df

    df = _lower(df)

    # MIMIC-IV: rename stay_id → icustay_id
    if "stay_id" in df.columns and "icustay_id" not in df.columns:
        df = df.rename({"stay_id": "icustay_id"})

    # Parse intime/outtime if still string
    for col in ("intime", "outtime"):
        if col in df.columns and df.schema[col] in (pl.Utf8, pl.String):
            df = df.with_columns(
                pl.col(col).str.to_datetime(format=_DT_FMT, strict=False)
            )

    for col in ("icustay_id", "hadm_id", "subject_id"):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))

    return df


def load_admissions(mimic_dir: Union[str, Path]) -> pl.DataFrame:
    path = _resolve(
        Path(mimic_dir),
        "admissions.csv.gz",   # MIMIC-IV
        "ADMISSIONS.csv.gz",   # MIMIC-III
    )
    df = _stream(path, schema_overrides={
        "ADMITTIME": pl.String, "DISCHTIME": pl.String,
        "admittime": pl.String, "dischtime": pl.String,
    })
    return _lower(df)


def load_patients(mimic_dir: Union[str, Path]) -> pl.DataFrame:
    path = _resolve(
        Path(mimic_dir),
        "patients.csv.gz",   # MIMIC-IV
        "PATIENTS.csv.gz",   # MIMIC-III
    )
    df = _stream(path, schema_overrides={"DOB": pl.String, "dob": pl.String})
    return _lower(df)


def load_diagnoses(mimic_dir: Union[str, Path]) -> pl.DataFrame:
    path = _resolve(
        Path(mimic_dir),
        "diagnoses_icd.csv.gz",   # MIMIC-IV
        "DIAGNOSES_ICD.csv.gz",   # MIMIC-III
    )
    df = _stream(path)
    df = _lower(df)
    # MIMIC-IV: icd_code → icd9_code (preprocess.py expects icd9_code)
    if "icd_code" in df.columns and "icd9_code" not in df.columns:
        df = df.rename({"icd_code": "icd9_code"})
    return df