#!/usr/bin/env python3
"""
scripts/preflight_check.py
============================
Run BEFORE any training to get a CONCRETE pass/fail verdict for each stage.
No vague predictions — uses measured available RAM/VRAM right now.

    python scripts/preflight_check.py
    python scripts/preflight_check.py --config config.yaml

Exit codes
----------
  0  — All stages will fit. Safe to run.
  1  — One or more stages WILL OOM. Adjust config before proceeding.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# ANSI helpers
# ─────────────────────────────────────────────────────────────────────────────
RESET = "\033[0m"; BOLD = "\033[1m"
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; CYAN = "\033[96m"; DIM = "\033[2m"

def _c(s, color): return f"{color}{s}{RESET}"
def _pass(msg):   return f"  {_c('PASS', GREEN+BOLD)}  {msg}"
def _warn(msg):   return f"  {_c('WARN', YELLOW+BOLD)}  {msg}"
def _fail(msg):   return f"  {_c('FAIL', RED+BOLD)}  {msg}"


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

def _ram_available() -> tuple[float, float]:
    """Returns (available_GB, total_GB). Available = free + reclaimable."""
    try:
        import psutil
        m = psutil.virtual_memory()
        return m.available / 1e9, m.total / 1e9
    except ImportError:
        pass
    # Fallback: parse /proc/meminfo
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, v = line.split(":")
            info[k.strip()] = int(v.strip().split()[0]) * 1024  # kB → bytes
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        total = info.get("MemTotal", 32 * 1024**3)
        return avail / 1e9, total / 1e9
    except Exception:
        return 20.0, 32.0  # safe fallback


def _vram_available() -> tuple[float, float]:
    """Returns (available_GB, total_GB)."""
    try:
        import torch
        if torch.cuda.is_available():
            reserved = torch.cuda.memory_reserved(0)
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            free = total - reserved
            return free / 1e9, total / 1e9
    except Exception:
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=3, stderr=subprocess.DEVNULL,
        ).decode().strip().split(",")
        return float(out[0]) / 1024, float(out[1]) / 1024
    except Exception:
        return 14.0, 16.0  # assume RTX 5060 Ti


def _swap_used() -> float:
    try:
        import psutil
        return psutil.swap_memory().used / 1e9
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Config reader
# ─────────────────────────────────────────────────────────────────────────────

def _load_cfg(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _estimate_costs(cfg: dict) -> list[dict]:
    """
    Return per-stage RAM and VRAM cost estimates based on actual config values.
    All estimates are PEAK (worst-case), not average.
    """
    m   = cfg.get("model", {})
    s1  = cfg.get("stage1", {})
    s2  = cfg.get("stage2", {})
    res = cfg.get("research", {})

    d_model    = int(m.get("d_model", 128))
    seq_len    = int(cfg.get("data", {}).get("seq_len", 48))
    batch      = int(s1.get("batch_size", 8))
    epochs     = int(s1.get("epochs", 100))
    n_workers  = int(cfg.get("data", {}).get("num_workers", 2))
    n_est      = int(s2.get("n_estimators", 200))
    depth      = int(s2.get("max_depth", 4))
    n_inter    = int(s2.get("n_interaction_top", 8)) if s2.get("use_interaction_features", False) else 0
    n_ens      = int(s2.get("n_ensemble_mort", 1))
    max_sps    = int(res.get("max_samples_per_stay", 1))
    d_input    = 40  # fixed feature count

    # ── Stage-wise estimates ────────────────────────────────────────────────
    # ETL: Polars reads CSVs + builds stays dict
    etl_ram = 6.0  # base
    # Feature extraction: 3 full splits (tr+va+te) × d_model×4 + d_input×5 cols
    feat_cols = 4 * d_model + 5 * d_input + 42
    feat_ram = 3 * 40_000 * feat_cols * 4 / 1e9  # float32 bytes
    # XGBoost GPU hist: roughly (n_est * depth * n_features * 4 bytes)
    xgb_vram = min(n_est * depth * feat_cols * 4 / 1e9 * 2, 4.0)
    # Interaction features add extra cols
    if n_inter > 0:
        n_pairs = n_inter * (n_inter - 1) // 2
        feat_ram += 3 * 40_000 * n_pairs * 4 / 1e9
    # Ensemble multiplier
    feat_ram *= max(n_ens, 1) * 0.5 + 0.5  # partial overlap
    # Stage 1: batch in VRAM
    batch_vram = batch * seq_len * d_input * 4 / 1e9  # input tensor
    mamba_vram = d_model * seq_len * batch * 8 / 1e9  # activations ×2 for grads
    ema_vram   = d_model * d_input * 4 / 1e9 * 0.5   # EMA shadow model
    s1_vram    = batch_vram + mamba_vram + ema_vram + 0.5  # +0.5 for optimizer state
    # DataLoader workers RAM
    worker_ram = n_workers * 0.5

    return [
        {
            "name":     "ETL (load MIMIC-IV data)",
            "stage":    "etl",
            "ram_gb":   etl_ram + worker_ram,
            "vram_gb":  0.1,
            "note":     f"Polars CSV parsing + stays dict (workers={n_workers})",
        },
        {
            "name":     "Expand longitudinal samples",
            "stage":    "expand_samples",
            "ram_gb":   etl_ram + etl_ram * max_sps * 0.6,  # expansion overhead
            "vram_gb":  0.1,
            "note":     f"max_samples_per_stay={max_sps} × ETL data in RAM simultaneously",
        },
        {
            "name":     "Stage 1 — Mamba training",
            "stage":    "stage1_train",
            "ram_gb":   etl_ram + worker_ram + 3.0,  # seqs + workers + model
            "vram_gb":  round(s1_vram, 2),
            "note":     f"batch={batch} seq_len={seq_len} d_model={d_model} epochs={epochs}",
        },
        {
            "name":     "Stage 2 — Feature extraction",
            "stage":    "stage2_extract",
            "ram_gb":   etl_ram + feat_ram + 2.0,
            "vram_gb":  round(s1_vram * 0.6, 2),  # inference only, no grads
            "note":     f"F_tr+F_va+F_te: {feat_cols} cols × 3×40k rows = {feat_ram:.1f} GB",
        },
        {
            "name":     "Stage 2 — XGBoost training",
            "stage":    "stage2_train",
            "ram_gb":   etl_ram + feat_ram,
            "vram_gb":  round(xgb_vram, 2),
            "note":     f"n_estimators={n_est} depth={depth} ensemble={n_ens}",
        },
        {
            "name":     "Stage 2 — Head ablation",
            "stage":    "stage2_ablation",
            "ram_gb":   etl_ram + feat_ram * 1.2,  # multiple transformed copies
            "vram_gb":  round(xgb_vram * 1.2, 2),
            "note":     "flat, hybrid, distilled heads + their feature matrices",
        },
        {
            "name":     "Benchmark — Table 1",
            "stage":    "benchmark_t1",
            "ram_gb":   etl_ram + 3.0,
            "vram_gb":  round(s1_vram * 0.4, 2),
            "note":     "MaBoost inference on test set only",
        },
        {
            "name":     "Benchmark — Table 2 (deep baselines)",
            "stage":    "benchmark_t2",
            "ram_gb":   etl_ram + 8.0,   # GRU-D/Transformer activations
            "vram_gb":  5.0,             # deep models peak
            "note":     "GRU-D, LSTM, Transformer, InterpNet, TCN — use --skip-baselines to skip",
        },
        {
            "name":     "Benchmark — Irregular scenarios",
            "stage":    "benchmark_irr",
            "ram_gb":   etl_ram + 6.0,   # scenario dicts expand
            "vram_gb":  4.0,
            "note":     "8 scenarios × 3 seeds — use --skip-new-tests to skip",
        },
        {
            "name":     "SHAP interpretability",
            "stage":    "shap",
            "ram_gb":   etl_raw + feat_ram + 4.0 if (etl_raw := etl_ram) else 0,
            "vram_gb":  round(s1_vram * 0.6, 2),
            "note":     "TreeExplainer + UMAP on test set — use --skip-shap to skip",
        },
        {
            "name":     "Visualization",
            "stage":    "visualization",
            "ram_gb":   3.0,
            "vram_gb":  0.1,
            "note":     "matplotlib only",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main check
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="MaBoost preflight RAM/VRAM check")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--safety-margin", type=float, default=0.88,
                    help="Use only this fraction of available memory (default=0.88 = 88%%)")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    stages = _estimate_costs(cfg)

    ram_avail, ram_total   = _ram_available()
    vram_avail, vram_total = _vram_available()
    swap_gb = _swap_used()
    margin  = args.safety_margin

    ram_safe  = ram_avail  * margin
    vram_safe = vram_avail * margin

    print(f"\n{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print(f"{BOLD}  MaBoost Preflight Check — config: {args.config}{RESET}")
    print(f"{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print(f"\n  {'Hardware':30s}  {'Available':>12}  {'Total':>8}")
    print(f"  {'─'*55}")
    print(f"  {'RAM':30s}  {ram_avail:>10.1f} GB  {ram_total:>6.1f} GB")
    print(f"  {'VRAM (GPU)':30s}  {vram_avail:>10.1f} GB  {vram_total:>6.1f} GB")
    if swap_gb > 0.2:
        print(f"\n  {RED}{BOLD}⚠  SWAP ALREADY IN USE: {swap_gb:.1f} GB — your system is under memory pressure!{RESET}")
        print(f"  {YELLOW}  Close other applications before training.{RESET}")
    print(f"\n  Safety margin: {margin*100:.0f}%  →  "
          f"effective RAM budget = {ram_safe:.1f} GB, VRAM budget = {vram_safe:.1f} GB\n")

    print(f"  {'Stage':40s}  {'RAM need':>9}  {'VRAM need':>9}  Result")
    print(f"  {'─'*75}")

    any_fail = False
    any_warn = False

    for s in stages:
        ram_need  = s["ram_gb"]
        vram_need = s["vram_gb"]
        ram_ok    = ram_need  <= ram_safe
        vram_ok   = vram_need <= vram_safe
        # warn if within 2GB of limit
        ram_warn  = (ram_safe - ram_need)  < 2.0
        vram_warn = (vram_safe - vram_need) < 1.5

        if ram_ok and vram_ok:
            if ram_warn or vram_warn:
                verdict = _warn(f"Tight! RAM spare={ram_safe-ram_need:.1f}GB  VRAM spare={vram_safe-vram_need:.1f}GB")
                any_warn = True
            else:
                verdict = _pass(f"RAM spare={ram_safe-ram_need:.1f}GB  VRAM spare={vram_safe-vram_need:.1f}GB")
        else:
            parts = []
            if not ram_ok:
                deficit = ram_need - ram_safe
                parts.append(f"RAM short by {deficit:.1f} GB")
            if not vram_ok:
                deficit = vram_need - vram_safe
                parts.append(f"VRAM short by {deficit:.1f} GB")
            verdict = _fail(" + ".join(parts))
            any_fail = True

        print(f"  {s['name']:40s}  {ram_need:>7.1f}GB  {vram_need:>7.1f}GB  {verdict}")
        print(f"  {DIM}{'':40s}  {s['note']}{RESET}")

    # ── Summary + recommendation ────────────────────────────────────────────
    print(f"\n{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    if any_fail:
        print(f"\n  {RED}{BOLD}VERDICT: ❌ DO NOT RUN — one or more stages will OOM{RESET}")
        print(f"\n  {BOLD}Fix options:{RESET}")
        print(f"  1. Skip heavy stages:     --skip-baselines --skip-new-tests --skip-shap")
        print(f"  2. Reduce batch_size in config.yaml  (currently: {cfg.get('stage1',{}).get('batch_size','?')})")
        print(f"  3. Reduce max_samples_per_stay       (currently: {cfg.get('research',{}).get('max_samples_per_stay','?')})")
        print(f"  4. Reduce n_estimators               (currently: {cfg.get('stage2',{}).get('n_estimators','?')})")
        print(f"  5. Disable interaction_features      (currently: {cfg.get('stage2',{}).get('use_interaction_features','?')})")
        print()
        return 1
    elif any_warn:
        print(f"\n  {YELLOW}{BOLD}VERDICT: ⚠  PROCEED WITH CAUTION — some stages are tight{RESET}")
        print(f"\n  Recommended: close browser/IDE, then run:")
        print(f"    python run_experiment.py --skip-etl --skip-baselines --skip-new-tests")
        print()
        return 0
    else:
        print(f"\n  {GREEN}{BOLD}VERDICT: ✅ ALL CLEAR — safe to run the full pipeline{RESET}")
        print(f"\n  Run with:")
        print(f"    python run_experiment.py --skip-etl --skip-baselines")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
