#!/usr/bin/env python3
"""
scripts/monitor_resources.py
=============================
Real-time RAM + VRAM monitor for the ENTIRE MaBoost product pipeline.

Tracks end-to-end:
  ─ Training pipeline   (ETL → expand → Stage 1 → Stage 2 → benchmark → SHAP)
  ─ Benchmark pipeline  (setup → per-seed → per-scenario → per-model → aggregate)
  ─ Data audit          (audit_dataset.py 4-proof validation)
  ─ Norm-fix pipeline   (norm_stats persistence + injection verification)

Run in a SEPARATE terminal alongside your training/benchmark command:

    Terminal 1:  python run_experiment.py --skip-etl
    Terminal 2:  python scripts/monitor_resources.py

    Terminal 1:  python scripts/run_hospital_irregular_benchmark.py --skip-etl
    Terminal 2:  python scripts/monitor_resources.py

What it shows
-------------
  • Exact RAM used / total (GB) with color bar + trend arrow
  • Exact VRAM used / total (GB) with color bar
  • SWAP usage warning
  • Current pipeline stage (from results/pipeline_progress.json)
  • Per-stage elapsed time + stage history timeline
  • CRASH PREDICTION for the NEXT step
  • Benchmark progress: seed/scenario/model counts
  • Summary statistics: total runtime, stages completed, peak RAM/VRAM

Color legend
------------
  GREEN  < 70% — safe
  YELLOW  70-85% — approaching limit
  RED    > 85% — danger, likely to OOM soon

Usage
-----
    python scripts/monitor_resources.py [--interval 2] [--results results/]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ─────────────────────────────────────────────────────────────────────────────
# Known memory costs per stage — calibrated for RTX 5060 Ti 16GB / 32GB RAM
# ─────────────────────────────────────────────────────────────────────────────

# ── Training pipeline stages (run_experiment.py) ──────────────────────────────
STAGE_VRAM_GB = {
    # Training pipeline
    "etl":              0.1,
    "expand_samples":   0.1,
    "stage1_train":     4.5,
    "stage1_eval":      1.5,
    "stage2_extract":   3.0,
    "stage2_train":     3.0,
    "stage2_ablation":  3.5,
    "benchmark_t1":     2.0,
    "benchmark_t2":     5.0,
    "benchmark_irr":    4.5,
    "shap":             4.0,
    "visualization":    0.5,
    # Benchmark pipeline
    "bench_setup":      0.5,
    "bench_etl":        0.1,
    "bench_encoder":    2.0,
    "bench_seed":       0.5,
    "bench_scenario":   4.0,
    "bench_model":      5.0,
    "bench_aggregate":  0.1,
    # Audit / normfix
    "audit":            1.0,
    "norm_persist":     0.1,
    "norm_verify":      1.0,
    # Terminal
    "done":             0.0,
    "crashed":          0.0,
    "waiting":          0.0,
}

STAGE_RAM_GB = {
    # Training pipeline
    "etl":              8.0,
    "expand_samples":   12.0,
    "stage1_train":     10.0,
    "stage1_eval":      8.0,
    "stage2_extract":   14.0,
    "stage2_train":     8.0,
    "stage2_ablation":  12.0,
    "benchmark_t1":     8.0,
    "benchmark_t2":     18.0,
    "benchmark_irr":    16.0,
    "shap":             12.0,
    "visualization":    4.0,
    # Benchmark pipeline
    "bench_setup":      4.0,
    "bench_etl":        8.0,
    "bench_encoder":    10.0,
    "bench_seed":       10.0,
    "bench_scenario":   16.0,
    "bench_model":      18.0,
    "bench_aggregate":  4.0,
    # Audit / normfix
    "audit":            6.0,
    "norm_persist":     4.0,
    "norm_verify":      6.0,
    # Terminal
    "done":             0.0,
    "crashed":          0.0,
    "waiting":          0.0,
}

# ── Full stage labels ────────────────────────────────────────────────────────
STAGE_LABELS = {
    # Training pipeline
    "etl":              "1.  ETL — loading MIMIC-IV data",
    "expand_samples":   "1b. Expanding longitudinal samples",
    "stage1_train":     "2.  Stage 1 — Training Mamba encoder",
    "stage1_eval":      "2b. Stage 1 — Evaluating encoder",
    "stage2_extract":   "3a. Stage 2 — Extracting features",
    "stage2_train":     "3b. Stage 2 — Training XGBoost heads",
    "stage2_ablation":  "3c. Stage 2 — Head ablation",
    "benchmark_t1":     "4a. Benchmark — Table 1 (MaBoost vs ablations)",
    "benchmark_t2":     "4b. Benchmark — Table 2 (deep baselines)",
    "benchmark_irr":    "4c. Benchmark — Irregular scenarios",
    "shap":             "5.  SHAP interpretability",
    "visualization":    "6.  Visualization & plots",
    # Benchmark pipeline
    "bench_setup":      "B0. Benchmark — Setup & config",
    "bench_etl":        "B1. Benchmark — Loading ETL data",
    "bench_encoder":    "B2. Benchmark — Loading frozen encoder",
    "bench_seed":       "B3. Benchmark — Seed loop",
    "bench_scenario":   "B4. Benchmark — Scenario",
    "bench_model":      "B5. Benchmark — Model training/eval",
    "bench_aggregate":  "B6. Benchmark — Aggregation & report",
    # Audit / normfix
    "audit":            "A1. Audit — 4-proof dataset validation",
    "norm_persist":     "N1. Norm fix — Persisting global stats",
    "norm_verify":      "N2. Norm fix — Verification",
    # Terminal
    "done":             "✅ DONE",
    "crashed":          "💥 CRASHED",
    "waiting":          "⏳ Waiting for pipeline…",
}

# Stage ordering for the training pipeline
TRAIN_PIPELINE = [
    "etl", "expand_samples",
    "stage1_train", "stage1_eval",
    "stage2_extract", "stage2_train", "stage2_ablation",
    "benchmark_t1", "benchmark_t2", "benchmark_irr",
    "shap", "visualization", "done",
]

# Stage ordering for the benchmark pipeline
BENCH_PIPELINE = [
    "bench_setup", "bench_etl", "bench_encoder",
    "bench_seed", "bench_scenario", "bench_model",
    "bench_aggregate", "done",
]

# Combined ordering for unknown pipelines
ALL_STAGES = list(dict.fromkeys(TRAIN_PIPELINE + BENCH_PIPELINE + [
    "audit", "norm_persist", "norm_verify",
]))

# ─────────────────────────────────────────────────────────────────────────────
# ANSI color helpers
# ─────────────────────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
DIM    = "\033[2m"
MAGENTA = "\033[95m"
BLUE   = "\033[94m"


def _color(pct: float) -> str:
    if pct < 0.70: return GREEN
    if pct < 0.85: return YELLOW
    return RED


def _bar(used: float, total: float, width: int = 30) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    pct = min(used / total, 1.0)
    filled = int(pct * width)
    c = _color(pct)
    bar = c + "█" * filled + DIM + "░" * (width - filled) + RESET
    return f"[{bar}]"


def _trend_arrow(history: deque, threshold: float = 0.5) -> str:
    """Return ↑ ↓ → based on recent RAM history trend."""
    if len(history) < 3:
        return " "
    recent = list(history)[-5:]
    delta = recent[-1] - recent[0]
    if delta > threshold:
        return f"{RED}↑{RESET}"
    elif delta < -threshold:
        return f"{GREEN}↓{RESET}"
    return f"{DIM}→{RESET}"


def _verdict(used: float, total: float, next_cost: float) -> str:
    """Return explicit crash prediction for the NEXT stage."""
    if total <= 0:
        return f"{DIM}unknown{RESET}"
    headroom = total - used
    margin = headroom - next_cost
    pct_after = (used + next_cost) / total
    if margin > 2.0:
        return f"{GREEN}{BOLD}SAFE{RESET} ({margin:+.1f} GB headroom)"
    if margin > 0:
        return f"{YELLOW}{BOLD}WARNING{RESET} (only {margin:.1f} GB spare)"
    return f"{RED}{BOLD}⚠ WILL OOM{RESET} (need {next_cost:.1f} GB, have {headroom:.1f} GB free)"


def _fmt_duration(seconds: float) -> str:
    """Format duration nicely."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h{m:02d}m"


# ─────────────────────────────────────────────────────────────────────────────
# Hardware queries
# ─────────────────────────────────────────────────────────────────────────────

def _ram() -> tuple[float, float]:
    """Returns (used_GB, total_GB) for system RAM."""
    if not HAS_PSUTIL:
        return 0.0, 32.0
    m = psutil.virtual_memory()
    return m.used / 1e9, m.total / 1e9


def _vram() -> tuple[float, float]:
    """Returns (used_GB, total_GB) for GPU VRAM via nvidia-smi."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        ).decode().strip().split(",")
        used_mb, total_mb = float(out[0]), float(out[1])
        return used_mb / 1024, total_mb / 1024
    except Exception:
        return 0.0, 16.0


def _swap() -> tuple[float, float]:
    if not HAS_PSUTIL:
        return 0.0, 0.0
    s = psutil.swap_memory()
    return s.used / 1e9, s.total / 1e9


def _gpu_temp() -> Optional[float]:
    """Returns GPU temperature in Celsius, or None if unavailable."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(out)
    except Exception:
        return None


def _gpu_util() -> Optional[float]:
    """Returns GPU utilization percentage, or None."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            timeout=2, stderr=subprocess.DEVNULL,
        ).decode().strip()
        return float(out)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Progress file reader
# ─────────────────────────────────────────────────────────────────────────────

def _read_progress(progress_file: Path) -> dict:
    if not progress_file.exists():
        return {"stage": "waiting", "label": "Waiting for pipeline to start…",
                "ts": None, "epoch": None, "epoch_total": None,
                "auroc": None, "message": None}
    try:
        return json.loads(progress_file.read_text())
    except Exception:
        return {"stage": "waiting", "label": "Reading progress…",
                "ts": None, "epoch": None, "epoch_total": None,
                "auroc": None, "message": None}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline detection + next stage logic
# ─────────────────────────────────────────────────────────────────────────────

def _detect_pipeline(stage: str) -> str:
    """Detect which pipeline is running based on stage prefix."""
    if stage.startswith("bench"):
        return "benchmark"
    if stage in ("audit", "norm_persist", "norm_verify"):
        return "normfix"
    if stage in TRAIN_PIPELINE:
        return "training"
    return "unknown"


def _get_pipeline_stages(pipeline: str) -> list[str]:
    if pipeline == "training":
        return TRAIN_PIPELINE
    elif pipeline == "benchmark":
        return BENCH_PIPELINE
    elif pipeline == "normfix":
        return ["norm_persist", "audit", "norm_verify", "done"]
    return ALL_STAGES


def _next_stage(current: str, pipeline: str) -> str | None:
    stages = _get_pipeline_stages(pipeline)
    try:
        idx = stages.index(current)
        if idx + 1 < len(stages):
            return stages[idx + 1]
    except ValueError:
        pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage history tracker
# ─────────────────────────────────────────────────────────────────────────────

class StageTracker:
    """Tracks stage transitions, elapsed times, and peak resource usage."""

    def __init__(self):
        self.history: list[dict] = []     # [{stage, start, end, duration_s}]
        self.current_stage: str = "waiting"
        self.current_start: Optional[datetime] = None
        self.peak_ram: float = 0.0
        self.peak_vram: float = 0.0
        self.pipeline_start: Optional[datetime] = None
        self.ram_history: deque = deque(maxlen=30)   # last N measurements
        self.vram_history: deque = deque(maxlen=30)

        # Benchmark-specific counters
        self.bench_seeds_done: int = 0
        self.bench_seeds_total: int = 0
        self.bench_scenarios_done: int = 0
        self.bench_scenarios_total: int = 0
        self.bench_models_done: int = 0
        self.bench_current_seed: Optional[int] = None
        self.bench_current_scenario: str = ""
        self.bench_current_model: str = ""

    def update(self, progress: dict, ram_used: float, vram_used: float):
        stage = progress.get("stage", "waiting")
        message = progress.get("message", "") or ""
        ts_str = progress.get("ts", "")

        # Track peaks
        self.peak_ram = max(self.peak_ram, ram_used)
        self.peak_vram = max(self.peak_vram, vram_used)
        self.ram_history.append(ram_used)
        self.vram_history.append(vram_used)

        # Pipeline start
        if self.pipeline_start is None and stage not in ("waiting", "done", "crashed"):
            self.pipeline_start = datetime.now()

        # Detect stage transition
        if stage != self.current_stage:
            # Close previous stage
            if self.current_stage not in ("waiting",) and self.current_start is not None:
                self.history.append({
                    "stage": self.current_stage,
                    "start": self.current_start,
                    "end": datetime.now(),
                    "duration_s": (datetime.now() - self.current_start).total_seconds(),
                })

            self.current_stage = stage
            if ts_str:
                try:
                    self.current_start = datetime.fromisoformat(ts_str)
                except Exception:
                    self.current_start = datetime.now()
            else:
                self.current_start = datetime.now()

        # Parse benchmark-specific metadata from the message/label
        self._parse_benchmark_progress(progress)

    def _parse_benchmark_progress(self, progress: dict):
        """Extract seed/scenario/model counts from benchmark progress messages."""
        label = progress.get("label", "") or ""
        message = progress.get("message", "") or ""
        stage = progress.get("stage", "")

        # Parse "Hospital Benchmark: 7 models, 8 scenarios, 3 seeds"
        if "models" in label and "scenarios" in label and "seeds" in label:
            import re
            m = re.search(r"(\d+)\s+models.*?(\d+)\s+scenarios.*?(\d+)\s+seeds", label)
            if m:
                self.bench_scenarios_total = int(m.group(2))
                self.bench_seeds_total = int(m.group(3))

        # Parse "Seed 42 — Scenario: early_6h"
        if "Seed" in label and "Scenario:" in label:
            import re
            m = re.search(r"Seed\s+(\d+)\s.*Scenario:\s+(\S+)", label)
            if m:
                new_seed = int(m.group(1))
                new_scenario = m.group(2)
                if self.bench_current_seed != new_seed:
                    if self.bench_current_seed is not None:
                        self.bench_seeds_done += 1
                    self.bench_current_seed = new_seed
                    self.bench_scenarios_done = 0
                if new_scenario != self.bench_current_scenario:
                    if self.bench_current_scenario:
                        self.bench_scenarios_done += 1
                    self.bench_current_scenario = new_scenario

        # Count from message patterns
        if "AUROC=" in label or "ERROR" in label:
            self.bench_models_done += 1

    @property
    def total_elapsed(self) -> float:
        if self.pipeline_start is None:
            return 0.0
        return (datetime.now() - self.pipeline_start).total_seconds()

    @property
    def stages_completed(self) -> int:
        return len(self.history)


# ─────────────────────────────────────────────────────────────────────────────
# Main display
# ─────────────────────────────────────────────────────────────────────────────

def _render(progress: dict, tracker: StageTracker) -> None:
    ram_used, ram_total   = _ram()
    vram_used, vram_total = _vram()
    swap_used, swap_total = _swap()
    gpu_temp = _gpu_temp()
    gpu_util = _gpu_util()

    tracker.update(progress, ram_used, vram_used)

    stage   = progress.get("stage", "waiting")
    label   = progress.get("label", STAGE_LABELS.get(stage, stage))
    ts_str  = progress.get("ts", "")
    epoch   = progress.get("epoch")
    ep_tot  = progress.get("epoch_total")
    auroc   = progress.get("auroc")
    message = progress.get("message", "")

    pipeline = _detect_pipeline(stage)
    next_st  = _next_stage(stage, pipeline)
    next_vram = STAGE_VRAM_GB.get(next_st, 0.0) if next_st else 0.0
    next_ram  = STAGE_RAM_GB.get(next_st, 0.0) if next_st else 0.0

    # Elapsed time in current stage
    elapsed_str = ""
    if tracker.current_start:
        elapsed = (datetime.now() - tracker.current_start).total_seconds()
        elapsed_str = _fmt_duration(elapsed)

    # Clear screen
    os.system("clear")
    now = datetime.now().strftime("%H:%M:%S")

    # ── Header ──
    pipeline_badge = {
        "training": f"{CYAN}TRAINING{RESET}",
        "benchmark": f"{MAGENTA}BENCHMARK{RESET}",
        "normfix": f"{BLUE}NORM-FIX{RESET}",
        "unknown": f"{DIM}UNKNOWN{RESET}",
    }.get(pipeline, f"{DIM}?{RESET}")

    total_elapsed = _fmt_duration(tracker.total_elapsed) if tracker.total_elapsed > 0 else "—"
    
    print(f"{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
    print(f"{BOLD}{WHITE}  MaBoost Pipeline Monitor   {RESET}"
          f"{DIM}[{now}]{RESET}  "
          f"Pipeline: {pipeline_badge}  "
          f"Total: {BOLD}{total_elapsed}{RESET}")
    print(f"{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

    # ── Current stage ──
    stage_color = GREEN if stage == "done" else (RED if stage == "crashed" else CYAN)
    print(f"\n  {BOLD}Stage:{RESET}  {stage_color}{BOLD}{label}{RESET}  {DIM}({elapsed_str}){RESET}")

    # Epoch bar (for Stage 1 training)
    if epoch is not None and ep_tot is not None:
        ep_bar = _bar(epoch, ep_tot, width=20)
        auroc_str = f"  AUROC={auroc:.4f}" if auroc else ""
        print(f"  {BOLD}Epoch:{RESET}  {ep_bar} {epoch}/{ep_tot}{auroc_str}")

    # Extra message
    if message:
        # Truncate long messages
        msg = message if len(message) < 100 else message[:97] + "…"
        print(f"  {DIM}{msg}{RESET}")

    # ── Benchmark progress panel ──
    if pipeline == "benchmark" and tracker.bench_seeds_total > 0:
        print(f"\n  {BOLD}Benchmark Progress:{RESET}")
        seed_display = f"{tracker.bench_current_seed}" if tracker.bench_current_seed else "—"
        print(f"    Seeds:     {tracker.bench_seeds_done}/{tracker.bench_seeds_total}  "
              f"{DIM}(current: {seed_display}){RESET}")
        print(f"    Scenario:  {CYAN}{tracker.bench_current_scenario or '—'}{RESET}  "
              f"({tracker.bench_scenarios_done} done this seed)")
        print(f"    Models evaluated: {tracker.bench_models_done}")

    # ── RAM ──
    ram_pct = ram_used / max(ram_total, 1)
    ram_c   = _color(ram_pct)
    ram_trend = _trend_arrow(tracker.ram_history)
    print(f"\n  {BOLD}RAM:{RESET}   {_bar(ram_used, ram_total)}  "
          f"{ram_c}{ram_used:.1f}{RESET} / {ram_total:.1f} GB  "
          f"({ram_c}{ram_pct*100:.1f}%{RESET}) {ram_trend}  "
          f"{DIM}peak={tracker.peak_ram:.1f}GB{RESET}")

    # Swap warning
    if swap_used > 0.5:
        print(f"  {RED}{BOLD}  ⚠ SWAP ACTIVE: {swap_used:.1f}/{swap_total:.1f} GB — "
              f"system is already swapping! Crash risk HIGH.{RESET}")

    # ── VRAM ──
    vram_pct = vram_used / max(vram_total, 1)
    vram_c   = _color(vram_pct)
    vram_trend = _trend_arrow(tracker.vram_history, threshold=0.3)
    gpu_suffix = ""
    if gpu_temp is not None:
        temp_c = RED if gpu_temp > 85 else (YELLOW if gpu_temp > 75 else GREEN)
        gpu_suffix += f"  {temp_c}{gpu_temp:.0f}°C{RESET}"
    if gpu_util is not None:
        gpu_suffix += f"  {DIM}util={gpu_util:.0f}%{RESET}"
    print(f"  {BOLD}VRAM:{RESET}  {_bar(vram_used, vram_total)}  "
          f"{vram_c}{vram_used:.1f}{RESET} / {vram_total:.1f} GB  "
          f"({vram_c}{vram_pct*100:.1f}%{RESET}) {vram_trend}  "
          f"{DIM}peak={tracker.peak_vram:.1f}GB{RESET}{gpu_suffix}")

    # ── Next step prediction ──
    if next_st and stage not in ("done", "crashed", "waiting"):
        next_label = STAGE_LABELS.get(next_st, next_st)
        print(f"\n  {BOLD}Next step:{RESET}  {DIM}{next_label}{RESET}")
        print(f"    RAM  — needs ~{next_ram:.1f} GB  →  "
              + _verdict(ram_used, ram_total, next_ram))
        print(f"    VRAM — needs ~{next_vram:.1f} GB  →  "
              + _verdict(vram_used, vram_total, next_vram))
    elif stage == "done":
        print(f"\n  {GREEN}{BOLD}✅ Pipeline completed successfully!{RESET}")
        if tracker.total_elapsed > 0:
            print(f"  {DIM}Total runtime: {_fmt_duration(tracker.total_elapsed)}{RESET}")
    elif stage == "crashed":
        crash_msg = progress.get("message", "unknown error")
        print(f"\n  {RED}{BOLD}💥 CRASHED: {crash_msg}{RESET}")

    # ── Pipeline overview ──
    pipeline_stages = _get_pipeline_stages(pipeline)
    # Only show the overview for known pipelines
    if pipeline in ("training", "benchmark", "normfix"):
        print(f"\n  {BOLD}Pipeline overview ({pipeline}):{RESET}")
        for s in pipeline_stages:
            if s == "done":
                continue
            si = pipeline_stages.index(s) if s in pipeline_stages else 99
            ci = pipeline_stages.index(stage) if stage in pipeline_stages else -1
            if s == stage:
                icon = f"{CYAN}▶{RESET}"
            elif si < ci:
                icon = f"{GREEN}✓{RESET}"
            else:
                icon = f"{DIM}○{RESET}"

            lbl = STAGE_LABELS.get(s, s)
            vram_cost = STAGE_VRAM_GB.get(s, 0)
            ram_cost  = STAGE_RAM_GB.get(s, 0)
            suffix = f"{DIM}  RAM~{ram_cost:.0f}GB  VRAM~{vram_cost:.1f}GB{RESET}"
            txt_color = CYAN if s == stage else (DIM if si > ci else "")
            print(f"    {icon}  {txt_color}{lbl}{RESET}{suffix}")

            # Show elapsed time for completed stages
            for h in tracker.history:
                if h["stage"] == s:
                    print(f"       {DIM}└ completed in {_fmt_duration(h['duration_s'])}{RESET}")
                    break

    # ── Stage history (recent) ──
    if tracker.history:
        recent = tracker.history[-5:]  # show last 5 completed stages
        print(f"\n  {BOLD}Recent stage history:{RESET}")
        for h in recent:
            lbl = STAGE_LABELS.get(h["stage"], h["stage"])
            dur = _fmt_duration(h["duration_s"])
            print(f"    {GREEN}✓{RESET}  {lbl}  {DIM}({dur}){RESET}")

    # ── Summary stats ──
    if tracker.stages_completed > 0:
        print(f"\n  {BOLD}Summary:{RESET}")
        print(f"    Stages completed: {tracker.stages_completed}  "
              f"| Peak RAM: {tracker.peak_ram:.1f} GB  "
              f"| Peak VRAM: {tracker.peak_vram:.1f} GB")

    # ── Footer ──
    print(f"\n  {DIM}Refresh every 2s  |  progress: results/pipeline_progress.json  |  Ctrl+C to stop{RESET}")
    print(f"{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="MaBoost real-time resource monitor — tracks full end-to-end pipeline"
    )
    ap.add_argument("--interval", type=float, default=2.0,
                    help="Refresh interval in seconds (default=2)")
    ap.add_argument("--results", default="results",
                    help="Results directory containing pipeline_progress.json")
    args = ap.parse_args()

    if not HAS_PSUTIL:
        print("⚠  psutil not found. Install it: pip install psutil")
        print("   RAM monitoring will use fallback (32GB assumed).\n")

    progress_file = ROOT / args.results / "pipeline_progress.json"
    tracker = StageTracker()

    print(f"Monitoring pipeline…  (progress file: {progress_file})")
    print("Press Ctrl+C to exit.\n")
    time.sleep(1)

    try:
        while True:
            progress = _read_progress(progress_file)
            _render(progress, tracker)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        if tracker.stages_completed > 0:
            print(f"\nSession summary:")
            print(f"  Total runtime:      {_fmt_duration(tracker.total_elapsed)}")
            print(f"  Stages completed:   {tracker.stages_completed}")
            print(f"  Peak RAM:           {tracker.peak_ram:.1f} GB")
            print(f"  Peak VRAM:          {tracker.peak_vram:.1f} GB")
            print(f"\n  Stage timings:")
            for h in tracker.history:
                lbl = STAGE_LABELS.get(h["stage"], h["stage"])
                print(f"    {h['stage']:20s}  {_fmt_duration(h['duration_s']):>8s}  {lbl}")
            print()


if __name__ == "__main__":
    main()
