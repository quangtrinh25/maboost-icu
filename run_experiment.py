"""
run_experiment.py
=================
MaBoost — single entry point.

Usage
-----
  python run_experiment.py                      # full pipeline
  python run_experiment.py --skip-etl          # reuse cached ETL
  python run_experiment.py --config my.yaml    # custom config
  python run_experiment.py --skip-baselines    # skip deep baselines
  python run_experiment.py --skip-new-tests    # skip early/sparse/rolling

Fixes vs previous version
--------------------------
1. feat_names corrected: 4*d_model + 5*d_input + 42 names
   (was: d_model + 42 = 170 names → SHAP labels were wrong)
2. enc_kw forwarded to OfflineBenchmark so eval_maboost() loads
   encoder with correct architecture matching the checkpoint.
3. Baselines wrapped in try/except with model_type fallback so
   a single broken baseline does not abort the entire benchmark.
4. Early prediction, sparse/dense, and rolling prediction tests
   added via OfflineBenchmark — results saved to separate CSV.
5. cfg["model"]["d_model"] fixed (was cfg.model.d_model — dict not object).
6. n_static_dims uses d_static (was x_static.shape[-1] — undefined var).
7. Duplicate z_T block removed; single block placed before run_shap().
8. cfg["device"] → s1["device"] (cfg has no top-level "device" key).
9. All new stage2 params forwarded from config (interactions, ensemble, cross).
"""
from __future__ import annotations
import argparse, pickle
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

from src.data.preprocess      import run_etl, load_etl_output
from src.data.dataset         import make_loaders
from src.data.temporal_samples import build_event_driven_samples
from src.models.mamba_encoder import DualHeadMamba, MambaEncoder
from src.models.xgboost_head  import XGBMortality, XGBLos
from src.training.stage1_train import train_stage1
from src.training.stage2_train import (
    load_frozen_encoder, extract_features, train_stage2, apply_stage2_transforms
)
from src.inference.offline_pipeline import OfflineBenchmark, BenchResult
from src.inference.offline_pipeline import _flat as _flat_features
from src.inference.online_pipeline import MaBoostOnlinePipeline
from src.visualization.training_plots import save_all
from src.visualization.shap_explain   import run_shap, build_feature_names


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _train_eval_xgb_heads(
    X_tr, ym_tr, yl_tr, X_va, ym_va, yl_va, X_te, ym_te, yl_te, device: str = "cuda"
):
    m = XGBMortality(device=device)
    l = XGBLos(device=device)
    m.fit(X_tr, ym_tr, X_va, ym_va)
    l.fit(X_tr, yl_tr, X_va, yl_va)
    yp_m = m.predict(X_te)
    yp_l = l.predict_days(X_te)
    return {
        "auroc": float(roc_auc_score(ym_te, yp_m)),
        "auprc": float(average_precision_score(ym_te, yp_m)),
        "los_mae": float(mean_absolute_error(yl_te, yp_l)),
        "los_rmse": float(np.sqrt(((yl_te - yp_l) ** 2).mean())),
    }


def _save_head_ablation(rows, out_csv: str):
    import csv
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["head", "auroc", "auprc", "los_mae", "los_rmse"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "head": r["head"],
                "auroc": f"{r['auroc']:.6f}",
                "auprc": f"{r['auprc']:.6f}",
                "los_mae": f"{r['los_mae']:.4f}",
                "los_rmse": f"{r['los_rmse']:.4f}",
            })
    print(f"[Ablation] Head comparison saved → {out_csv}")


def _print_head_ablation(rows):
    print("\n" + "=" * 70)
    print("  HEAD ABLATION — Frozen Mamba + XGBoost family")
    print("=" * 70)
    h = f"{'Head':<16}  {'AUROC':>6}  {'AUPRC':>6}  {'MAE(d)':>7}  {'RMSE(d)':>8}"
    print(h)
    print("─" * len(h))
    for r in sorted(rows, key=lambda x: x["auroc"], reverse=True):
        print(
            f"{r['head']:<16}  {r['auroc']:>6.4f}  {r['auprc']:>6.4f}  "
            f"{r['los_mae']:>7.3f}  {r['los_rmse']:>8.3f}"
        )
    print("─" * len(h))


def _online_loop(cfg: dict, event_file: str, poll_seconds: float = 0.25) -> None:
    """
    Simple online service loop over JSONL events.

    Each line in `event_file` should be a JSON object with at least:
      {
        "type": "observation",
        "stay_id": "12345",
        "x_new": [...],          # time-series feature vector
        "tau_new": 300.0,        # seconds since previous event
        "x_static": [...]        # static vector
      }

    Optional label/update event:
      {
        "type": "label",
        "stay_id": "12345",
        "y_mort": 0 or 1,
        "y_los": 3.25
      }
    """
    ckpt_dir = cfg["ckpt_dir"]
    m = cfg["model"]
    s1 = cfg["stage1"]
    online_cfg = cfg.get("online", {})
    history_len = int(online_cfg.get("history_len", cfg["data"].get("seq_len", 128)))
    thresh_mild = float(online_cfg.get("thresh_mild", 0.05))
    thresh_severe = float(online_cfg.get("thresh_severe", 0.10))
    flush_every = int(online_cfg.get("update_flush_every", 32))

    enc_kw = dict(
        d_state=m["d_state"],
        n_layers=m.get("n_layers", 3),
        n_heads=m["n_heads"],
        dropout=m["dropout"],
        topk=m.get("topk", 5),
    )
    if "ref_points" in m:
        enc_kw["ref_points"] = m["ref_points"]

    pipe = MaBoostOnlinePipeline.load(
        ckpt_dir=ckpt_dir,
        d_input=int(online_cfg.get("d_input", 40)),
        d_model=m["d_model"],
        device=s1["device"],
        **enc_kw,
    )
    pipe.history_len = history_len
    pipe.thresh_mild = thresh_mild
    pipe.thresh_severe = thresh_severe

    pending_feats = {}  # stay_id -> latest (1, D) feature matrix
    update_F_m, update_F_l, update_y, update_los = [], [], [], []
    ef = Path(event_file)
    ef.parent.mkdir(parents=True, exist_ok=True)
    ef.touch(exist_ok=True)

    print(f"[Online] Watching event stream: {ef}")
    print("[Online] Ctrl+C to stop")

    with ef.open("r") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_seconds)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                print(f"[Online] skip malformed JSON: {line[:100]}")
                continue

            etype = str(evt.get("type", "observation"))
            stay_id = str(evt.get("stay_id", ""))
            if not stay_id:
                print("[Online] skip event without stay_id")
                continue

            if etype == "observation":
                x_new = np.asarray(evt.get("x_new", []), dtype=np.float32)
                tau_new = float(evt.get("tau_new", 60.0))
                x_static = np.asarray(evt.get("x_static", []), dtype=np.float32)
                pred, F_cur = pipe.predict_with_features(stay_id, x_new, tau_new, x_static)
                F_cur_m = apply_stage2_transforms(F_cur, ckpt_dir=ckpt_dir, for_mortality=True)
                F_cur_l = apply_stage2_transforms(F_cur, ckpt_dir=ckpt_dir, for_mortality=False)
                pending_feats[stay_id] = (F_cur_m, F_cur_l)
                print(
                    f"[Online] stay={stay_id} risk={pred['mortality_risk']:.4f} "
                    f"los={pred['los_days']:.2f}d level={pred['risk_level']}"
                )
            elif etype == "label":
                if stay_id not in pending_feats:
                    print(f"[Online] label arrived but no cached feature for stay={stay_id}")
                    continue
                if "y_mort" not in evt or "y_los" not in evt:
                    print(f"[Online] skip label without y_mort/y_los for stay={stay_id}")
                    continue
                f_m, f_l = pending_feats[stay_id]
                update_F_m.append(f_m[0])
                update_F_l.append(f_l[0])
                update_y.append(int(evt["y_mort"]))
                update_los.append(float(evt["y_los"]))
                print(f"[Online] queued label for stay={stay_id} ({len(update_y)}/{flush_every})")

                if len(update_y) >= flush_every:
                    F_new_m = np.asarray(update_F_m, dtype=np.float32)
                    F_new_l = np.asarray(update_F_l, dtype=np.float32)
                    y_new = np.asarray(update_y, dtype=np.int32)
                    yl_new = np.asarray(update_los, dtype=np.float32)
                    drift = pipe.update(F_new_m, y_new, yl_new, F_new_los=F_new_l)
                    print(f"[Online] model update applied with drift={drift}")
                    update_F_m.clear()
                    update_F_l.clear()
                    update_y.clear()
                    update_los.clear()
            elif etype == "discharge":
                pipe.discharge(stay_id)
                pending_feats.pop(stay_id, None)
                print(f"[Online] discharged stay={stay_id}")
            else:
                print(f"[Online] unknown event type={etype} (ignored)")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config",         default="config.yaml")
    pa.add_argument("--skip-etl",       action="store_true")
    pa.add_argument("--skip-baselines", action="store_true",
                    help="Skip deep learning baselines (GRU-D, LSTM, etc.)")
    pa.add_argument("--skip-new-tests", action="store_true",
                    help="Skip early/sparse/rolling prediction tests")
    pa.add_argument("--mode", choices=["offline", "online_service"], default="offline")
    pa.add_argument("--event-file", default=None,
                    help="Path to JSONL stream file for --mode online_service")
    pa.add_argument("--poll-seconds", type=float, default=0.25,
                    help="Tail polling interval for online service")
    pa.add_argument("--fixed_48h", action="store_true",
                    help="Use fixed first-48h ETL window for offline training/eval")
    pa.add_argument("--expanding_window", action="store_true",
                    help="Use full-stay ETL window for offline training/eval")
    args = pa.parse_args()
    cfg  = load_cfg(args.config)
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if args.mode == "online_service":
        event_file = args.event_file or cfg.get("online", {}).get(
            "event_file", str(Path(cfg["results_dir"]) / "online_events.jsonl")
        )
        _online_loop(cfg, event_file, poll_seconds=args.poll_seconds)
        return

    if args.fixed_48h and args.expanding_window:
        raise ValueError("Choose only one of --fixed_48h or --expanding_window")
    if args.expanding_window:
        window_mode = "expanding_window"
    elif args.fixed_48h:
        window_mode = "fixed_48h"
    else:
        window_mode = cfg.get("data", {}).get("window_mode", "fixed_48h")
    window_hours = float(cfg["data"].get("window_hours", 48.0))
    crop_strategy = cfg.get("data", {}).get(
        "sequence_crop",
        "first" if window_mode == "fixed_48h" else "last",
    )
    suffix = "expanding" if window_mode == "expanding_window" else "fixed48"

    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["ckpt_dir"]).mkdir(parents=True, exist_ok=True)

    # ── 1. ETL ──────────────────────────────────────────────────────────
    etl_cache = Path(cfg["data"]["processed"]) / f"etl_output_{suffix}.pkl"
    if args.skip_etl and etl_cache.exists():
        etl = load_etl_output(str(etl_cache))
    else:
        etl = run_etl(
            mimic_dir = cfg["data"]["mimic_dir"],
            seq_len   = cfg["data"]["seq_len"],
            min_age   = cfg["data"]["min_age"],
            window_mode = window_mode,
            window_hours = window_hours,
            crop_strategy = crop_strategy,
            save_dir  = cfg["data"]["processed"],
        )

    seqs, static  = etl["sequences"], etl["static_features"]
    y_mort, y_los = etl["mortality_labels"], etl["los_labels"]
    stay_outcomes = etl.get("stay_outcomes", {})
    y_los_rem = None
    ts_names      = etl["feature_names"]   # 40 time-series feature names
    st_names      = etl["static_names"]    # 42 static feature names
    d_input       = len(ts_names)
    # Optional research-grade longitudinal expansion for expanding-window mode.
    research_cfg = cfg.get("research", {})
    sample_to_stay = None
    if research_cfg.get("use_longitudinal_samples", True):
        temporal = build_event_driven_samples(
            sequences=seqs,
            mortality_labels=y_mort,
            los_labels=y_los,
            stay_outcomes=stay_outcomes,
            static_features=static,
            min_elapsed_hours=float(research_cfg.get("min_elapsed_hours", 6.0)),
            step_hours=float(research_cfg.get("step_hours", 6.0)),
            max_samples_per_stay=int(research_cfg.get("max_samples_per_stay", 64)),
            obs_window_hours=(
                float(research_cfg["obs_window_hours"])
                if research_cfg.get("use_observation_window", False)
                else None
            ),
            mortality_horizon_hours=(
                float(research_cfg["mortality_horizon_hours"])
                if research_cfg.get("use_horizon_labels", True)
                else None
            ),
            los_target=str(research_cfg.get("los_target", "remaining")),
        )
        seqs = temporal["sequences"]
        static = temporal["static_features"] if temporal["static_features"] is not None else static
        y_mort = temporal["mortality_labels"]
        y_los = temporal["los_labels"]
        y_los_rem = temporal["los_remaining_labels"]
        sample_to_stay = temporal["sample_to_stay"]
        print(
            f"[Research] Longitudinal samples: {len(seqs):,} "
            f"from {len(set(sample_to_stay.values())):,} stays | "
            f"obs_window={research_cfg.get('obs_window_hours', 'prefix')}h | "
            f"mort_h={research_cfg.get('mortality_horizon_hours', 'final')}h | "
            f"los={research_cfg.get('los_target', 'remaining')}"
        )

    # FIX 5: cfg is a dict — use cfg["model"]["d_model"], not cfg.model.d_model
    d_model       = cfg["model"]["d_model"]

    # FIX 1 — feat_names: 4*d_model + 5*d_input + len(st_names)
    feat_names = (
        [f"h_last_{i}"  for i in range(d_model)] +
        [f"h_mean_{i}"  for i in range(d_model)] +
        [f"h_max_{i}"   for i in range(d_model)] +
        [f"h_attn_{i}"  for i in range(d_model)] +
        [f"last_{n}"    for n in ts_names] +
        [f"mean_{n}"    for n in ts_names] +
        [f"max_{n}"     for n in ts_names] +
        [f"std_{n}"     for n in ts_names] +
        [f"miss_{n}"    for n in ts_names] +
        list(st_names)
    )
    assert len(feat_names) == 4 * d_model + 5 * d_input + len(st_names), (
        f"feat_names length mismatch: {len(feat_names)} vs "
        f"{4*d_model + 5*d_input + len(st_names)}"
    )

    # ── 2. Split ─────────────────────────────────────────────────────────
    all_ids = sorted(set(seqs) & set(y_mort))
    labels  = [y_mort[s] for s in all_ids]
    tr_f    = cfg["data"]["train_frac"]
    va_f    = cfg["data"]["val_frac"]
    if sample_to_stay is not None and research_cfg.get("group_aware_split", True):
        groups = np.array([sample_to_stay[s] for s in all_ids])
        gss1 = GroupShuffleSplit(n_splits=1, train_size=tr_f, random_state=cfg["seed"])
        tr_idx, tmp_idx = next(gss1.split(all_ids, labels, groups=groups))
        tr_ids = [all_ids[i] for i in tr_idx]
        tmp_ids = [all_ids[i] for i in tmp_idx]
        tmp_groups = np.array([sample_to_stay[s] for s in tmp_ids])
        te_frac = (1 - tr_f - va_f) / (1 - tr_f)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=te_frac, random_state=cfg["seed"])
        va_i, te_i = next(gss2.split(tmp_ids, [y_mort[s] for s in tmp_ids], groups=tmp_groups))
        va_ids = [tmp_ids[i] for i in va_i]
        te_ids = [tmp_ids[i] for i in te_i]
        print("[Research] Group-aware split enabled (no patient leakage across splits).")
    else:
        tr_ids, tmp, _, tmp_l = train_test_split(
            all_ids, labels,
            test_size=1 - tr_f,
            stratify=labels,
            random_state=cfg["seed"],
        )
        te_frac = (1 - tr_f - va_f) / (1 - tr_f)
        va_ids, te_ids = train_test_split(
            tmp, test_size=te_frac,
            stratify=tmp_l,
            random_state=cfg["seed"],
        )

    # Save split IDs
    splits_dir = Path(cfg["data"]["splits"])
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("train", tr_ids), ("val", va_ids), ("test", te_ids)]:
        with open(splits_dir / f"{name}_ids.pkl", "wb") as f:
            pickle.dump(ids, f)

    s1 = cfg["stage1"]
    tr_loader, va_loader, te_loader = make_loaders(
        seqs, y_mort, y_los, tr_ids, va_ids, te_ids,
        static_features=static,
        los_remaining_labels=y_los_rem,
        batch_size=s1["batch_size"],
    )

    # ── 3. Stage 1 — train Mamba encoder ────────────────────────────────
    m      = cfg["model"]
    enc_kw = dict(
        d_state   = m["d_state"],
        n_layers  = m.get("n_layers", 3),
        n_heads   = m["n_heads"],
        dropout   = m["dropout"],
        topk      = m.get("topk", 5),
    )
    if "ref_points" in m:
        enc_kw["ref_points"] = m["ref_points"]
    # FIX 6: d_static computed here; used later for n_static_dims in train_stage2
    d_static = next(iter(static.values())).shape[0] if static else 0
    model    = DualHeadMamba(
        d_input=d_input, d_model=d_model,
        d_static=d_static,
        enable_remaining_head=bool(y_los_rem is not None and research_cfg.get("enable_remaining_head", True)),
        **enc_kw,
    )
    use_mixup = bool(s1.get("use_mixup", True))
    if y_los_rem is not None:
        # In longitudinal / early-sample mode, mixup blurs temporal supervision
        # and disables the remaining-LOS loss path in the current trainer.
        # Disable it whenever per-cutpoint remaining-LOS labels are present,
        # regardless of fixed_48h vs expanding_window ETL mode.
        use_mixup = bool(s1.get("use_mixup_expanding", False))
    model = train_stage1(
        model, tr_loader, va_loader,
        epochs     = s1["epochs"],
        lr         = s1["lr"],
        patience   = s1["patience"],
        clip_norm  = s1["clip_norm"],
        ckpt_dir   = cfg["ckpt_dir"],
        device     = s1["device"],
        warmup_frac = s1.get("warmup_frac", 0.10),
        ema_decay   = s1.get("ema_decay", 0.999),
        focal_gamma = s1.get("focal_gamma", 2.0),
        loss_weight = s1.get("loss_weight", 0.5),
        use_mixup   = use_mixup,
        mixup_alpha = s1.get("mixup_alpha", 0.4),
        aux_weight = s1.get("aux_weight",    0.1),
        aux_held_rate = s1.get("aux_held_rate", 0.2),
        remaining_los_weight = s1.get("remaining_los_weight", 0.3),
        trend_stop_window = s1.get("trend_stop_window", 8),
        trend_stop_delta  = s1.get("trend_stop_delta", 0.01),
    )

    # ── 4. Stage 2 — freeze + dual XGBoost ──────────────────────────────
    ckpt_path = str(Path(cfg["ckpt_dir"]) / "encoder_best.pth")
    enc       = load_frozen_encoder(ckpt_path, d_input, d_model, **enc_kw)
    s2        = cfg["stage2"]

    # OPT-7: optional encoder fine-tune before feature extraction
    finetune_epochs = s2.get("finetune_enc_epochs", 0)
    if finetune_epochs > 0:
        from src.training.stage2_train import _finetune_encoder
        enc = _finetune_encoder(
            enc, tr_loader,
            device = s1["device"],
            epochs = finetune_epochs,
            lr     = s2.get("finetune_enc_lr", 1e-5),
        )

    F_tr, ym_tr, yl_tr = extract_features(enc, tr_loader, s1["device"])
    F_va, ym_va, yl_va = extract_features(enc, va_loader, s1["device"])
    F_te, ym_te, yl_te = extract_features(enc, te_loader, s1["device"])

    xgb_mort, xgb_los, F_te_mort, F_te_los = train_stage2(
        F_tr, ym_tr, yl_tr,
        F_va, ym_va, yl_va,
        F_te             = F_te,
        ckpt_dir         = cfg["ckpt_dir"],
        xgb_device       = s2["device"],
        n_estimators     = s2["n_estimators"],
        max_depth        = s2["max_depth"],
        learning_rate    = s2["learning_rate"],
        subsample        = s2["subsample"],
        colsample_bytree = s2["colsample_bytree"],
        min_child_weight = s2["min_child_weight"],
        reg_lambda       = s2["reg_lambda"],
        early_stopping   = s2["early_stopping"],
        use_optuna           = s2.get("use_optuna",            True),
        n_optuna_trials      = s2.get("n_optuna_trials",       100),
        optuna_depth_max     = s2.get("optuna_depth_max",      10),
        optuna_n_max         = s2.get("optuna_n_max",          2000),
        use_dart             = s2.get("use_dart",              False),
        use_calibration      = s2.get("use_calibration",       True),
        use_isotonic         = s2.get("use_isotonic",          True),
        drop_low_importance  = s2.get("drop_low_importance",   True),
        # OPT-4: interaction features
        use_interaction_features = s2.get("use_interaction_features", True),
        n_interaction_top        = s2.get("n_interaction_top",        8),
        # OPT-5: mortality ensemble
        n_ensemble_mort          = s2.get("n_ensemble_mort",          2),
        # OPT-6: static × z_multi cross features
        # FIX 5+6: cfg["model"]["d_model"] and d_static (not cfg.model / x_static)
        use_static_cross         = s2.get("use_static_cross",         True),
        n_zmulti_dims            = 4 * d_model,   # 4 * d_model from cfg["model"]["d_model"]
        n_static_dims            = d_static,       # computed above from static features
    )

    # ── 5. MaBoost predictions on test set ───────────────────────────────
    # IMPORTANT: use F_te_final (after interaction + cross + filter transforms),
    # NOT the raw F_te from extract_features — the booster was trained on F_te_final's space.
    y_prob_mort = xgb_mort.predict(F_te_mort)
    y_pred_los  = xgb_los.predict_days(F_te_los)

    # ── 5b. Head ablation table (keeps frozen Mamba + XGBoost architecture) ──
    n_z = 4 * d_model
    # raw_stats is 5*d_input; optional static starts after that
    F_tr_enc = F_tr[:, :n_z]
    F_va_enc = F_va[:, :n_z]
    F_te_enc = F_te[:, :n_z]
    F_tr_flat_head = F_tr[:, n_z:]
    F_va_flat_head = F_va[:, n_z:]
    F_te_flat_head = F_te[:, n_z:]

    X_tr_flat = _flat_features(seqs, tr_ids)
    X_va_flat = _flat_features(seqs, va_ids)
    X_te_flat = _flat_features(seqs, te_ids)
    # IMPORTANT: train/val loaders are shuffled, so ym_tr/yl_tr coming from
    # extract_features() are NOT aligned with tr_ids ordering.
    # For XGBoost-flat, use labels derived from ids to match X_*_flat row order.
    ym_tr_flat = np.array([y_mort[s] for s in tr_ids], dtype=np.int32)
    ym_va_flat = np.array([y_mort[s] for s in va_ids], dtype=np.int32)
    ym_te_flat = np.array([y_mort[s] for s in te_ids], dtype=np.int32)
    yl_tr_flat = np.array([y_los[s] for s in tr_ids], dtype=np.float32)
    yl_va_flat = np.array([y_los[s] for s in va_ids], dtype=np.float32)
    yl_te_flat = np.array([y_los[s] for s in te_ids], dtype=np.float32)

    head_rows = []
    # 1) XGBoost-flat baseline
    head_rows.append({
        "head": "xgboost_flat",
        **_train_eval_xgb_heads(
            X_tr_flat, ym_tr_flat, yl_tr_flat,
            X_va_flat, ym_va_flat, yl_va_flat,
            X_te_flat, ym_te_flat, yl_te_flat,
            device=s2["device"],
        ),
    })
    # 2) Encoder head only (z_multi)
    head_rows.append({
        "head": "encoder_head",
        **_train_eval_xgb_heads(
            F_tr_enc, ym_tr, yl_tr, F_va_enc, ym_va, yl_va, F_te_enc, ym_te, yl_te,
            device=s2["device"],
        ),
    })
    # 3) Flat head from encoder extraction (raw_stats + static)
    head_rows.append({
        "head": "flat_head",
        **_train_eval_xgb_heads(
            F_tr_flat_head, ym_tr, yl_tr, F_va_flat_head, ym_va, yl_va, F_te_flat_head, ym_te, yl_te,
            device=s2["device"],
        ),
    })
    # 4) Hybrid head = your current frozen Mamba + XGBoost stage2
    head_rows.append({
        "head": "hybrid_head",
        "auroc": float(roc_auc_score(ym_te, y_prob_mort)),
        "auprc": float(average_precision_score(ym_te, y_prob_mort)),
        "los_mae": float(mean_absolute_error(yl_te, y_pred_los)),
        "los_rmse": float(np.sqrt(((yl_te - y_pred_los) ** 2).mean())),
    })
    _print_head_ablation(head_rows)
    _save_head_ablation(head_rows, f"{cfg['results_dir']}/head_ablation.csv")

    # ── 6. Offline benchmark ─────────────────────────────────────────────
    # FIX 2 — enc_kw forwarded so eval_maboost() loads correct architecture
    bench = OfflineBenchmark(
        sequences        = seqs,
        mortality_labels = y_mort,
        los_labels       = y_los,
        train_ids        = tr_ids,
        val_ids          = va_ids,
        test_ids         = te_ids,
        d_input          = d_input,
        d_model          = d_model,
        static_features  = static,
        encoder_path     = ckpt_path,
        mort_path        = str(Path(cfg["ckpt_dir"]) / "xgb_mortality.ubj"),
        los_path         = str(Path(cfg["ckpt_dir"]) / "xgb_los.ubj"),
        device           = s1["device"],
        enc_kw           = enc_kw,   # FIX 2
    )

    skip_list = ["MaBoost (ours)"]
    if args.skip_baselines:
        skip_list += [
            "GRU-D", "Transformer", "LSTM",
            "GRU-D (cell)", "SAnD", "STraTS", "InterpNet", "TCN",
        ]
    if args.skip_new_tests:
        skip_list += ["early_prediction", "sparse_vs_dense", "rolling_prediction"]
    if sample_to_stay is not None and research_cfg.get("use_observation_window", False):
        skip_list += ["early_prediction", "sparse_vs_dense", "rolling_prediction", "irregular_gaps"]
        print("[Benchmark] Skipping legacy temporal subgroup tests in early-sample mode.")

    # FIX 3 — baselines wrapped in try/except inside run_all()
    bench_results = bench.run_all(skip=skip_list)

    maboost_result = BenchResult(
        name     = "MaBoost (ours)",
        auroc    = float(roc_auc_score(ym_te, y_prob_mort)),
        auprc    = float(average_precision_score(ym_te, y_prob_mort)),
        los_mae  = float(mean_absolute_error(yl_te, y_pred_los)),
        los_rmse = float(np.sqrt(((yl_te - y_pred_los) ** 2).mean())),
    )

    full_results = [maboost_result] + [
        r for r in bench_results
        if r.extra.get("table", "full") == "full"
    ]
    new_test_results = [
        r for r in bench_results
        if r.extra.get("test") in ("early", "sparse_dense", "rolling", "gap_variance")
    ]

    bench.print_table(full_results)
    bench.save_csv(full_results, f"{cfg['results_dir']}/benchmark.csv")

    if new_test_results:
        bench.save_csv(
            new_test_results,
            f"{cfg['results_dir']}/benchmark_temporal.csv",
        )
        print(f"[Benchmark] Temporal results saved → "
              f"{cfg['results_dir']}/benchmark_temporal_irregular.csv")

    # ── 7. Visualization ─────────────────────────────────────────────────
    save_all(
        ym_te, y_prob_mort, yl_te, y_pred_los,
        benchmark_results=full_results,
        out_dir=cfg["results_dir"],
    )

    # FIX 7: z_T block appears once, before run_shap (was duplicated)
    # FIX 8: s1["device"] instead of cfg["device"] (no top-level "device" in cfg)
    enc_vis  = enc.to(s1["device"])
    z_T_list = []
    with torch.no_grad():
        for batch in te_loader:
            x    = batch[0].to(s1["device"])
            tau  = batch[1].to(s1["device"])
            mask = batch[2].to(s1["device"])
            z_T_list.append(enc_vis(x, tau, mask).cpu().numpy())
    z_T = np.vstack(z_T_list)

    F_te_shap_mort = F_te_mort
    F_te_shap_los = F_te_los

    feat_names_shap = build_feature_names(
        ts_names = ts_names,
        st_names = st_names,
        d_model  = d_model,
        keep_idx = xgb_mort.keep_idx,
    )

    print(
        f"[SHAP] F_te_mort={F_te_shap_mort.shape[1]} "
        f"F_te_los={F_te_shap_los.shape[1]} feat_names={len(feat_names_shap)}"
    )

    run_shap(
        xgb_mort, xgb_los,
        F_te_shap_mort, F_te_shap_los,
        feat_names_shap,
        y_prob_mort, ym_te.astype(int),
        z_T,
        cfg["results_dir"],
    )

    # ── 8. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  FINAL TEST RESULTS")
    print("=" * 55)
    print(f"  Mortality AUROC   : {roc_auc_score(ym_te, y_prob_mort):.4f}")
    print(f"  Mortality AUPRC   : {average_precision_score(ym_te, y_prob_mort):.4f}")
    print(f"  LOS MAE           : {mean_absolute_error(yl_te, y_pred_los):.3f} days")
    print(f"  LOS RMSE          : {np.sqrt(((yl_te - y_pred_los)**2).mean()):.3f} days")
    print(f"  Results →           {cfg['results_dir']}/")
    print(f"  Checkpoints →       {cfg['ckpt_dir']}/")
    print("=" * 55)

    if new_test_results and not args.skip_new_tests:
        early   = [r for r in new_test_results if r.extra.get("test") == "early"]
        rolling = [r for r in new_test_results if r.extra.get("test") == "rolling"]

        if early:
            print("\n  EARLY PREDICTION SUMMARY")
            print("  " + "-" * 45)
            mb_early   = [r for r in early if r.name.startswith("MaBoost")]
            flat_early = [r for r in early if r.name.startswith("XGB")]
            for mb, fl in zip(mb_early, flat_early):
                h     = mb.extra.get("hours", "?")
                delta = mb.auroc - fl.auroc
                print(f"  {h:>3.0f}h  MaBoost={mb.auroc:.4f}  "
                      f"XGB-flat={fl.auroc:.4f}  Δ={delta:+.4f}")

        if rolling:
            print("\n  ROLLING PREDICTION SUMMARY")
            print("  " + "-" * 45)
            mb_roll   = [r for r in rolling if "MaBoost" in r.name]
            flat_roll = [r for r in rolling if "XGB"     in r.name]
            for mb, fl in zip(mb_roll, flat_roll):
                h     = mb.extra.get("hours", "?")
                delta = mb.auroc - fl.auroc
                print(f"  {h:>3.0f}h  MaBoost={mb.auroc:.4f}  "
                      f"XGB-flat={fl.auroc:.4f}  Δ={delta:+.4f}")
        print("=" * 55)


if __name__ == "__main__":
    main()
