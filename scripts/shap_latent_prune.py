"""
scripts/shap_latent_prune.py
==============================
Priority 4: SHAP-Guided Latent Dimension Pruning

Problem
-------
MaBoost feeds 4 × d_model Mamba latent dims to XGBoost.
With d_model=256 that is 1024 latent dims; with d_model=128 that is 512.
Many of these dims are low-variance or correlated with raw_stats, adding
noise that reduces XGBoost AUROC.

Solution
--------
1. Load the trained Stage 2 XGBoost mortality booster.
2. Get cover-based importance for every feature.
3. Identify the LATENT dims (prefix "z_") with cover below threshold.
4. Build a pruned keep_idx that retains:
   - ALL raw_stats dims (last/mean/max/std/miss × features)
   - ALL static dims
   - Only top-K latent dims by cover importance
5. Save the pruned keep_idx to checkpoints/stage2_latent_pruned_idx.pkl
6. Retrain Stage 2 XGBoost on pruned feature set.
7. Evaluate and print comparison.

Usage
-----
  python scripts/shap_latent_prune.py [--top_k 64] [--dry_run]

After running, Stage 2 will automatically use the pruned indices if
stage2_latent_pruned_idx.pkl exists in the checkpoints directory.
Add these lines to your inference code (offline_pipeline.py):

    pruned_path = Path(ckpt_dir) / "stage2_latent_pruned_idx.pkl"
    if pruned_path.exists():
        with open(pruned_path, "rb") as f:
            keep_idx = pickle.load(f)
        F_test = F_test[:, keep_idx]
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CKPT = ROOT / "checkpoints"
DATA = ROOT / "data"
RES  = ROOT / "results"


def _load_splits_and_features(ckpt_dir: Path, data_dir: Path):
    """Load F_train, F_val, F_test and labels from stage2 checkpoint data."""
    # Try loading stage2 training features if cached
    cache_path = ckpt_dir / "stage2_features_cache.pkl"
    if cache_path.exists():
        print(f"  Loading feature cache from {cache_path.name} …")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    raise FileNotFoundError(
        f"stage2_features_cache.pkl not found in {ckpt_dir}.\n"
        "The features cache is saved automatically during run_experiment.py.\n"
        "Please run run_experiment.py --stage stage2 first, or pass "
        "--from_etl to regenerate."
    )


def get_latent_mask(feature_names: list[str]) -> np.ndarray:
    """Return boolean array: True where feature is a Mamba latent dim (z_*)."""
    return np.array([n.startswith("z_") for n in feature_names])


def prune_latent_dims(
    F_tr:   np.ndarray,
    F_va:   np.ndarray,
    F_te:   np.ndarray,
    y_tr:   np.ndarray,
    feature_names: list[str],
    top_k:  int = 64,
    importance_type: str = "cover",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[str]]:
    """
    Train a shallow XGBoost to measure latent importance, then reduce
    the latent dims to top_k. Returns pruned arrays and keep_idx.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    n_feats = F_tr.shape[1]
    latent_mask = get_latent_mask(feature_names)
    n_latent    = int(latent_mask.sum())
    n_other     = n_feats - n_latent

    print(f"\n  Feature breakdown:")
    print(f"    Latent dims (z_*): {n_latent}")
    print(f"    Other features:    {n_other}")
    print(f"    Total:             {n_feats}")

    # ── Train a shallow XGBoost to measure importance ─────────────────────
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": spw,
        "seed": 42,
        "verbosity": 0,
    }
    dm_tr = xgb.DMatrix(np.nan_to_num(F_tr, nan=0.0), label=y_tr)
    dm_va = xgb.DMatrix(np.nan_to_num(F_va, nan=0.0))

    print(f"  Training shallow XGBoost (300 rounds) for importance …")
    bst = xgb.train(params, dm_tr, 300,
                    evals=[(dm_tr, "train")], verbose_eval=False)

    scores = bst.get_score(importance_type=importance_type)
    imp_arr = np.array([scores.get(f"f{i}", 0.0) for i in range(n_feats)])

    # ── Separate latent and other importance ──────────────────────────────
    latent_idx = np.where(latent_mask)[0]
    other_idx  = np.where(~latent_mask)[0]

    latent_imp = imp_arr[latent_idx]  # importance of each latent dim
    other_imp  = imp_arr[other_idx]

    print(f"\n  Cover importance stats:")
    print(f"    Latent dims — mean={latent_imp.mean():.4f}  "
          f"max={latent_imp.max():.4f}  zero_frac={float((latent_imp==0).mean()):.1%}")
    print(f"    Other feats — mean={other_imp.mean():.4f}  "
          f"max={other_imp.max():.4f}  zero_frac={float((other_imp==0).mean()):.1%}")

    # ── Keep top_k latent dims + all other features ───────────────────────
    top_latent_local = np.argsort(latent_imp)[-top_k:]   # indices within latent group
    top_latent_global = latent_idx[top_latent_local]      # global indices in F

    keep_idx = sorted(list(top_latent_global) + list(other_idx))
    keep_names = [feature_names[i] if i < len(feature_names) else f"f{i}"
                  for i in keep_idx]

    print(f"\n  Pruning: {n_feats} → {len(keep_idx)} features")
    print(f"    Kept latent dims: {len(top_latent_global)} / {n_latent}")
    print(f"    Kept other feats: {len(other_idx)} / {n_other}")

    # ── Evaluate: before vs after pruning ────────────────────────────────
    pred_full = bst.predict(xgb.DMatrix(np.nan_to_num(F_va, nan=0.0)))

    bst_pruned = xgb.train(
        params,
        xgb.DMatrix(np.nan_to_num(F_tr[:, keep_idx], nan=0.0), label=y_tr),
        300,
        evals=[(xgb.DMatrix(np.nan_to_num(F_tr[:, keep_idx], nan=0.0), label=y_tr), "tr")],
        verbose_eval=False,
    )
    pred_pruned = bst_pruned.predict(
        xgb.DMatrix(np.nan_to_num(F_va[:, keep_idx], nan=0.0))
    )

    print(f"\n  Shallow XGBoost val AUROC comparison (indicative, not final):")
    print(f"    Full features    ({n_feats} dims): {roc_auc_score(y_va, pred_full):.5f}")
    print(f"    Pruned features  ({len(keep_idx)} dims): {roc_auc_score(y_va, pred_pruned):.5f}")

    F_tr_pruned = F_tr[:, keep_idx]
    F_va_pruned = F_va[:, keep_idx]
    F_te_pruned = F_te[:, keep_idx]

    return F_tr_pruned, F_va_pruned, F_te_pruned, keep_idx, keep_names


def main():
    ap = argparse.ArgumentParser(description="SHAP Latent Pruning — Priority 4 Upgrade")
    ap.add_argument("--top_k",    type=int,  default=64,  help="Keep top-K latent dims (default=64)")
    ap.add_argument("--dry_run",  action="store_true",    help="Analyze importance only, no retraining")
    ap.add_argument("--ckpt_dir", default=str(CKPT),      help="Checkpoint directory")
    ap.add_argument("--out",      default="",             help="Output pkl path (default: ckpt_dir/stage2_latent_pruned_idx.pkl)")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    out_path = Path(args.out) if args.out else ckpt_dir / "stage2_latent_pruned_idx.pkl"

    print("=" * 60)
    print("Priority 4: SHAP-Guided Latent Pruning")
    print("=" * 60)
    print(f"  top_k={args.top_k}  dry_run={args.dry_run}")
    print(f"  ckpt_dir={ckpt_dir}")

    # ── Load cached features ──────────────────────────────────────────────
    try:
        cache = _load_splits_and_features(ckpt_dir, DATA)
        F_tr, y_m_tr, _ = cache["train"]
        F_va, y_m_va, _ = cache["val"]
        F_te, y_m_te, _ = cache["test"]
        feat_names       = cache.get("feature_names", [])
        print(f"  Loaded features: train={F_tr.shape}, val={F_va.shape}, test={F_te.shape}")
    except FileNotFoundError as e:
        print(f"\n  ⚠  {e}")
        print("\n  Fallback: running importance analysis on pre-saved XGB model only\n")
        # Load existing mortality booster for importance analysis
        import xgboost as xgb
        mort_ckpt = ckpt_dir / "xgb_mortality.pkl"
        if not mort_ckpt.exists():
            print(f"  ✗  No checkpoint found: {mort_ckpt}")
            return
        with open(mort_ckpt, "rb") as f:
            xgb_mort = pickle.load(f)
        bst = xgb_mort.booster
        scores = bst.get_score(importance_type="cover")
        n_feats = max(int(k[1:]) for k in scores if k.startswith("f")) + 1 if scores else 0
        imp = np.array([scores.get(f"f{i}", 0.0) for i in range(n_feats)])
        zero_pct = float((imp == 0).mean()) * 100
        print(f"  Existing booster: {bst.num_boosted_rounds()} trees, {n_feats} features")
        print(f"  Zero-importance features: {zero_pct:.1f}%")
        print(f"  Top-20 important feature indices: {np.argsort(imp)[-20:][::-1].tolist()}")
        print(f"\n  Run run_experiment.py to generate stage2_features_cache.pkl,")
        print(f"  then re-run this script for full pruning + retraining.")
        return

    if args.dry_run:
        print("\n  Dry-run mode: analyzing importance, skipping retraining …")
        import xgboost as xgb
        spw = float((y_m_tr == 0).sum() / max((y_m_tr == 1).sum(), 1))
        bst = xgb.train(
            {"objective": "binary:logistic", "eval_metric": "auc",
             "tree_method": "hist", "device": "cuda", "max_depth": 4,
             "scale_pos_weight": spw, "seed": 42, "verbosity": 0},
            xgb.DMatrix(np.nan_to_num(F_tr, nan=0.0), label=y_m_tr),
            300, verbose_eval=False,
        )
        scores = bst.get_score(importance_type="cover")
        n = F_tr.shape[1]
        imp = np.array([scores.get(f"f{i}", 0.0) for i in range(n)])
        # Show top-20
        top20 = np.argsort(imp)[-20:][::-1]
        print("\n  Top-20 features by cover importance:")
        for rank, i in enumerate(top20):
            name = feat_names[i] if i < len(feat_names) else f"f{i}"
            print(f"    [{rank+1:2d}] f{i:4d}  cover={imp[i]:.2f}  {name}")
        return

    # ── Full pruning + retraining ─────────────────────────────────────────
    F_tr_p, F_va_p, F_te_p, keep_idx, keep_names = prune_latent_dims(
        F_tr, F_va, F_te, y_m_tr,
        feature_names=feat_names,
        top_k=args.top_k,
    )

    # Save keep_idx
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"keep_idx": keep_idx, "keep_names": keep_names,
                     "top_k": args.top_k, "n_original": F_tr.shape[1]}, f)
    print(f"\n  ✓  Saved pruned idx → {out_path}")
    print(f"     Keep {len(keep_idx)} / {F_tr.shape[1]} features")
    print(f"\n  Now retrain Stage 2 XGBoost with pruned features …")
    print(f"  [Run: python run_experiment.py --stage stage2 --pruned_idx {out_path}]")
    print("\n  Or integrate automatically via run_experiment.py config option:")
    print("    stage2:\n      latent_prune_path: checkpoints/stage2_latent_pruned_idx.pkl")


if __name__ == "__main__":
    main()
