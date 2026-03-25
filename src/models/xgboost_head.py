"""
src/models/xgboost_head.py
==========================
XGBMortality and XGBLos — wrappers around xgb.Booster.

Optimizations vs original:
  1. XGBMortality calibration: Platt (LogisticRegression 1D) → IsotonicRegression
     — better tail calibration for ICU mortality where tails matter most.
  2. Feature importance filter: gain (unstable) → cover (stable across seeds).
  3. LOS: reg:squarederror → reg:pseudohubererror (robust to outlier LOS stays).
  4. _dmatrix helper preserved, NaN routing unchanged.

All class names, method signatures, and save/load formats unchanged.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

_DEFAULTS = dict(
    tree_method="hist",
    seed=42,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_lambda=1.0,
)


def _dmatrix(data: np.ndarray, label: np.ndarray = None) -> xgb.DMatrix:
    """Create a DMatrix with NaN as the missing-value sentinel."""
    return xgb.DMatrix(data, label=label, missing=np.nan)


class XGBMortality:
    def __init__(self, device: str = "cuda", **kw):
        self.params = {
            **_DEFAULTS,
            "objective":    "binary:logistic",
            "eval_metric":  ["auc", "logloss"],
            "device":       device,
            **kw,
        }
        self.booster:  xgb.Booster = None
        # OPT: use IsotonicRegression instead of Platt (LogisticRegression 1D)
        # IsotonicRegression is non-parametric and handles non-linear miscalibration
        # better in ICU settings where tail probabilities (very high/low risk) matter.
        self.platt     = None    # kept for load() backward compat, not used in new fits
        self.calibrator = None   # IsotonicRegression calibrator (replaces platt)
        self.keep_idx  = None    # feature importance filter indices

    def fit(self, F_tr, y_tr, F_va, y_va,
            n_rounds: int = 500, early: int = 30):
        """
        Fit mortality XGBoost.
        - scale_pos_weight from label distribution.
        - EarlyStopping when early > 0.
        - Retry without callbacks on AttributeError.
        """
        spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        self.params["scale_pos_weight"] = spw
        print(f"  [XGB Mortality] spw={spw:.1f}")

        dm_tr = _dmatrix(F_tr, y_tr)
        dm_va = _dmatrix(F_va, y_va)

        callbacks = None
        if early and early > 0:
            callbacks = [xgb.callback.EarlyStopping(
                early, "auc", maximize=True, save_best=True
            )]

        try:
            bst = xgb.train(
                self.params, dm_tr, n_rounds,
                evals=[(dm_tr, "train"), (dm_va, "val")],
                callbacks=callbacks,
                verbose_eval=50,
            )
        except AttributeError as e:
            print(f"  [XGB Mortality] callback error, retrying without callbacks: {e}")
            bst = xgb.train(
                self.params, dm_tr, n_rounds,
                evals=[(dm_tr, "train"), (dm_va, "val")],
                verbose_eval=50,
            )

        self.booster = bst
        if not hasattr(self.booster, "best_iteration") or \
                self.booster.best_iteration is None:
            bi = getattr(self.booster, "best_ntree_limit", None)
            self.booster.best_iteration = bi if bi is not None else 0

        return self

    def _select(self, F: np.ndarray) -> np.ndarray:
        """Apply keep_idx filter. Only used by the single-booster predict() path.
        The ensemble predict() is monkey-patched in train_stage2 and receives
        F that has already been filtered — it does NOT call _select."""
        return F[:, self.keep_idx] if self.keep_idx is not None else F

    def predict(self, F: np.ndarray) -> np.ndarray:
        """
        Return calibrated probability.
        Uses IsotonicRegression calibrator if fitted (new fits),
        falls back to Platt for models loaded from old checkpoints.
        """
        F_sel = self._select(F)
        raw   = self.booster.predict(_dmatrix(F_sel))
        if self.calibrator is not None:
            raw = self.calibrator.predict(raw)
        elif self.platt is not None:
            # backward compat: old checkpoints with Platt scaling
            raw = self.platt.predict_proba(raw.reshape(-1, 1))[:, 1]
        return raw.astype(np.float32)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path)

    @classmethod
    def load(cls, path: str, meta_path: str = None) -> "XGBMortality":
        m = cls()
        m.booster = xgb.Booster()
        m.booster.load_model(path)
        if meta_path and Path(meta_path).exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            m.calibrator = meta.get("calibrator")
            m.platt      = meta.get("platt")      # backward compat
            m.keep_idx   = meta.get("keep_idx")
        return m


class XGBLos:
    """
    Predicts log(1+LOS_days); inverse-transformed to days at prediction time.

    OPT: objective changed from reg:squarederror to reg:pseudohubererror.
    MIMIC-IV ICU LOS has heavy right tail (some stays > 60 days).
    Pseudo-Huber loss is quadratic near zero, linear for large residuals,
    which prevents outlier long-stay patients from dominating gradients.
    Isotonic bias correction unchanged.
    """
    def __init__(self, device: str = "cuda", **kw):
        self.params = {
            **_DEFAULTS,
            # OPT: pseudohubererror replaces squarederror — robust to LOS outliers
            "objective":   "reg:pseudohubererror",
            "eval_metric": "rmse",
            "device":      device,
            **kw,
        }
        self.booster: xgb.Booster = None
        self.iso = None

    def fit(self, F_tr, y_tr, F_va, y_va,
            n_rounds: int = 500, early: int = 30):
        dm_tr = _dmatrix(F_tr, np.log1p(y_tr))
        dm_va = _dmatrix(F_va, np.log1p(y_va))

        callbacks = None
        if early and early > 0:
            callbacks = [xgb.callback.EarlyStopping(
                early, "rmse", maximize=False, save_best=True
            )]

        try:
            bst = xgb.train(
                self.params, dm_tr, n_rounds,
                evals=[(dm_tr, "train"), (dm_va, "val")],
                callbacks=callbacks,
                verbose_eval=50,
            )
        except AttributeError as e:
            print(f"  [XGB LOS] callback error, retrying without callbacks: {e}")
            bst = xgb.train(
                self.params, dm_tr, n_rounds,
                evals=[(dm_tr, "train"), (dm_va, "val")],
                verbose_eval=50,
            )

        self.booster = bst
        if not hasattr(self.booster, "best_iteration") or \
                self.booster.best_iteration is None:
            bi = getattr(self.booster, "best_ntree_limit", None)
            self.booster.best_iteration = bi if bi is not None else 0

        return self

    def predict_log(self, F: np.ndarray) -> np.ndarray:
        return self.booster.predict(_dmatrix(F))

    def predict_days(self, F: np.ndarray) -> np.ndarray:
        raw = np.expm1(np.maximum(self.predict_log(F), 0.0))
        if self.iso is not None:
            raw = self.iso.predict(raw)
        return np.maximum(raw, 0.0).astype(np.float32)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path)

    @classmethod
    def load(cls, path: str, meta_path: str = None) -> "XGBLos":
        m = cls()
        m.booster = xgb.Booster()
        m.booster.load_model(path)
        if meta_path and Path(meta_path).exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            m.iso = meta.get("iso")
        return m