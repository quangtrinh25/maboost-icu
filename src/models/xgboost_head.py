"""
src/models/xgboost_head.py
==========================
XGBMortality and XGBLos — wrappers around xgb.Booster.

Extended to carry optional post-processing:
  XGBMortality.platt    — sklearn LogisticRegression for Platt scaling
  XGBMortality.keep_idx — feature indices after importance filtering
  XGBLos.iso            — sklearn IsotonicRegression for LOS bias correction

NaN handling
------------
All xgb.DMatrix calls pass `missing=np.nan` explicitly.  This tells XGBoost
to treat IEEE NaN as the "missing value" sentinel and route it via the
learned sparsity-aware split direction — the correct behaviour for features
that were never measured in a given ICU stay.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import xgboost as xgb

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

# Helper so every DMatrix call is consistent
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
        self.platt     = None    # Platt scaling calibrator
        self.keep_idx  = None    # feature importance filter indices

    def fit(self, F_tr, y_tr, F_va, y_va,
            n_rounds: int = 500, early: int = 30):
        """
        Fit mortality XGBoost with safe callback handling.

        - scale_pos_weight computed from label distribution.
        - EarlyStopping attached only when early > 0.
        - Retry without callbacks on AttributeError (xgb version quirks).
        - best_iteration guaranteed on the returned booster.
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
        """Apply feature importance filter if set."""
        return F[:, self.keep_idx] if self.keep_idx is not None else F

    def predict(self, F: np.ndarray) -> np.ndarray:
        """Return calibrated probability if Platt is fitted, else raw sigmoid."""
        F_sel = self._select(F)
        raw   = self.booster.predict(_dmatrix(F_sel))
        if self.platt is not None:
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
            m.platt    = meta.get("platt")
            m.keep_idx = meta.get("keep_idx")
        return m


class XGBLos:
    """
    Predicts log(1+LOS_days); inverse-transformed to days at prediction time.
    Optionally corrected by isotonic regression to remove systematic bias.
    """
    def __init__(self, device: str = "cuda", **kw):
        self.params = {
            **_DEFAULTS,
            "objective":   "reg:squarederror",
            "eval_metric": "rmse",
            "device":      device,
            **kw,
        }
        self.booster: xgb.Booster = None
        self.iso = None    # isotonic regression bias correction

    def fit(self, F_tr, y_tr, F_va, y_va,
            n_rounds: int = 500, early: int = 30):
        """
        Fit LOS XGBoost (trains on log1p(LOS)).

        - EarlyStopping only when early > 0.
        - Retry without callbacks on AttributeError.
        - best_iteration guaranteed.
        """
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
        """Raw log1p(LOS) predictions."""
        return self.booster.predict(_dmatrix(F))

    def predict_days(self, F: np.ndarray) -> np.ndarray:
        """Inverse-transformed LOS in days, optionally isotonic-corrected."""
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