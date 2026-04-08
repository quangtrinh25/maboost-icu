"""
src/inference/flat_head_pipeline.py
===================================
Online / deployment pipeline for the flat_head model.

Architecture
------------
windowed irregular sequence -> raw_stats(last, mean, max, std, miss_rate)
                           + static features
                           -> XGBoost mortality head
                           -> XGBoost LOS head

Unlike the full MaBoost pipeline, this variant does not use z_multi / Mamba
latent features and does not require Stage-2 feature transforms.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import xgboost as xgb
from river import metrics as river_metrics

from src.models.xgboost_head import XGBLos, XGBMortality


class FlatHeadPipeline:
    _RISK = [(0.30, "LOW"), (0.60, "MODERATE"), (1.01, "HIGH")]

    def __init__(
        self,
        xgb_mort: XGBMortality,
        xgb_los: XGBLos,
        thresh_mild: float = 0.05,
        thresh_severe: float = 0.10,
        history_len: int = 48,
        obs_window_hours: float | None = None,
        device: str = "cpu",
    ):
        self.xgb_mort = xgb_mort
        self.xgb_los = xgb_los
        self.thresh_mild = thresh_mild
        self.thresh_severe = thresh_severe
        self.history_len = history_len
        self.obs_window_hours = obs_window_hours
        self.device = device
        self.auc_tracker = river_metrics.ROCAUC()
        self.auc_history: List[float] = []
        self.n_updates = 0
        self._hist: Dict[str, deque] = {}
        self.ckpt_dir: str | None = None

    def _trim_history(
        self,
        hist: list[tuple[np.ndarray, float, np.ndarray]],
    ) -> list[tuple[np.ndarray, float, np.ndarray]]:
        if not hist:
            return hist
        if self.obs_window_hours is None or self.obs_window_hours <= 0:
            return hist[-self.history_len:]

        max_seconds = float(self.obs_window_hours) * 3600.0
        start_idx = len(hist) - 1
        elapsed = 0.0
        for i in range(len(hist) - 1, -1, -1):
            start_idx = i
            if i < len(hist) - 1:
                elapsed += float(np.nan_to_num(
                    hist[i + 1][1], nan=60.0, posinf=3600.0, neginf=1.0
                ))
                if elapsed > max_seconds:
                    start_idx = i + 1
                    break
        return hist[max(start_idx, len(hist) - self.history_len):]

    @staticmethod
    def _raw_stats(seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
        feats = []
        for j in range(seq.shape[1]):
            v = seq[:, j]
            m = mask[:, j].astype(bool)
            obs = v[m] if m.any() else None
            if obs is not None:
                obs = obs[~np.isnan(obs)]
            if obs is not None and len(obs) > 0:
                feats.extend([
                    float(obs[-1]),
                    float(np.nanmean(obs)),
                    float(np.nanmax(obs)),
                    float(np.nanstd(obs)) if len(obs) > 1 else 0.0,
                    float(1.0 - m.mean()),
                ])
            else:
                feats.extend([np.nan, np.nan, np.nan, np.nan, 1.0])
        return np.asarray(feats, dtype=np.float32)

    def _build_feature_vector(
        self,
        stay_id: str,
        x_new: np.ndarray,
        tau_new: float,
        x_static: np.ndarray,
    ) -> np.ndarray:
        if stay_id not in self._hist:
            self._hist[stay_id] = deque(maxlen=self.history_len)
        x_arr = np.asarray(x_new, dtype=np.float32).ravel()
        mask_arr = np.isfinite(x_arr).astype(np.float32)
        x_arr = np.nan_to_num(x_arr, nan=0.0, posinf=0.0, neginf=0.0)
        self._hist[stay_id].append((x_arr, tau_new, mask_arr))

        hist = self._trim_history(list(self._hist[stay_id]))
        seq = np.stack([h[0] for h in hist]).astype(np.float32)
        mask = np.stack([h[2] for h in hist]).astype(np.float32)
        raw = self._raw_stats(seq, mask).reshape(1, -1)
        x_static = np.asarray(x_static, dtype=np.float32).reshape(1, -1)
        return np.hstack([raw, x_static]).astype(np.float32)

    def prepare_update_features(self, F_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return F_base, F_base

    def predict(self, stay_id: str, x_new: np.ndarray,
                tau_new: float, x_static: np.ndarray) -> Dict:
        F = self._build_feature_vector(stay_id, x_new, tau_new, x_static)
        mort_risk = float(self.xgb_mort.predict(F)[0])
        los_days = float(self.xgb_los.predict_days(F)[0])
        level = next(lbl for thr, lbl in self._RISK if mort_risk < thr)
        return {
            "mortality_risk": round(mort_risk, 4),
            "los_days": round(max(los_days, 0.0), 2),
            "risk_level": level,
        }

    def predict_with_features(
        self, stay_id: str, x_new: np.ndarray, tau_new: float, x_static: np.ndarray
    ) -> tuple[Dict, np.ndarray]:
        F = self._build_feature_vector(stay_id, x_new, tau_new, x_static)
        mort_risk = float(self.xgb_mort.predict(F)[0])
        los_days = float(self.xgb_los.predict_days(F)[0])
        level = next(lbl for thr, lbl in self._RISK if mort_risk < thr)
        return {
            "mortality_risk": round(mort_risk, 4),
            "los_days": round(max(los_days, 0.0), 2),
            "risk_level": level,
        }, F

    def update(self, F_new: np.ndarray, y_new: np.ndarray,
               yl_new: np.ndarray, F_new_los: np.ndarray | None = None) -> str:
        F_los = F_new if F_new_los is None else F_new_los
        dm_m = xgb.DMatrix(F_new, label=y_new, missing=np.nan)
        dm_l = xgb.DMatrix(F_los, label=np.log1p(yl_new), missing=np.nan)

        pred = self.xgb_mort.booster.predict(dm_m)
        for yi, pi in zip(y_new, pred):
            self.auc_tracker.update(int(yi), float(pi))
        self.auc_history.append(self.auc_tracker.get())

        drift = self._drift_state()
        if drift == "severe":
            mort_params = {
                "process_type": "default",
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "learning_rate": 0.03,
                "max_depth": 4,
                "subsample": 0.8,
                "reg_lambda": 1.5,
            }
            los_params = {
                "process_type": "default",
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "learning_rate": 0.03,
                "max_depth": 4,
                "subsample": 0.8,
                "reg_lambda": 1.5,
            }
            self.xgb_mort.booster = xgb.train(
                mort_params, dm_m, num_boost_round=10,
                xgb_model=self.xgb_mort.booster, verbose_eval=False)
            self.xgb_mort.ensemble_boosters = [self.xgb_mort.booster]
            self.xgb_los.booster = xgb.train(
                los_params, dm_l, num_boost_round=10,
                xgb_model=self.xgb_los.booster, verbose_eval=False)

        elif drift == "mild":
            self.xgb_mort.booster = xgb.train(
                {"process_type": "update", "updater": "refresh",
                 "refresh_leaf": True, "reg_lambda": 1.0},
                dm_m, num_boost_round=1,
                xgb_model=self.xgb_mort.booster, verbose_eval=False)
            self.xgb_mort.ensemble_boosters = [self.xgb_mort.booster]
            self.xgb_los.booster = xgb.train(
                {"process_type": "update", "updater": "refresh",
                 "refresh_leaf": True, "reg_lambda": 1.0},
                dm_l, num_boost_round=1,
                xgb_model=self.xgb_los.booster, verbose_eval=False)

        print(f"[Update #{self.n_updates:04d}] {drift.upper():<8} "
              f"AUC={self.auc_history[-1]:.4f}  batch={len(y_new)}")
        self.n_updates += 1
        return drift

    def _drift_state(self) -> str:
        if len(self.auc_history) < 3:
            return "stable"
        d = self.auc_history[-3] - self.auc_history[-1]
        if d > self.thresh_severe:
            return "severe"
        if d > self.thresh_mild:
            return "mild"
        return "stable"

    def discharge(self, stay_id: str):
        self._hist.pop(stay_id, None)

    @classmethod
    def load(cls, ckpt_dir: str, device: str = "cpu") -> "FlatHeadPipeline":
        p = Path(ckpt_dir)
        xgb_m = XGBMortality.load(
            str(p / "flat_head_mortality.ubj"),
            meta_path=str(p / "flat_head_mort_meta.pkl"),
        )
        xgb_l = XGBLos.load(
            str(p / "flat_head_los.ubj"),
            meta_path=str(p / "flat_head_los_meta.pkl"),
        )
        print(f"[Online] Loaded flat_head from {ckpt_dir}")
        obj = cls(xgb_m, xgb_l, device=device)
        obj.ckpt_dir = str(p)
        return obj
