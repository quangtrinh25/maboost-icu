"""
src/inference/online_pipeline.py
=================================
Hospital-ready online predictor.

predict()  →  { mortality_risk, los_days, risk_level }
update()   →  drift-adaptive XGBoost refresh (CPU, < 100 ms)

Bugs fixed vs. original
-----------------------
1. Severe drift params missing "objective" key
   → XGBoost silently falls back to reg:squarederror (regression),
     turns a binary classifier into a regressor mid-deployment.

2. LOS model never updated in update()
   → LOS predictions drift silently while only mortality is corrected.

3. load() drops Platt scaling and isotonic calibration
   → mort_meta.pkl / los_meta.pkl not loaded, so predictions are
     uncalibrated sigmoid scores instead of Platt-corrected probabilities.

4. predict() used enc(xt, tt) returning z_T (d_model-dim) instead of
   extract_features() returning [z_multi; raw_stats] (4*d+5*F dim).
   XGBoost was trained on 1010-dim features but received 170-dim at
   inference — causing crash or silent garbage predictions.

5. Leaf Refresh used num_boost_round=n_trees (re-processes all trees
   n_trees times) instead of num_boost_round=1 (one pass through all
   existing trees). Previous value was ~800x too expensive.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import xgboost as xgb
from river import metrics as river_metrics

from src.models.mamba_encoder import MambaEncoder
from src.models.xgboost_head  import XGBMortality, XGBLos


class MaBoostOnlinePipeline:
    """
    Frozen Mamba encoder + two live XGBoost models.

    Drift-adaptive update (Algorithm 1):
      3-step AUC drop δ:
        δ > thresh_severe → Continued Boosting  (+10 trees)
        δ > thresh_mild   → Leaf Refresh          (refit weights only)
        else              → Stable               (no change)

    All updates run on CPU — no GPU required at inference time.
    """

    _RISK = [(0.30, "LOW"), (0.60, "MODERATE"), (1.01, "HIGH")]

    def __init__(
        self,
        encoder:       MambaEncoder,
        xgb_mort:      XGBMortality,
        xgb_los:       XGBLos,
        thresh_mild:   float = 0.05,
        thresh_severe: float = 0.10,
        history_len:   int   = 48,
        device:        str   = "cuda",
    ):
        assert all(not p.requires_grad for p in encoder.parameters())
        self.enc           = encoder.eval().to(device)
        self.xgb_mort      = xgb_mort
        self.xgb_los       = xgb_los
        self.thresh_mild   = thresh_mild
        self.thresh_severe = thresh_severe
        self.history_len   = history_len
        self.device        = device
        self.auc_tracker   = river_metrics.ROCAUC()
        self.auc_history:  List[float] = []
        self.n_updates     = 0
        self._hist: Dict[str, deque] = {}

    # ------------------------------------------------------------------
    def predict(self, stay_id: str, x_new: np.ndarray,
                tau_new: float, x_static: np.ndarray) -> Dict:
        """
        FIX 4: dùng extract_features() thay vì enc() để lấy đúng
        feature vector [z_multi(4*d); raw_stats(5*F)] khớp với
        feature vector mà XGBoost được train.

        Trước đây enc(xt,tt) trả về z_T (d_model-dim = 128) nhưng
        XGBoost train trên 1010-dim → crash hoặc garbage predictions.
        """
        if stay_id not in self._hist:
            self._hist[stay_id] = deque(maxlen=self.history_len)
        self._hist[stay_id].append((x_new.ravel().astype(np.float32), tau_new))

        hist = list(self._hist[stay_id])
        seq  = np.stack([h[0] for h in hist]).astype(np.float32)
        taus = np.array([h[1] for h in hist], dtype=np.float32)

        with torch.no_grad():
            xt     = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
            tt     = torch.from_numpy(taus).float().unsqueeze(0).to(self.device)
            # FIX 4: mask cho các timestep thật (tất cả = 1 vì đây là live data)
            mask_t = torch.ones(1, len(hist), dtype=torch.float32).to(self.device)
            # FIX 4: dùng extract_features() trả về (z_multi, raw_stats)
            z_multi, raw_stats = self.enc.extract_features(xt, tt, mask_t)

        # FIX 4: ghép đúng thứ tự như Stage 2 extract_features()
        F = np.hstack([
            z_multi.cpu().numpy(),
            raw_stats.cpu().numpy(),
            x_static.reshape(1, -1),
        ]).astype(np.float32)

        mort_risk = float(self.xgb_mort.predict(F)[0])
        los_days  = float(self.xgb_los.predict_days(F)[0])
        level     = next(lbl for thr, lbl in self._RISK if mort_risk < thr)
        return {"mortality_risk": round(mort_risk, 4),
                "los_days":       round(max(los_days, 0), 2),
                "risk_level":     level}

    # ------------------------------------------------------------------
    def update(self, F_new: np.ndarray, y_new: np.ndarray,
               yl_new: np.ndarray) -> str:
        """
        Receive discharge batch, update rolling AUC, apply drift strategy.

        Parameters
        ----------
        F_new  : (N, d_features) feature matrix from frozen encoder
        y_new  : (N,) mortality labels  0/1
        yl_new : (N,) LOS labels in days

        FIX 1 — severe drift params now include "objective": "binary:logistic"
                 so the continued boosting round stays a classifier.

        FIX 2 — LOS model updated alongside mortality model in every
                 drift state (severe and mild) so LOS drift is corrected too.
        """
        dm_m = xgb.DMatrix(F_new, label=y_new,            missing=np.nan)
        dm_l = xgb.DMatrix(F_new, label=np.log1p(yl_new), missing=np.nan)

        # Update rolling AUC tracker
        pred = self.xgb_mort.booster.predict(dm_m)
        for yi, pi in zip(y_new, pred):
            self.auc_tracker.update(int(yi), float(pi))
        self.auc_history.append(self.auc_tracker.get())

        drift = self._drift_state()

        if drift == "severe":
            # Continued Boosting: add 10 new trees to adapt to new distribution.
            # FIX 1: "objective" key is mandatory — omitting it caused XGBoost
            # to silently switch to reg:squarederror (regression objective).
            mort_params = {
                "process_type": "default",
                "objective":    "binary:logistic",
                "eval_metric":  "auc",
                "learning_rate": 0.03,
                "max_depth":    4,
                "subsample":    0.8,
                "reg_lambda":   1.5,
            }
            los_params = {
                "process_type": "default",
                "objective":    "reg:squarederror",
                "eval_metric":  "rmse",
                "learning_rate": 0.03,
                "max_depth":    4,
                "subsample":    0.8,
                "reg_lambda":   1.5,
            }
            self.xgb_mort.booster = xgb.train(
                mort_params, dm_m, num_boost_round=10,
                xgb_model=self.xgb_mort.booster, verbose_eval=False)
            # FIX 2: update LOS model too
            self.xgb_los.booster = xgb.train(
                los_params, dm_l, num_boost_round=10,
                xgb_model=self.xgb_los.booster, verbose_eval=False)

        elif drift == "mild":
            # Leaf Refresh: keep tree structure, refit leaf weights only.
            # FIX 5: num_boost_round=1 — one pass through all existing trees.
            # Previous value was n_trees (~800) which ran 800 passes → very slow.
            self.xgb_mort.booster = xgb.train(
                {"process_type": "update", "updater": "refresh",
                 "refresh_leaf": True, "reg_lambda": 1.0},
                dm_m, num_boost_round=1,
                xgb_model=self.xgb_mort.booster, verbose_eval=False)
            # FIX 2: update LOS model too
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
        if d > self.thresh_severe: return "severe"
        if d > self.thresh_mild:   return "mild"
        return "stable"

    def discharge(self, stay_id: str):
        """Remove patient history after discharge to free memory."""
        self._hist.pop(stay_id, None)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, ckpt_dir: str, d_input: int = 47, d_model: int = 128,
             device: str = "cuda", **enc_kw) -> "MaBoostOnlinePipeline":
        """
        Load all components from checkpoint directory.

        FIX 3: now loads mort_meta.pkl and los_meta.pkl so that
        Platt scaling (mortality) and isotonic regression (LOS)
        calibration are preserved in production.
        Previously predictions were uncalibrated sigmoid scores.
        """
        p = Path(ckpt_dir)

        enc = MambaEncoder(d_input, d_model, **enc_kw)
        enc.load_state_dict(
            torch.load(str(p / "encoder_best.pth"),
                       map_location="cpu", weights_only=True)
        )
        for param in enc.parameters():
            param.requires_grad = False

        # FIX 3: pass meta_path so Platt / isotonic calibration is loaded
        xgb_m = XGBMortality.load(
            str(p / "xgb_mortality.ubj"),
            meta_path=str(p / "mort_meta.pkl"),
        )
        xgb_l = XGBLos.load(
            str(p / "xgb_los.ubj"),
            meta_path=str(p / "los_meta.pkl"),
        )
        print(f"[Online] Loaded from {ckpt_dir}")
        return cls(enc, xgb_m, xgb_l, device=device)