"""
src/data/dataset.py
===================
PyTorch Dataset returning (x, tau, mask, x_static, y_mort, y_los).

Key changes vs previous version
---------------------------------
NaN-aware normalization:
  Sequences now contain NaN for features not observed at a given timestamp.
  Normalization must only apply to observed positions — unobserved stay as
  0.0 so GRUDImputer can distinguish "not measured" from "measured = 0".

  Old (wrong):
    x = (x - mean) / std          # NaN propagates through
    x = np.clip(x * mask, ...)    # masks to 0 — loses NaN structure

  New (correct):
    x_norm = (x - mean) / std
    x = np.where(mask > 0, x_norm, 0.0)   # observed → normalized
                                           # unobserved → 0.0
    x = np.clip(x, -CLIP, CLIP)

  GRUDImputer inside MambaEncoder will then impute unobserved positions
  using time-decay toward the batch mean — correct irregular TS behavior.

Normalisation statistics are computed on the TRAINING split ONLY using
only OBSERVED values (mask > 0). Zero-padded positions are excluded so
mean/std reflect true clinical distributions.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_CLIP = 5.0   # hard clip after normalisation — prevents fp16/bf16 overflow


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def filter_ids_by_obs(sequences: Dict, ids: List, min_obs: int = 3) -> List:
    """
    Return subset of ids with at least min_obs observed values.
    Call AFTER train/val/test split, on each split separately.
    """
    return [sid for sid in ids if sequences[sid][2].sum() >= min_obs]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MaBoostDataset(Dataset):
    def __init__(
        self,
        sequences:        Dict,
        mortality_labels: Dict,
        los_labels:       Dict,
        stay_ids:         List,
        los_remaining_labels: Optional[Dict] = None,
        teacher_mortality_probs: Optional[Dict] = None,
        teacher_los_preds: Optional[Dict] = None,
        static_features:  Optional[Dict] = None,
        seq_mean:         Optional[np.ndarray] = None,
        seq_std:          Optional[np.ndarray] = None,
        is_train:         bool = False,
        use_dynamic_drop: bool = False,
    ):
        self.seqs   = sequences
        self.mort   = mortality_labels
        self.los    = los_labels
        self.los_rem = los_remaining_labels
        self.teacher_mort = teacher_mortality_probs
        self.teacher_los = teacher_los_preds
        self.ids    = stay_ids
        self.static = static_features
        self.mean   = seq_mean
        self.std    = seq_std
        self.is_train = is_train
        self.use_dynamic_drop = use_dynamic_drop
        self._diagnostic_printed = False

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid        = self.ids[idx]
        seq, tau, mask = self.seqs[sid][:3]

        # Work on float32 copies
        x   = seq.copy().astype(np.float32)
        tau = tau.copy().astype(np.float32)
        m   = mask.copy().astype(np.float32)

        # Dynamic Causal Drop (Targeting early prediction resilience)
        # Controlled by use_dynamic_drop, NOT is_train — so benchmark
        # can disable augmentation while still shuffling.
        if self.use_dynamic_drop and np.random.rand() > 0.5:
            v_idx = np.where(m.sum(axis=1) > 0)[0]
            if len(v_idx) > 1:
                # Randomly pick a cut point representing 'current time'
                cut_idx = np.random.choice(v_idx)
                
                # Alternate between Prefix tracking (early_6h) and Sliding Window (raw_event_stream=4h)
                obs_win = 4.0 if np.random.rand() > 0.5 else 999.0
                
                from src.data.temporal_samples import _window_and_pad
                x, tau, m = _window_and_pad(x, tau, m, cut_idx, obs_window_hours=obs_win)

        # 1. Sanitise tau — NaN/inf in tau causes ZOH gate to explode
        tau = np.nan_to_num(tau, nan=60.0, posinf=3600.0, neginf=1.0)
        tau = np.clip(tau, 1.0, 86400.0)

        # 2. NaN-aware Z-score normalisation
        #    - Only normalize observed positions (mask > 0)
        #    - Unobserved positions stay 0.0 for GRUDImputer
        #    - NaN in x at unobserved positions → set to 0.0 first
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if self.mean is not None:
            x_norm = (x - self.mean) / np.maximum(self.std, 1e-6)
            # Only apply normalization where feature was actually observed
            x = np.where(m > 0, x_norm, 0.0)

        # 3. Hard clip to prevent overflow in Mamba ZOH step
        x = np.clip(x, -_CLIP, _CLIP)

        x_t    = torch.from_numpy(x).float()
        tau_t  = torch.from_numpy(tau).float()
        mask_t = torch.from_numpy(m).float()
        y_mort = torch.tensor(self.mort[sid], dtype=torch.long)
        y_los  = torch.tensor(
            max(float(self.los[sid]), 0.0), dtype=torch.float32
        )
        y_los_rem = None
        if self.los_rem is not None and sid in self.los_rem:
            y_los_rem = torch.tensor(
                max(float(self.los_rem[sid]), 0.0), dtype=torch.float32
            )
        t_mort = None
        if self.teacher_mort is not None and sid in self.teacher_mort:
            t_mort = torch.tensor(
                float(np.clip(self.teacher_mort[sid], 1e-6, 1.0 - 1e-6)),
                dtype=torch.float32,
            )
        t_los = None
        if self.teacher_los is not None and sid in self.teacher_los:
            t_los = torch.tensor(
                max(float(self.teacher_los[sid]), 0.0), dtype=torch.float32
            )
        has_teacher = t_mort is not None and t_los is not None

        if self.static is not None:
            xs = torch.from_numpy(
                np.nan_to_num(
                    self.static[sid].copy(), nan=0.0
                ).astype(np.float32)
            ).float()
            if y_los_rem is not None:
                if has_teacher:
                    return x_t, tau_t, mask_t, xs, y_los_rem, t_mort, t_los, y_mort, y_los
                return x_t, tau_t, mask_t, xs, y_los_rem, y_mort, y_los
            if has_teacher:
                return x_t, tau_t, mask_t, xs, t_mort, t_los, y_mort, y_los
            return x_t, tau_t, mask_t, xs, y_mort, y_los

        if y_los_rem is not None:
            if has_teacher:
                return x_t, tau_t, mask_t, y_los_rem, t_mort, t_los, y_mort, y_los
            return x_t, tau_t, mask_t, y_los_rem, y_mort, y_los
        if has_teacher:
            return x_t, tau_t, mask_t, t_mort, t_los, y_mort, y_los
        return x_t, tau_t, mask_t, y_mort, y_los

    # ------------------------------------------------------------------
    @staticmethod
    def norm_stats(
        sequences: Dict, ids: List
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-feature mean and std over OBSERVED values in ids.

        Uses running-sum Welford method — never allocates a giant array.
        Excludes NaN and padding positions (mask == 0).

        MUST be called with train_ids only — never with all_ids or val/test.
        """
        n_feats = next(iter(sequences.values()))[0].shape[1]
        s1  = np.zeros(n_feats, dtype=np.float64)
        s2  = np.zeros(n_feats, dtype=np.float64)
        cnt = np.zeros(n_feats, dtype=np.int64)

        for sid in ids:
            seq, _, mask = sequences[sid][:3]
            m   = (mask > 0).astype(bool)          # (L, F) True = observed
            if not m.any():
                continue
            # Replace NaN with 0 only for accumulation — NaN positions
            # have mask=0 anyway so they won't be counted
            vals = np.nan_to_num(
                seq.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
            s1  += (vals * m).sum(axis=0)
            s2  += ((vals ** 2) * m).sum(axis=0)
            cnt += m.sum(axis=0)

        mean = np.zeros(n_feats, dtype=np.float32)
        std  = np.ones(n_feats,  dtype=np.float32)
        ok   = cnt > 0
        if ok.any():
            mean[ok] = (s1[ok] / cnt[ok]).astype(np.float32)
            var      = s2[ok] / cnt[ok] - mean[ok].astype(np.float64) ** 2
            std[ok]  = np.maximum(
                np.sqrt(np.maximum(var, 0.0)), 1e-6
            ).astype(np.float32)
        return mean, std


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(
    sequences,
    mortality_labels,
    los_labels,
    train_ids,
    val_ids,
    test_ids,
    static_features = None,
    los_remaining_labels = None,
    teacher_mortality_probs = None,
    teacher_los_preds = None,
    batch_size: int = 64,
    num_workers: int = 0,
    oversample_pos: bool = False,
    shuffle_train: bool = True,
    device: str = "cpu",
    force_stats = None,
    use_dynamic_drop: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    norm_stats computed from train_ids ONLY — val/test use same stats.
    oversample_pos: WeightedRandomSampler to balance training set.
    Note: do not use oversample_pos together with Focal Loss class weights
    as this double-compensates for class imbalance.

    force_stats: if provided, dict {"mean": ndarray, "std": ndarray} that
                 overrides local norm computation. Use to inject global stats
                 from Stage 1 training into benchmark evaluation.
    use_dynamic_drop: if True, enables Dynamic Causal Drop augmentation on
                      the training set. Decoupled from shuffle_train so that
                      benchmarks can shuffle without augmenting.
    """
    if force_stats is not None:
        mean = force_stats["mean"]
        std  = force_stats["std"]
        _norm_source = "Global (injected)"
    else:
        mean, std = MaBoostDataset.norm_stats(sequences, train_ids)
        _norm_source = "Local (computed from train_ids)"

    kw = dict(
        sequences        = sequences,
        mortality_labels = mortality_labels,
        los_labels       = los_labels,
        los_remaining_labels = los_remaining_labels,
        teacher_mortality_probs = teacher_mortality_probs,
        teacher_los_preds = teacher_los_preds,
        static_features  = static_features,
        seq_mean         = mean,
        seq_std          = std,
    )
    pin = device != "cpu"

    _drop_mode = "Dynamic" if use_dynamic_drop else "Fixed"
    train_ds = MaBoostDataset(
        stay_ids=train_ids, is_train=True,
        use_dynamic_drop=use_dynamic_drop, **kw,
    )

    if oversample_pos:
        labels  = np.array([int(train_ds.mort[sid]) for sid in train_ids])
        counts  = np.bincount(labels, minlength=2).astype(float)
        counts[counts == 0] = 1.0
        weights = [1.0 / counts[int(l)] for l in labels]
        sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True
        )
        tr = DataLoader(
            train_ds, sampler=sampler, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin,
        )
    else:
        tr = DataLoader(
            train_ds, shuffle=shuffle_train, batch_size=batch_size,
            num_workers=num_workers, pin_memory=pin,
        )

    va = DataLoader(
        MaBoostDataset(**kw, stay_ids=val_ids),
        shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin,
    )
    te = DataLoader(
        MaBoostDataset(**kw, stay_ids=test_ids),
        shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin,
    )

    print(
        f"[Dataset] train={len(train_ids):,}  val={len(val_ids):,}  "
        f"test={len(test_ids):,}  | "
        f"Drop: {_drop_mode} | Norm: {_norm_source}"
    )
    return tr, va, te
