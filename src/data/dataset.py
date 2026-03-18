"""
src/data/dataset.py
===================
PyTorch Dataset returning (x, tau, mask, x_static, y_mort, y_los).

Normalisation statistics are computed on the TRAINING split ONLY.
Test and validation data are normalised with training statistics — never
the other way around (that would be data leakage).

filter_ids_by_obs() must be called AFTER train/val/test split and applied
to each split separately. Calling it on all_ids before splitting biases
the study population using information from all splits.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_CLIP = 5.0   # hard clip after normalisation — prevents fp16/bf16 overflow in Mamba


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def filter_ids_by_obs(sequences: Dict, ids: List, min_obs: int = 3) -> List:
    """
    Return the subset of `ids` with at least `min_obs` observed values.

    IMPORTANT: call this AFTER the train/val/test split, applied to each
    split separately. Do NOT call on all_ids before splitting.
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
        static_features:  Optional[Dict] = None,
        seq_mean:         Optional[np.ndarray] = None,
        seq_std:          Optional[np.ndarray] = None,
    ):
        self.seqs   = sequences
        self.mort   = mortality_labels
        self.los    = los_labels
        self.ids    = stay_ids
        self.static = static_features
        self.mean   = seq_mean
        self.std    = seq_std

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        seq, tau, mask = self.seqs[sid]
        x = seq.copy().astype(np.float32)

        # 1. Sanitise raw values from ETL
        x   = np.nan_to_num(x,   nan=0.0, posinf=0.0, neginf=0.0)
        tau = np.nan_to_num(tau, nan=60.0, posinf=3600.0, neginf=1.0)
        tau = np.clip(tau, 1.0, 86400.0)

        # 2. Z-score normalise using TRAINING-SPLIT statistics only
        if self.mean is not None:
            x = (x - self.mean) / np.maximum(self.std, 1e-6)

        # 3. Zero out unobserved positions, hard-clip to [-5, 5]
        #    Clipping prevents bf16 overflow inside Mamba ZOH step.
        x = np.clip(x * mask, -_CLIP, _CLIP)

        x_t    = torch.from_numpy(x).float()
        tau_t  = torch.from_numpy(tau.copy()).float()
        mask_t = torch.from_numpy(mask.copy()).float()
        y_mort = torch.tensor(self.mort[sid], dtype=torch.long)
        y_los  = torch.tensor(max(float(self.los[sid]), 0.0), dtype=torch.float32)

        if self.static is not None:
            xs = torch.from_numpy(
                np.nan_to_num(self.static[sid].copy(), nan=0.0).astype(np.float32)
            ).float()
            return x_t, tau_t, mask_t, xs, y_mort, y_los
        return x_t, tau_t, mask_t, y_mort, y_los

    # ------------------------------------------------------------------
    @staticmethod
    def norm_stats(sequences: Dict, ids: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-feature min and max over OBSERVED values in `ids`.
        Returns (min, range) so caller can apply min-max scaling [0,1].
        APRICOT-M paper §2.3 uses min-max (not z-score) for ICU features.

        Uses Welford / running-sum method — never allocates a giant array,
        safe for 15k+ training stays.

        MUST be called with train_ids only — never with all_ids or val/test.
        """
        n_feats = next(iter(sequences.values()))[0].shape[1]
        s1  = np.zeros(n_feats, dtype=np.float64)   # sum of values
        s2  = np.zeros(n_feats, dtype=np.float64)   # sum of squares
        cnt = np.zeros(n_feats, dtype=np.int64)

        for sid in ids:
            seq, _, mask = sequences[sid]
            # Use per-feature mask — NOT row-level mask.
            # Row-level mask (any feature observed in that row) would include
            # zero-padded values for unobserved features, pulling mean toward
            # 0 and deflating variance for rarely-observed features.
            m   = (mask > 0).astype(bool)   # (L, F)  True = truly observed
            if not m.any():
                continue
            vals = np.nan_to_num(
                seq.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
            # Only accumulate where the feature was actually observed
            s1  += (vals * m).sum(axis=0)
            s2  += ((vals ** 2) * m).sum(axis=0)
            cnt += m.sum(axis=0)

        mean = np.zeros(n_feats, dtype=np.float32)
        std  = np.ones(n_feats,  dtype=np.float32)
        ok   = cnt > 0
        if ok.any():
            mean[ok] = (s1[ok] / cnt[ok]).astype(np.float32)
            var      = s2[ok] / cnt[ok] - mean[ok].astype(np.float64) ** 2
            std[ok]  = np.maximum(np.sqrt(np.maximum(var, 0.0)), 1e-6).astype(np.float32)
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
    static_features=None,
    batch_size: int    = 64,
    num_workers: int   = 4,
    oversample_pos: bool = False,
    device: str        = "cpu",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    norm_stats is computed on train_ids ONLY — val and test receive the
    same mean/std for normalisation (no leakage).

    oversample_pos: use WeightedRandomSampler to balance the training
    set instead of relying solely on class weights in the loss.
    """
    # Compute normalisation stats from TRAINING split only
    mean, std = MaBoostDataset.norm_stats(sequences, train_ids)

    kw  = dict(sequences=sequences, mortality_labels=mortality_labels,
               los_labels=los_labels, static_features=static_features,
               seq_mean=mean, seq_std=std)
    pin = device != "cpu"

    train_ds = MaBoostDataset(**kw, stay_ids=train_ids)

    if oversample_pos:
        labels  = np.array([int(train_ds.mort[sid]) for sid in train_ids])
        counts  = np.bincount(labels, minlength=2).astype(float)
        counts[counts == 0] = 1.0
        weights = [1.0 / counts[int(l)] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights),
                                        replacement=True)
        tr = DataLoader(train_ds, sampler=sampler, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=pin)
    else:
        tr = DataLoader(train_ds, shuffle=True, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=pin)

    va = DataLoader(MaBoostDataset(**kw, stay_ids=val_ids),
                    shuffle=False, batch_size=batch_size,
                    num_workers=num_workers, pin_memory=pin)
    te = DataLoader(MaBoostDataset(**kw, stay_ids=test_ids),
                    shuffle=False, batch_size=batch_size,
                    num_workers=num_workers, pin_memory=pin)

    print(f"[Dataset] train={len(train_ids):,}  val={len(val_ids):,}  "
          f"test={len(test_ids):,}  (norm from train only, no leakage)")
    return tr, va, te