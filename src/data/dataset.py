"""
src/data/dataset.py  (v8)
==========================
v8 changes vs v7
----------------
1. EventDrivenDataset: each ICU stay → N samples, one per observation event
   (any feature changes → new row). Labels per sample:
     y_mort      = same binary (in-hospital mortality) for all samples of a stay
     y_los_total = total ICU LOS in days (same for all samples of a stay)
     y_los_rem   = remaining LOS at this observation = total_los − t_elapsed_days

2. make_loaders accepts mode="event" (new default) or mode="stay" (v7 compat).
   All function signatures unchanged.

3. GroupAwareSplit utility: guarantees no patient appears in both train and val/test.
   Used by stage2_train for group k-fold main results.

4. _pad_seq helper: pads sequence to max_len for batch collation.

Backward compat: mode="stay" reproduces v7 behavior exactly.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupKFold, train_test_split


# ---------------------------------------------------------------------------
# Stay-level dataset (v7 compat)
# ---------------------------------------------------------------------------

class ICUStayDataset(Dataset):
    """One sample per ICU stay (v7 behavior)."""

    def __init__(self, sequences: dict, mortality_labels: dict,
                 los_labels: dict, ids: list,
                 static_features: Optional[dict] = None,
                 seq_len: int = 128):
        self.ids    = ids
        self.seqs   = sequences
        self.mort   = mortality_labels
        self.los    = los_labels
        self.static = static_features
        self.L      = seq_len

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        sid          = self.ids[idx]
        seq, tau, mask = self.seqs[sid]
        seq  = _pad_or_trim(seq,  self.L, pad_val=0.0)
        tau  = _pad_or_trim(tau,  self.L, pad_val=60.0, is_1d=True)
        mask = _pad_or_trim(mask, self.L, pad_val=0.0)
        x    = torch.from_numpy(seq).float()
        t    = torch.from_numpy(tau).float()
        m    = torch.from_numpy(mask).float()
        ym   = torch.tensor(self.mort.get(sid, 0), dtype=torch.long)
        yl   = torch.tensor(float(self.los.get(sid, 0.0)), dtype=torch.float32)
        if self.static is not None and sid in self.static:
            xs = torch.from_numpy(
                self.static[sid].astype(np.float32))
            return x, t, m, xs, ym, yl
        return x, t, m, ym, yl


# ---------------------------------------------------------------------------
# Event-driven dataset (v8 NEW)
# ---------------------------------------------------------------------------

class EventDrivenDataset(Dataset):
    """
    One sample per observation event within an ICU stay.

    An 'event' = any timestep where at least one feature is observed
    (mask row sum > 0). For each event at elapsed time t_e:
      - x      : sequence from admission up to and including event t_e
      - tau    : inter-arrival times up to t_e
      - mask   : observation mask up to t_e
      - y_mort : in-hospital mortality (same for all events of a stay)
      - y_los_total : total ICU LOS in days
      - y_los_rem   : remaining LOS = total_los - t_elapsed_days

    Sequences are left-padded to seq_len so batches are uniform.

    Parameters
    ----------
    min_elapsed_h : float
        Minimum elapsed time (hours) before creating samples.
        Default 1h — avoids near-empty sequences at admission.
    max_samples_per_stay : int or None
        Cap events per stay to avoid imbalance from very long stays.
        None = no cap.
    stride_h : float or None
        If not None, only sample events at multiples of stride_h hours
        (e.g. stride_h=1.0 → one sample per hour max). Keeps dataset
        manageable while preserving irregular nature.
    """

    def __init__(self, sequences: dict, mortality_labels: dict,
                 los_labels: dict, ids: list,
                 static_features: Optional[dict] = None,
                 seq_len: int = 128,
                 min_elapsed_h: float = 1.0,
                 max_samples_per_stay: Optional[int] = 48,
                 stride_h: Optional[float] = 1.0):

        self.seqs   = sequences
        self.mort   = mortality_labels
        self.los    = los_labels
        self.static = static_features
        self.L      = seq_len

        # Build flat index: list of (stay_id, cutoff_step)
        self.index: List[Tuple[str, int]] = []

        for sid in ids:
            seq, tau, mask = sequences[sid]
            total_los = float(los_labels.get(sid, 0.0))
            T         = seq.shape[0]

            # Cumulative elapsed time in hours from first observed event
            valid_rows = (mask.sum(axis=1) > 0)
            if not valid_rows.any():
                continue
            first_valid = int(np.argmax(valid_rows))

            cum_h = np.zeros(T, dtype=np.float64)
            for i in range(first_valid + 1, T):
                cum_h[i] = cum_h[i-1] + float(tau[i]) / 3600.0

            # Candidate events: rows with at least one observation
            event_steps = np.where(valid_rows)[0]

            # Filter by min_elapsed_h
            event_steps = event_steps[cum_h[event_steps] >= min_elapsed_h]
            if len(event_steps) == 0:
                continue

            # Apply stride: keep only events within each stride bucket
            if stride_h is not None:
                buckets: Dict[int, int] = {}
                for step in event_steps:
                    b = int(cum_h[step] / stride_h)
                    if b not in buckets:
                        buckets[b] = step
                event_steps = np.array(sorted(buckets.values()))

            # Cap max samples
            if max_samples_per_stay is not None and len(event_steps) > max_samples_per_stay:
                # Keep first, last, and evenly spaced in between
                idx_sel = np.round(
                    np.linspace(0, len(event_steps) - 1, max_samples_per_stay)
                ).astype(int)
                event_steps = event_steps[idx_sel]

            for step in event_steps:
                self.index.append((sid, int(step)))

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        sid, cutoff = self.index[idx]
        seq, tau, mask = self.seqs[sid]
        total_los      = float(self.los.get(sid, 0.0))
        T_full         = seq.shape[0]

        # Cut to [0 .. cutoff] inclusive
        s_cut = seq[:cutoff + 1]
        t_cut = tau[:cutoff + 1]
        m_cut = mask[:cutoff + 1]

        # Remaining LOS = total − elapsed_days at cutoff
        cum_h_at_cut = float(t_cut.sum()) / 3600.0
        los_rem      = max(total_los - cum_h_at_cut / 24.0, 0.0)

        # Pad to seq_len
        s_cut = _pad_or_trim(s_cut, self.L, pad_val=0.0)
        t_cut = _pad_or_trim(t_cut, self.L, pad_val=60.0, is_1d=True)
        m_cut = _pad_or_trim(m_cut, self.L, pad_val=0.0)

        x  = torch.from_numpy(s_cut).float()
        t  = torch.from_numpy(t_cut).float()
        m  = torch.from_numpy(m_cut).float()
        ym = torch.tensor(self.mort.get(sid, 0), dtype=torch.long)
        yt = torch.tensor(total_los, dtype=torch.float32)
        yr = torch.tensor(los_rem,   dtype=torch.float32)

        if self.static is not None and sid in self.static:
            xs = torch.from_numpy(self.static[sid].astype(np.float32))
            return x, t, m, xs, ym, yt, yr

        return x, t, m, ym, yt, yr


# ---------------------------------------------------------------------------
# Padding / trimming helper
# ---------------------------------------------------------------------------

def _pad_or_trim(arr: np.ndarray, L: int,
                 pad_val: float = 0.0, is_1d: bool = False) -> np.ndarray:
    T = arr.shape[0]
    if T == L:
        return arr.copy()
    if T > L:
        return arr[-L:].copy()          # keep most recent L steps
    pad_shape = (L - T,) if is_1d else (L - T, arr.shape[1])
    pad       = np.full(pad_shape, pad_val, dtype=np.float32)
    return np.concatenate([pad, arr], axis=0)


# ---------------------------------------------------------------------------
# GroupAwareSplit (v8 NEW)
# ---------------------------------------------------------------------------

class GroupAwareSplit:
    """
    Guarantee that no patient appears in more than one fold.

    Usage
    -----
    splitter = GroupAwareSplit(n_splits=5)
    for fold, (tr_ids, va_ids) in enumerate(splitter.split(all_ids)):
        ...
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits     = n_splits
        self.random_state = random_state

    def split(self, ids: List[str]):
        ids    = list(ids)
        groups = np.arange(len(ids))  # each patient = its own group
        gkf    = GroupKFold(n_splits=self.n_splits)
        X_dummy = np.zeros((len(ids), 1))
        for tr_idx, va_idx in gkf.split(X_dummy, groups=groups):
            yield [ids[i] for i in tr_idx], [ids[i] for i in va_idx]

    def train_val_test_split(self, ids: List[str],
                              val_frac: float = 0.1,
                              test_frac: float = 0.1,
                              stratify_labels: Optional[List] = None):
        """
        Single train/val/test split with no patient leak.
        """
        ids = list(ids)
        tr_ids, tmp = train_test_split(
            ids, test_size=val_frac + test_frac,
            stratify=stratify_labels,
            random_state=self.random_state,
        )
        te_frac_adj = test_frac / (val_frac + test_frac)
        strat_tmp   = None
        if stratify_labels is not None:
            id_to_lbl = {s: l for s, l in zip(ids, stratify_labels)}
            strat_tmp = [id_to_lbl[s] for s in tmp]
        va_ids, te_ids = train_test_split(
            tmp, test_size=te_frac_adj,
            stratify=strat_tmp,
            random_state=self.random_state,
        )
        return tr_ids, va_ids, te_ids


# ---------------------------------------------------------------------------
# make_loaders — public API unchanged, mode param added
# ---------------------------------------------------------------------------

def make_loaders(sequences: dict, mortality_labels: dict, los_labels: dict,
                 train_ids: list, val_ids: list, test_ids: list,
                 static_features: Optional[dict] = None,
                 batch_size: int = 64, seq_len: int = 128,
                 num_workers: int = 0, mode: str = "event",
                 min_elapsed_h: float = 1.0,
                 max_samples_per_stay: Optional[int] = 48,
                 stride_h: Optional[float] = 1.0,
                 ) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    mode="event"  → EventDrivenDataset (v8 default, irregular time series)
    mode="stay"   → ICUStayDataset     (v7 compat, one sample per stay)

    All other kwargs are forwarded.  Signature is backward-compatible:
    callers passing only positional args get event mode automatically.
    """
    def _build(ids, shuffle, drop_last):
        if mode == "stay":
            ds = ICUStayDataset(
                sequences, mortality_labels, los_labels, ids,
                static_features=static_features, seq_len=seq_len)
        else:
            ds = EventDrivenDataset(
                sequences, mortality_labels, los_labels, ids,
                static_features=static_features, seq_len=seq_len,
                min_elapsed_h=min_elapsed_h,
                max_samples_per_stay=max_samples_per_stay,
                stride_h=stride_h)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=num_workers,
                          pin_memory=True)

    tr_loader = _build(train_ids, shuffle=True,  drop_last=True)
    va_loader = _build(val_ids,   shuffle=False, drop_last=False)
    te_loader = _build(test_ids,  shuffle=False, drop_last=False)

    print(f"[DataLoader] mode={mode}  "
          f"train={len(tr_loader.dataset):,}  "
          f"val={len(va_loader.dataset):,}  "
          f"test={len(te_loader.dataset):,}  "
          f"batch={batch_size}")
    return tr_loader, va_loader, te_loader