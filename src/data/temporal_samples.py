from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np


PAD_TAU = 60.0


def _cut_and_pad(
    seq: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    cut_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    L = seq.shape[0]
    cut_idx = int(max(0, min(cut_idx, L - 1)))
    s = seq[: cut_idx + 1]
    t = tau[: cut_idx + 1]
    m = mask[: cut_idx + 1]
    pad = L - s.shape[0]
    if pad <= 0:
        return s.astype(np.float32), t.astype(np.float32), m.astype(np.float32)
    s_pad = np.vstack([np.full((pad, seq.shape[1]), np.nan, dtype=np.float32), s.astype(np.float32)])
    t_pad = np.concatenate([np.full(pad, 60.0, dtype=np.float32), t.astype(np.float32)])
    m_pad = np.vstack([np.zeros((pad, mask.shape[1]), dtype=np.float32), m.astype(np.float32)])
    return s_pad, t_pad, m_pad


def _window_and_pad(
    seq: np.ndarray,
    tau: np.ndarray,
    mask: np.ndarray,
    cut_idx: int,
    obs_window_hours: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return either:
      - prefix up to cut_idx (obs_window_hours is None), or
      - trailing observation window ending at cut_idx.

    Output is padded back to original sequence length with front padding.
    """
    if obs_window_hours is None or obs_window_hours <= 0:
        return _cut_and_pad(seq, tau, mask, cut_idx)

    L = seq.shape[0]
    cut_idx = int(max(0, min(cut_idx, L - 1)))
    valid = (mask.sum(axis=1) > 0)
    if not valid.any():
        return _cut_and_pad(seq, tau, mask, cut_idx)

    max_seconds = float(obs_window_hours) * 3600.0
    start_idx = cut_idx
    elapsed = 0.0

    for i in range(cut_idx, -1, -1):
        if not valid[i]:
            continue
        start_idx = i
        if i < cut_idx:
            elapsed += float(np.nan_to_num(tau[i + 1], nan=PAD_TAU, posinf=3600.0, neginf=1.0))
            if elapsed > max_seconds:
                start_idx = i + 1
                break

    s = seq[start_idx: cut_idx + 1].astype(np.float32)
    t = tau[start_idx: cut_idx + 1].astype(np.float32)
    m = mask[start_idx: cut_idx + 1].astype(np.float32)

    if t.shape[0] > 0:
        t[0] = PAD_TAU

    pad = L - s.shape[0]
    if pad <= 0:
        return s, t, m

    s_pad = np.vstack([np.full((pad, seq.shape[1]), np.nan, dtype=np.float32), s])
    t_pad = np.concatenate([np.full(pad, PAD_TAU, dtype=np.float32), t])
    m_pad = np.vstack([np.zeros((pad, mask.shape[1]), dtype=np.float32), m])
    return s_pad, t_pad, m_pad


def _select_cutpoints(
    valid_idxs: np.ndarray,
    elapsed_s: np.ndarray,
    anchor_mode: str,
    min_elapsed_s: float,
    step_s: float,
    min_anchor_gap_s: float,
) -> List[int]:
    """
    Select cutpoints from valid observation rows.

    anchor_mode:
      - "step":  legacy periodic anchors every `step_s`
      - "event": anchor whenever a new observed timestamp arrives,
                 optionally respecting `min_anchor_gap_s`
    """
    cutpoints: List[int] = []
    mode = str(anchor_mode).lower().strip()

    if mode == "event":
        last_anchor_s = None
        for i in valid_idxs:
            t_s = float(elapsed_s[i])
            if t_s < min_elapsed_s:
                continue
            if last_anchor_s is None or (t_s - last_anchor_s) >= min_anchor_gap_s:
                cutpoints.append(int(i))
                last_anchor_s = t_s
        if len(valid_idxs) > 0 and int(valid_idxs[-1]) not in cutpoints:
            cutpoints.append(int(valid_idxs[-1]))
        return cutpoints

    next_thr = min_elapsed_s
    for i in valid_idxs:
        if float(elapsed_s[i]) >= next_thr:
            cutpoints.append(int(i))
            next_thr += step_s
    if len(valid_idxs) > 0 and int(valid_idxs[-1]) not in cutpoints:
        cutpoints.append(int(valid_idxs[-1]))
    return cutpoints


def _window_meta(
    tau: np.ndarray,
    mask: np.ndarray,
    anchor_elapsed_hours: float,
) -> Dict[str, float]:
    valid = (mask.sum(axis=1) > 0)
    n_timestamps = int(valid.sum())
    obs_count = int(mask.sum())
    if n_timestamps > 1:
        tau_valid = np.nan_to_num(
            tau[valid], nan=PAD_TAU, posinf=3600.0, neginf=1.0
        ).astype(np.float32)
        gap_std_s = float(np.std(tau_valid[1:]))
    else:
        gap_std_s = 0.0
    return {
        "anchor_elapsed_hours": float(anchor_elapsed_hours),
        "n_timestamps": n_timestamps,
        "obs_count": obs_count,
        "gap_std_s": gap_std_s,
    }


def build_event_driven_samples(
    sequences: Dict,
    mortality_labels: Dict,
    los_labels: Dict,
    stay_outcomes: Optional[Dict] = None,
    static_features: Optional[Dict] = None,
    stay_subjects: Optional[Dict] = None,
    anchor_mode: str = "step",
    min_elapsed_hours: float = 6.0,
    step_hours: float = 6.0,
    min_anchor_gap_minutes: float = 0.0,
    max_samples_per_stay: int = 64,
    obs_window_hours: Optional[float] = None,
    mortality_horizon_hours: Optional[float] = None,
    los_target: str = "remaining",
) -> Dict:
    """
    Build longitudinal/event-driven samples from per-stay sequences.

    Returns dict with:
      - sequences, mortality_labels, los_labels, los_remaining_labels
      - static_features
      - sample_to_stay (for group-aware split)
      - sample_to_subject
      - sample_metadata

    If obs_window_hours is set, each sample contains only the trailing
    observation window ending at the cutpoint. This is better aligned with
    early prediction than using the full prefix every time.

    If stay_outcomes and mortality_horizon_hours are set, mortality becomes
    a true horizon label: death within the next `mortality_horizon_hours`.

    los_target:
      - "remaining": predict remaining ICU LOS at each cutpoint
      - "total":     predict total ICU LOS (legacy behavior)

    max_samples_per_stay:
      - > 0: downsample anchors to this many cutpoints per stay
      - <=0: keep all selected anchors
    """
    out_seq = {}
    out_m = {}
    out_l_total = {}
    out_l_rem = {}
    out_static = {}
    sample_to_stay = {}
    sample_to_subject = {}
    sample_meta = {}

    min_elapsed_s = float(min_elapsed_hours) * 3600.0
    step_s = max(float(step_hours) * 3600.0, 60.0)
    min_anchor_gap_s = max(float(min_anchor_gap_minutes) * 60.0, 0.0)

    for sid, tpl in sequences.items():
        if sid not in mortality_labels or sid not in los_labels:
            continue
        seq, tau, mask = tpl[:3]
        elapsed_arr = tpl[3] if len(tpl) > 3 else None
        valid = mask.sum(axis=1) > 0
        idxs = np.where(valid)[0]
        if len(idxs) == 0:
            continue

        if elapsed_arr is not None:
            elapsed_s = np.asarray(elapsed_arr, dtype=np.float64)
        else:
            elapsed_s = np.cumsum(
                np.nan_to_num(tau, nan=60.0, posinf=3600.0, neginf=1.0).astype(np.float64)
            )
        cutpoints = _select_cutpoints(
            idxs,
            elapsed_s,
            anchor_mode=anchor_mode,
            min_elapsed_s=min_elapsed_s,
            step_s=step_s,
            min_anchor_gap_s=min_anchor_gap_s,
        )
        if not cutpoints:
            continue

        if max_samples_per_stay > 0 and len(cutpoints) > max_samples_per_stay:
            keep = np.linspace(0, len(cutpoints) - 1, max_samples_per_stay).round().astype(int)
            cutpoints = [cutpoints[k] for k in keep]

        los_total_days = float(los_labels[sid])
        meta = stay_outcomes.get(sid, {}) if stay_outcomes is not None else {}
        icu_los_days = float(meta.get("icu_los_days", los_total_days))
        death_offset_days = meta.get("death_offset_days", None)
        horizon_days = (
            float(mortality_horizon_hours) / 24.0
            if mortality_horizon_hours is not None and mortality_horizon_hours > 0
            else None
        )
        for j, cut_idx in enumerate(cutpoints):
            sample_id = f"{sid}::t{j:03d}"
            s, t, m = _window_and_pad(seq, tau, mask, cut_idx, obs_window_hours)
            out_seq[sample_id] = (s, t, m)
            anchor_elapsed_hours = float(elapsed_s[cut_idx] / 3600.0)
            elapsed_days = float(elapsed_s[cut_idx] / 86400.0)

            if horizon_days is not None:
                out_m[sample_id] = int(
                    death_offset_days is not None and
                    death_offset_days >= elapsed_days and
                    death_offset_days <= (elapsed_days + horizon_days)
                )
            else:
                out_m[sample_id] = int(mortality_labels[sid])

            rem_los = max(icu_los_days - elapsed_days, 0.0)
            if str(los_target).lower() == "remaining":
                out_l_total[sample_id] = rem_los
            else:
                out_l_total[sample_id] = los_total_days
                out_l_rem[sample_id] = rem_los
            sample_to_stay[sample_id] = sid
            if stay_subjects is not None and sid in stay_subjects:
                sample_to_subject[sample_id] = stay_subjects[sid]
            sample_meta[sample_id] = _window_meta(t, m, anchor_elapsed_hours)
            if static_features is not None and sid in static_features:
                out_static[sample_id] = static_features[sid]

    return {
        "sequences": out_seq,
        "mortality_labels": out_m,
        "los_labels": out_l_total,
        "los_remaining_labels": out_l_rem if out_l_rem else None,
        "static_features": out_static if static_features is not None else None,
        "sample_to_stay": sample_to_stay,
        "sample_to_subject": sample_to_subject if sample_to_subject else None,
        "sample_metadata": sample_meta,
    }
