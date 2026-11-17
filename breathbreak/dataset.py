"""Turn Recordings into model-ready tables.

- build_feature_table(): HRV features per 60-s window   -> Model A
- build_raw_windows():   raw 60-s PPG windows            -> Model B (embedder input)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import features as F
from . import signals as S
from .data import Recording

META = ["subject", "session", "dataset", "label", "session_index", "t_start_s"]


def _fs_for(kind, cfg):
    return cfg.ecg_fs if kind == "ecg" else cfg.ppg_fs


def _prep(rec: Recording, cfg) -> tuple[np.ndarray, float]:
    fs = _fs_for(rec.kind, cfg)
    return S.resample(rec.signal, rec.fs, fs), fs


def build_feature_table(recs: list[Recording], cfg, normalize: bool | None = None) -> pd.DataFrame:
    """One row per valid window; features optionally normalized to each
    person's 2-minute baseline (first 2 min of their baseline session)."""
    normalize = cfg.normalize_to_baseline if normalize is None else normalize
    tables, baselines = [], {}
    for rec in recs:
        x, fs = _prep(rec, cfg)
        meta = {"subject": rec.subject, "session": rec.session, "dataset": rec.dataset,
                "label": rec.label, "session_index": rec.extra.get("session_index", np.nan)}
        df = F.signal_to_feature_windows(x, fs, rec.kind, cfg, meta)
        if rec.is_baseline_session:
            baselines[rec.subject] = F.baseline_features(x, fs, rec.kind, cfg)
            # drop windows that overlap the baseline itself (they'd be ~0 by construction)
            if not df.empty:
                df = df[df.t_start_s >= cfg.baseline_s]
        tables.append(df)
    table = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    if normalize:
        missing = set(table.subject) - set(baselines)
        if missing:
            print(f"[dataset] no baseline for {len(missing)} subject(s), dropping: {sorted(missing)[:5]}...")
            table = table[~table.subject.isin(missing)]
        table = F.normalize_to_baseline(table, baselines, list(F.FEATURE_NAMES))
    return table.reset_index(drop=True)


def _cached_beats(rec: Recording, cfg, cache_dir: str = "cache/beats") -> np.ndarray:
    """R-peak / pulse-peak times in seconds, cached per recording."""
    import os
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{rec.dataset}_{rec.subject}_{rec.session}_{rec.kind}.npy")
    if os.path.exists(path):
        return np.load(path)
    x, fs = _prep(rec, cfg)
    p = (S.ecg_rpeaks(x, fs) if rec.kind == "ecg" else S.ppg_peaks(x, fs)) / fs
    np.save(path, p)
    return p


def _clean_mask(raw: np.ndarray, cfg) -> np.ndarray:
    """Same rule as signals.clean_ibi, returned as a mask so beat times stay aligned."""
    from scipy import signal as sps
    ok = (raw >= cfg.ibi_min_ms) & (raw <= cfg.ibi_max_ms)
    if raw.size >= 5:
        med = sps.medfilt(raw, kernel_size=5)
        ok &= np.abs(raw - med) <= cfg.ibi_max_jump * med
    return ok


def build_extended_hrv_table(recs: list[Recording], cfg, window_s: float = 180.0,
                             step_s: float = 60.0, baseline_s: float = 300.0) -> pd.DataFrame:
    """Extended HRV per window (default 3 min), relative to each person's
    baseline (first `baseline_s` of their baseline session).

    Log-ratio for scale features, difference for pNN50 / respiration / DFA.
    Subjects without a usable baseline are dropped.
    """
    rows, base = [], {}
    min_beats = cfg.min_beats_per_window * window_s / 60.0 * 0.8
    for rec in recs:
        p = _cached_beats(rec, cfg)
        if len(p) < 3:
            continue
        ibi_all, t_all = np.diff(p) * 1000.0, p[1:]
        if rec.is_baseline_session:
            m = t_all < baseline_s
            ok = _clean_mask(ibi_all[m], cfg)
            if ok.sum() >= cfg.min_beats_per_window:
                base[rec.subject] = F.hrv_features_extended(t_all[m][ok], ibi_all[m][ok])
        for s0 in np.arange(0.0, p[-1] - window_s + 1e-9, step_s):
            if rec.is_baseline_session and s0 < baseline_s:
                continue
            m = (t_all >= s0) & (t_all < s0 + window_s)
            ok = _clean_mask(ibi_all[m], cfg)
            if ok.sum() < min_beats or ok.mean() < cfg.min_clean_fraction:
                continue
            row = F.hrv_features_extended(t_all[m][ok], ibi_all[m][ok])
            row.update(subject=rec.subject, session=rec.session, dataset=rec.dataset, label=rec.label,
                       session_index=rec.extra.get("session_index", np.nan), t_start_s=float(s0))
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df.subject.isin(base)].reset_index(drop=True)
    for subj, b in base.items():
        m = df.subject == subj
        for c in F.EXTENDED_FEATURE_NAMES:
            if c in F.LOG_RATIO_FEATURES:
                df.loc[m, c] = np.log(np.maximum(df.loc[m, c].astype(float), 1e-9) / max(b[c], 1e-9))
            else:
                df.loc[m, c] = df.loc[m, c].astype(float) - b[c]
    return df


def build_raw_windows(recs: list[Recording], cfg):
    """Return (X_windows [n, samples], meta DataFrame, baseline_windows dict).

    baseline_windows[subject] = the first `baseline_s` of their baseline session,
    used to compute a personal baseline embedding.
    """
    X, rows, base = [], [], {}
    for rec in recs:
        x, fs = _prep(rec, cfg)
        if rec.is_baseline_session:
            base[rec.subject] = x[: int(cfg.baseline_s * fs)]
        for start, end in F.iter_windows(len(x), fs, cfg.window_s, cfg.step_s):
            t0 = start / fs
            if rec.is_baseline_session and t0 < cfg.baseline_s:
                continue
            seg = x[start:end]
            if not np.isfinite(seg).all() or np.std(seg) < 1e-9:
                continue
            X.append(seg)
            rows.append({"subject": rec.subject, "session": rec.session, "dataset": rec.dataset,
                         "label": rec.label, "session_index": rec.extra.get("session_index", np.nan),
                         "t_start_s": t0})
    return np.stack(X), pd.DataFrame(rows), base
