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
