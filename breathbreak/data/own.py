"""Loader for your own AFE4950 recordings.

Export each recording as CSV with these columns (header row required):

    time_s, ecg, ppg            # ecg / ppg may be blank if not recorded

and keep one sessions file, `sessions.csv`:

    subject, session, file, fs, kss, is_baseline
    nhat,    rested,  nhat_rested.csv, 500, 3, 1
    nhat,    late,    nhat_late.csv,   500, 8, 0

`fs` is the sampling rate of the CSV rows. If ECG and PPG were sampled at
different rates on the AFE4950, resample to one rate before exporting
(or write two files and two rows).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import Recording


def load_own(root: str, cfg, sessions_csv: str = "sessions.csv", kinds=("ecg", "ppg")) -> list[Recording]:
    ses = pd.read_csv(os.path.join(root, sessions_csv))
    recs = []
    for _, r in ses.iterrows():
        df = pd.read_csv(os.path.join(root, r["file"]))
        for kind in kinds:
            if kind not in df or df[kind].isna().all():
                continue
            label = None if pd.isna(r.get("kss")) else int(r["kss"] >= cfg.kss_fatigue_threshold)
            recs.append(Recording(
                subject=f"own_{r['subject']}", session=str(r["session"]), kind=kind,
                signal=df[kind].to_numpy(dtype=float), fs=float(r["fs"]), label=label,
                dataset="own", is_baseline_session=bool(r.get("is_baseline", 0)),
                extra={"kss": r.get("kss"), "session_index": 0 if bool(r.get("is_baseline", 0)) else 1},
            ))
    return recs
