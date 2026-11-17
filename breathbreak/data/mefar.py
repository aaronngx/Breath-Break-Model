"""MEFAR loader (Data in Brief 2024). Wrist BVP (Empatica E4, 64 Hz), 23 workers.

Use the RAW folder (the preprocessed CSVs have no participant ID column, so
they cannot be split by person):

    MEFAR_raw/
      <participant_id>/
        morning/BVP.csv     # Empatica E4 format: row0 = start unix time,
        evening/BVP.csv     #                     row1 = sample rate, then samples
      general_info.xlsx     # demographics + Chalder Fatigue Scale (CFS) per session

Labels: CFS >= 12 -> fatigued (class 1). Build a small CSV once from
general_info.xlsx with columns:  subject,session,cfs   (session = morning/evening)
and pass it as `labels_csv`.

Baseline: the first 2 minutes of each person's MORNING session.
License: CC BY 4.0 (commercial use allowed with attribution).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from . import Recording


def read_e4_csv(path: str):
    """Empatica E4 single-channel CSV -> (signal, fs)."""
    raw = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=float)
    fs = float(raw[1])
    return raw[2:], fs


def load_mefar(root: str, labels_csv: str, cfg, bvp_name: str = "BVP.csv") -> list[Recording]:
    labels = pd.read_csv(labels_csv, dtype={"subject": str, "session": str})  # keep "01" as "01"
    recs = []
    for subj_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(subj_dir):
            continue
        subj = os.path.basename(subj_dir)
        for sess in ("morning", "evening"):
            path = os.path.join(subj_dir, sess, bvp_name)
            if not os.path.exists(path):
                continue
            row = labels[(labels.subject == subj) & (labels.session == sess)]
            if row.empty:
                print(f"[mefar] no label for {subj}/{sess}, skipping")
                continue
            sig, fs = read_e4_csv(path)
            recs.append(Recording(
                subject=f"mefar_{subj}", session=sess, kind="ppg", signal=sig, fs=fs,
                label=int(row.cfs.iloc[0] >= cfg.cfs_fatigue_threshold), dataset="mefar",
                is_baseline_session=(sess == "morning"),
                extra={"cfs": float(row.cfs.iloc[0]), "session_index": 0 if sess == "morning" else 1},
            ))
    if not recs:
        raise RuntimeError(f"No MEFAR recordings found under {root}")
    return recs
