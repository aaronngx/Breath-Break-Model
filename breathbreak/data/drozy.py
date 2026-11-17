"""DROZY loader (Massoz et al., WACV 2016). ECG at 512 Hz, 14 subjects x 3 sessions.

Expected layout after unzipping DROZY (adjust `edf_pattern` if yours differs):

    DROZY/
      psg/1-1.edf  1-2.edf  1-3.edf  2-1.edf ...     # <subject>-<session>.edf
      KSS.txt                                        # 14 rows x 3 cols (one KSS per session)

Sessions 1/2/3 are the three 10-min PVTs over ~28-30 h of sleep deprivation.
Session 1 (rested) is used as the personal baseline.

If your copy is organised differently, simplest fix: create
`drozy_labels.csv` with columns subject,session,kss and pass it as labels_csv.

License: research/evaluation only, no commercial development.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from . import Recording


def _read_edf_channel(path: str, want: str = "ECG"):
    import mne
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    names = raw.ch_names
    match = [n for n in names if want.lower() in n.lower()]
    if not match:
        raise ValueError(f"No channel containing '{want}' in {path}. Channels: {names}")
    sig = raw.get_data(picks=[match[0]])[0]
    return sig, float(raw.info["sfreq"])


def _load_kss(root: str, labels_csv: str | None) -> pd.DataFrame:
    if labels_csv:
        return pd.read_csv(labels_csv, dtype={"subject": str, "session": str})
    path = os.path.join(root, "KSS.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"KSS.txt not found in {root}; pass labels_csv=...")
    mat = np.loadtxt(path)
    rows = [{"subject": str(i + 1), "session": str(j + 1), "kss": int(mat[i, j])}
            for i in range(mat.shape[0]) for j in range(mat.shape[1])]
    return pd.DataFrame(rows)


def load_drozy(root: str, cfg, labels_csv: str | None = None,
               edf_pattern: str = "psg/*.edf") -> list[Recording]:
    kss = _load_kss(root, labels_csv)
    recs = []
    for path in sorted(glob.glob(os.path.join(root, edf_pattern))):
        m = re.match(r"(\d+)-(\d+)\.edf$", os.path.basename(path))
        if not m:
            continue
        subj, sess = m.group(1), m.group(2)
        row = kss[(kss.subject == subj) & (kss.session == sess)]
        if row.empty:
            continue
        k = int(row.kss.iloc[0])
        try:
            sig, fs = _read_edf_channel(path, "ECG")
        except Exception as e:  # missing / corrupt files exist in some copies
            print(f"[drozy] skip {path}: {e}")
            continue
        recs.append(Recording(
            subject=f"drozy_{subj}", session=sess, kind="ecg", signal=sig, fs=fs,
            label=int(k >= cfg.kss_fatigue_threshold), dataset="drozy",
            is_baseline_session=(sess == "1"), extra={"kss": k, "session_index": int(sess)},
        ))
    if not recs:
        raise RuntimeError(f"No DROZY recordings found under {root}/{edf_pattern}")
    return recs
