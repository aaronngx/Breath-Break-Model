"""WESAD loader (Schmidt et al., ICMI 2018). Used as a STRESS CONTROL, not fatigue.

    WESAD/S2/S2.pkl, WESAD/S3/S3.pkl, ...

Each pickle: data['signal']['chest']['ECG'] (700 Hz),
             data['signal']['wrist']['BVP'] (64 Hz),
             data['label'] at 700 Hz: 1=baseline, 2=stress, 3=amusement, 4=meditation.

We return one Recording per condition (baseline / stress) with label=None,
because nobody in WESAD is sleep-deprived. The baseline condition's first
2 minutes define the personal baseline.
License: scientific, non-commercial use only.
"""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np

from . import Recording

COND = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def load_wesad(root: str, kind: str = "ecg", conditions=(1, 2)) -> list[Recording]:
    recs = []
    for pkl in sorted(glob.glob(os.path.join(root, "S*", "S*.pkl"))):
        subj = os.path.basename(pkl).replace(".pkl", "")
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        lab = np.asarray(d["label"]).ravel()
        if kind == "ecg":
            sig, fs = np.asarray(d["signal"]["chest"]["ECG"]).ravel(), 700.0
            lab_s = lab
        else:
            sig, fs = np.asarray(d["signal"]["wrist"]["BVP"]).ravel(), 64.0
            # map 700 Hz labels onto 64 Hz samples
            idx = (np.arange(len(sig)) * 700 / 64).astype(int).clip(0, len(lab) - 1)
            lab_s = lab[idx]
        for c in conditions:
            m = lab_s == c
            if m.sum() < fs * 120:
                continue
            # take the longest contiguous block of this condition
            edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
            starts, ends = edges[::2], edges[1::2]
            k = np.argmax(ends - starts)
            recs.append(Recording(
                subject=f"wesad_{subj}", session=COND[c], kind=kind,
                signal=sig[starts[k]:ends[k]], fs=fs, label=None, dataset="wesad",
                is_baseline_session=(c == 1), extra={"condition": COND[c]},
            ))
    if not recs:
        raise RuntimeError(f"No WESAD pickles found under {root}")
    return recs
