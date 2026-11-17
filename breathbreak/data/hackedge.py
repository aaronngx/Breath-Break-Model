"""Loader for HackEdge-style AFE4950 recordings (raw TI GUI exports + a manifest).

    <root>/
      data_manifest.csv      file,person,condition,kss,notes[,wake,slept,pvt_json,recorded_at]
      normal/*.csv           TI exports: ECG_Time, ECG_Value (V, 500 Hz), TIA1-3_Time, TIA1-3_Value (V, 50 Hz)
      fatigue/*.csv

`file` may be relative to <root> or to <root>/.. (HackEdge used both). Rows whose
file is missing are skipped, and byte-identical files are loaded once.

Labels: KSS >= 7 when `kss` is filled in, otherwise condition (fatigue = 1, normal = 0).
Optional columns enable the schedule + PVT model (breathbreak/fusion.py):
    wake         wake-up time that day, HH:MM
    slept        hours slept the night before
    pvt_json     PVT result saved by `python -m breathbreak.pvt --out ...` (path relative to <root>)
    recorded_at  recording start, YYYY-MM-DD HH:MM
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

from . import Recording


def _volts(df: pd.DataFrame, ch: str) -> tuple[np.ndarray, np.ndarray]:
    t = df[f"{ch}_Time"].dropna().to_numpy(float) if f"{ch}_Time" in df else np.array([])
    if f"{ch}_Value" in df:
        v = df[f"{ch}_Value"].dropna().to_numpy(float)
    elif f"{ch}_RAW_Value" in df:                       # 24-bit two's complement, +-4.8 V range
        raw = df[f"{ch}_RAW_Value"].dropna().to_numpy(float)
        v = np.where(raw >= 2 ** 23, raw - 2 ** 24, raw) / (2 ** 23 / 4.8)
    else:
        return t, np.array([])
    n = min(len(t), len(v))
    return t[:n], v[:n]


def read_ti_export(path: str) -> dict:
    """-> {'ecg': (signal, fs), 'ppg': (signal, fs) or None, 'duration_s': float}"""
    df = pd.read_csv(path, index_col=False)            # every row ends with a trailing comma
    t_e, ecg = _volts(df, "ECG")
    fs_e = float(round(1.0 / np.median(np.diff(t_e))))
    out = {"ecg": (ecg, fs_e), "ppg": None, "duration_s": float(t_e[-1] - t_e[0])}
    t_o, opt = _volts(df, "TIA1-3")
    if len(opt) > 10 and np.ptp(opt) * 1000 > 5:       # > 5 mV peak-to-peak -> optical channel present
        out["ppg"] = (opt, float(round(1.0 / np.median(np.diff(t_o)))))
    return out


def _resolve(root: str, f: str) -> str | None:
    for p in (os.path.join(root, f), os.path.join(os.path.dirname(os.path.abspath(root)), f)):
        if os.path.exists(p):
            return p
    return None


def load_hackedge(root: str, cfg, manifest: str = "data_manifest.csv", kinds=("ecg",)) -> list[Recording]:
    man = pd.read_csv(os.path.join(root, manifest))
    recs, seen = [], {}
    for _, row in man.iterrows():
        path = _resolve(root, str(row["file"]))
        if path is None:
            print(f"[hackedge] skip (missing): {row['file']}")
            continue
        digest = hashlib.md5(open(path, "rb").read()).hexdigest()
        if digest in seen:
            print(f"[hackedge] skip (duplicate of {seen[digest]}): {row['file']}")
            continue
        seen[digest] = row["file"]
        sig = read_ti_export(path)
        kss = row.get("kss")
        has_kss = pd.notna(kss) and str(kss).strip() != ""
        label = int(float(kss) >= cfg.kss_fatigue_threshold) if has_kss else int(row["condition"] == "fatigue")
        extra = {k: row.get(k) for k in ("wake", "slept", "pvt_json", "recorded_at", "notes") if k in row}
        extra.update(file=row["file"], condition=row["condition"], kss=float(kss) if has_kss else None,
                     duration_s=sig["duration_s"], root=root)
        for kind in kinds:
            if sig[kind] is None:
                continue
            x, fs = sig[kind]
            recs.append(Recording(subject=f"hackedge_{row['person']}",
                                  session=os.path.splitext(str(row["file"]))[0].replace("/", "_"),
                                  kind=kind, signal=x, fs=fs, label=label, dataset="hackedge",
                                  is_baseline_session=(row["condition"] == "normal"), extra=dict(extra)))
    if not recs:
        raise RuntimeError(f"No recordings found from {root}/{manifest}")
    return recs
