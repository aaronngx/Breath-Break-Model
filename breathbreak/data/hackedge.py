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


def read_ti_export(path: str, cfg=None) -> dict:
    """-> {'ecg': (signal, fs) or None, 'ppg': (signal, fs) or None, 'duration_s',
           'ecg_quality', 'ppg_quality'}

    Each channel passes through the signal-quality gate (signal_quality.py). A channel
    that fails (sensor not worn, noise, clipping, motion) is returned as None so it is
    never used; the Quality object says why.
    """
    from .. import CFG
    from ..signal_quality import check_ecg, check_ppg
    cfg = cfg or CFG
    df = pd.read_csv(path, index_col=False)            # every row ends with a trailing comma
    t_e, ecg = _volts(df, "ECG")
    fs_e = float(round(1.0 / np.median(np.diff(t_e))))
    qe = check_ecg(ecg, fs_e, cfg)
    out = {"ecg": (ecg, fs_e) if qe.ok else None, "ppg": None,
           "duration_s": float(t_e[-1] - t_e[0]), "ecg_quality": qe, "ppg_quality": None}
    t_o, opt = _volts(df, "TIA1-3")
    if len(opt) > 10:
        fs_o = float(round(1.0 / np.median(np.diff(t_o))))
        qp = check_ppg(opt, fs_o, qe.metrics.get("hr_bpm") if qe.ok else None)
        out["ppg_quality"] = qp
        if qp.ok:
            out["ppg"] = (opt, fs_o)
    return out


def _resolve(root: str, f: str) -> str | None:
    for p in (os.path.join(root, f), os.path.join(os.path.dirname(os.path.abspath(root)), f)):
        if os.path.exists(p):
            return p
    return None


def load_hackedge(root: str, cfg, manifest: str = "data_manifest.csv", kinds=("ecg",),
                  include_rejected: bool = False) -> list[Recording]:
    """Channels that fail the quality gate are left out. With include_rejected=True a
    recording whose ECG failed is still returned (signal=None) so reports can list it."""
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
        sig = read_ti_export(path, cfg)
        for ch in ("ecg", "ppg"):
            q = sig[f"{ch}_quality"]
            if q is not None and not q.ok:
                print(f"[hackedge] {ch.upper()} not used for {row['file']}: {q.message()}")
        kss = row.get("kss")
        has_kss = pd.notna(kss) and str(kss).strip() != ""
        label = int(float(kss) >= cfg.kss_fatigue_threshold) if has_kss else int(row["condition"] == "fatigue")
        extra = {k: row.get(k) for k in ("wake", "slept", "pvt_json", "recorded_at", "notes") if k in row}
        extra.update(file=row["file"], condition=row["condition"], kss=float(kss) if has_kss else None,
                     duration_s=sig["duration_s"], root=root,
                     ecg_quality=sig["ecg_quality"], ppg_quality=sig["ppg_quality"])
        for kind in kinds:
            if sig[kind] is None:
                if include_rejected and kind == "ecg":
                    recs.append(Recording(subject=f"hackedge_{row['person']}",
                                          session=os.path.splitext(str(row["file"]))[0].replace("/", "_"),
                                          kind=kind, signal=None, fs=0.0, label=label, dataset="hackedge",
                                          is_baseline_session=(row["condition"] == "normal"), extra=dict(extra)))
                continue
            x, fs = sig[kind]
            recs.append(Recording(subject=f"hackedge_{row['person']}",
                                  session=os.path.splitext(str(row["file"]))[0].replace("/", "_"),
                                  kind=kind, signal=x, fs=fs, label=label, dataset="hackedge",
                                  is_baseline_session=(row["condition"] == "normal"), extra=dict(extra)))
    if not recs:
        raise RuntimeError(f"No recordings found from {root}/{manifest}")
    return recs
