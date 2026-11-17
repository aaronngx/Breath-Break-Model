"""Windowing + time-domain HRV features + personal-baseline normalization."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import signals as S

FEATURE_NAMES = ("mean_hr", "mean_ibi", "sdnn", "rmssd", "pnn50", "cvnn")


def hrv_features(ibi_ms: np.ndarray) -> dict:
    """Time-domain HRV features that are valid on 60-s windows."""
    ibi = np.asarray(ibi_ms, dtype=float)
    d = np.diff(ibi)
    mean_ibi = ibi.mean()
    sdnn = ibi.std(ddof=1)
    return {
        "mean_hr": 60000.0 / mean_ibi,
        "mean_ibi": mean_ibi,
        "sdnn": sdnn,
        "rmssd": np.sqrt(np.mean(d ** 2)),
        "pnn50": 100.0 * np.mean(np.abs(d) > 50.0),
        "cvnn": sdnn / mean_ibi,
    }


EXTENDED_FEATURE_NAMES = ("mean_hr", "sdnn", "rmssd", "pnn50", "cvnn",
                          "lf", "hf", "lfhf", "resp_hz", "sd1sd2", "dfa_a1")
# how each extended feature is compared with the personal baseline
LOG_RATIO_FEATURES = {"mean_hr", "sdnn", "rmssd", "cvnn", "lf", "hf", "lfhf", "sd1sd2"}


def dfa_alpha1(ibi_ms: np.ndarray) -> float:
    """Short-term detrended fluctuation exponent (scales 4-16 beats)."""
    ibi = np.asarray(ibi_ms, dtype=float)
    y = np.cumsum(ibi - ibi.mean())
    scales, fluct = np.arange(4, 17), []
    for n in scales:
        k = len(y) // n
        if k < 2:
            return np.nan
        seg = y[: k * n].reshape(k, n)
        t = np.arange(n)
        res = [np.mean((s - np.polyval(np.polyfit(t, s, 1), t)) ** 2) for s in seg]
        fluct.append(np.sqrt(np.mean(res)))
    return float(np.polyfit(np.log(scales), np.log(fluct), 1)[0])


def hrv_features_extended(t_beats_s: np.ndarray, ibi_ms: np.ndarray) -> dict:
    """Time + frequency + nonlinear HRV. Needs ~180 s windows (LF needs >= 2-3 min)."""
    from scipy import signal as sps
    ibi = np.asarray(ibi_ms, dtype=float)
    f = hrv_features(ibi)
    f.pop("mean_ibi")
    d = np.diff(ibi)
    tt = np.arange(t_beats_s[0], t_beats_s[-1], 0.25)            # 4 Hz tachogram
    r = sps.detrend(np.interp(tt, t_beats_s, ibi))
    fr, pw = sps.welch(r, fs=4, nperseg=min(256, len(r)))

    def band(lo, hi):
        m = (fr >= lo) & (fr < hi)
        return float(np.trapezoid(pw[m], fr[m]))

    lf, hf = band(0.04, 0.15), band(0.15, 0.40)
    hfm = (fr >= 0.15) & (fr < 0.40)
    sd1 = np.sqrt(0.5) * d.std(ddof=1)
    sd2 = np.sqrt(max(2 * f["sdnn"] ** 2 - sd1 ** 2, 1e-6))
    f.update(lf=lf, hf=hf, lfhf=lf / max(hf, 1e-9), resp_hz=float(fr[hfm][np.argmax(pw[hfm])]),
             sd1sd2=sd1 / sd2, dfa_a1=dfa_alpha1(ibi))
    return f


def iter_windows(n_samples: int, fs: float, window_s: float, step_s: float):
    w, s = int(window_s * fs), int(step_s * fs)
    for start in range(0, n_samples - w + 1, s):
        yield start, start + w


def signal_to_feature_windows(x: np.ndarray, fs: float, kind: str, cfg,
                              meta: dict | None = None) -> pd.DataFrame:
    """Cut a whole recording into windows and compute HRV features per window.

    kind: "ecg" or "ppg". Detects beats once on the full signal (more robust
    than per-window detection), then slices IBIs per window.
    Returns one row per *valid* window with meta columns attached.
    """
    meta = meta or {}
    peaks = S.ecg_rpeaks(x, fs) if kind == "ecg" else S.ppg_peaks(x, fs)
    rows = []
    for start, end in iter_windows(len(x), fs, cfg.window_s, cfg.step_s):
        p = peaks[(peaks >= start) & (peaks < end)]
        ibi, frac = S.clean_ibi(S.peaks_to_ibi_ms(p, fs), cfg)
        if not S.window_ok(ibi, frac, cfg):
            continue
        row = hrv_features(ibi)
        row.update(meta)
        row["t_start_s"] = start / fs
        row["clean_frac"] = frac
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_features(x: np.ndarray, fs: float, kind: str, cfg) -> dict | None:
    """HRV features over the first `baseline_s` seconds (the 2-min rest)."""
    n = int(cfg.baseline_s * fs)
    seg = x[:n]
    peaks = S.ecg_rpeaks(seg, fs) if kind == "ecg" else S.ppg_peaks(seg, fs)
    ibi, frac = S.clean_ibi(S.peaks_to_ibi_ms(peaks, fs), cfg)
    if ibi.size < cfg.min_beats_per_window:
        return None
    return hrv_features(ibi)


def normalize_to_baseline(df: pd.DataFrame, baselines: dict, feature_cols, key="subject") -> pd.DataFrame:
    """Express each feature relative to that person's own baseline.

    Uses log-ratio log(f / f_baseline) so a 20% drop means the same for
    everyone, regardless of their absolute HRV level.
    """
    out = df.copy()
    for subj, base in baselines.items():
        m = out[key] == subj
        if base is None:
            out = out[~m]
            continue
        for c in feature_cols:
            if c == "pnn50":  # can be 0 -> use percentage-point difference
                out.loc[m, c] = (out.loc[m, c].astype(float) - base[c]) / 100.0
                continue
            b = max(base[c], 1e-6)
            out.loc[m, c] = np.log(np.maximum(out.loc[m, c].astype(float), 1e-6) / b)
    return out
