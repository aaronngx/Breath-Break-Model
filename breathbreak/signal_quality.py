"""Signal-quality gate: decide whether ECG / PPG data is usable, and say why not.

Data that fails is flagged and NOT used for features or scores. The user gets a
plain instruction (re-attach electrode, put finger on sensor, hold still...)
instead of a number computed from noise.

Checks, cheapest first:
  too_short        not enough signal to judge
  flatline         no signal at all (sensor not worn / not touching skin / LED off)
  clipped          flat plateaus at the signal's limits (gain too high, pressing too hard)
  no_rhythm        no dominant heartbeat frequency (PPG)
  implausible_hr   heart rate outside 30-200 bpm
  too_few_beats    not enough clean beats
  irregular_beats  many beat intervals rejected (motion, loose contact)
  beat_shape       beats do not look like each other (motion, loose electrode)
  ppg_ecg_mismatch optical pulse rate disagrees with the ECG heart rate

Thresholds were set from the HackEdge AFE4950 recordings (10 files) and generic
physiology; recheck them when more data is recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

from . import signals as S

MESSAGES = {
    "too_short": "Recording too short - record at least {min_s:.0f} s.",
    "flatline": "No signal - sensor not worn or not touching the skin (or channel switched off).",
    "clipped": "Signal is clipped at the sensor limit - reduce gain or press more lightly.",
    "no_rhythm": "No pulse rhythm found - put your finger fully on the optical sensor and keep still.",
    "implausible_hr": "Heart rate outside 30-200 bpm - this is noise, not a heartbeat.",
    "too_few_beats": "Too few clean heartbeats - hold still and re-record.",
    "irregular_beats": "Many beats rejected (movement or loose contact) - hold still and re-record.",
    "beat_shape": "Heartbeats look inconsistent (movement or loose electrode) - re-attach and re-record.",
    "ppg_ecg_mismatch": "Optical pulse does not match the ECG - optical sensor not reading correctly.",
}

# thresholds
ECG_MIN_PTP_V = 1e-3          # HackEdge ECG is 60-80 mV peak-to-peak; < 1 mV = nothing there
PPG_MIN_PTP_V = 5e-3          # real PPG 130-1400 mV; noise floor ~0.13 mV
CLIP_MIN_FRACTION = 0.005     # > 0.5 % of samples sitting on a flat plateau at min or max
CLIP_MIN_RUN_S = 0.01         # plateau must last >= 10 ms to count
PPG_MIN_PEAK_RATIO = 5.0      # dominant pulse power / median power in 0.7-3 Hz (real 7-21, noise 1.5-3.4)
ECG_MIN_TEMPLATE_CORR = 0.80  # a beat is "good" if it correlates >= 0.8 with the average beat
ECG_MIN_GOOD_BEATS = 0.80     # ... and at least 80 % of beats must be good
MAX_HR_DISAGREEMENT_BPM = 8.0
MIN_CLEAN_BEATS = 15
MIN_CLEAN_FRACTION = 0.80


@dataclass
class Quality:
    signal: str
    ok: bool
    issues: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def reason(self) -> str:
        return self.issues[0] if self.issues else "ok"

    def message(self) -> str:
        if self.ok:
            return f"{self.signal.upper()} signal OK"
        return MESSAGES[self.reason].format(**self.metrics)


def _plateau_fraction(x: np.ndarray, fs: float) -> float:
    """Share of samples in flat runs (>= CLIP_MIN_RUN_S) at the signal's min or max."""
    lo, hi = x.min(), x.max()
    tol = 1e-3 * max(hi - lo, 1e-12)
    at_edge = (x <= lo + tol) | (x >= hi - tol)
    if not at_edge.any():
        return 0.0
    edges = np.flatnonzero(np.diff(np.r_[0, at_edge.astype(int), 0]))
    runs = edges[1::2] - edges[::2]
    long_runs = runs[runs >= max(2, int(CLIP_MIN_RUN_S * fs))]
    return float(long_runs.sum() / len(x))


def _basic(x, fs, kind, min_ptp, min_s):
    m = {"seconds": round(len(x) / fs, 1), "min_s": min_s, "ptp_mV": round(float(np.ptp(x)) * 1000, 3)}
    if len(x) < min_s * fs:
        return ["too_short"], m
    if not np.isfinite(x).all() or np.ptp(x) < min_ptp:
        return ["flatline"], m
    m["clipped_pct"] = round(100 * _plateau_fraction(x, fs), 2)
    if m["clipped_pct"] > 100 * CLIP_MIN_FRACTION:
        return ["clipped"], m
    return [], m


def _good_beat_fraction(x, fs, peaks):
    """Share of beats whose shape correlates >= ECG_MIN_TEMPLATE_CORR with the median beat."""
    rr = np.median(np.diff(peaks)) if len(peaks) > 2 else 0.8 * fs
    pre, post = int(min(0.2 * fs, 0.3 * rr)), int(min(0.4 * fs, 0.6 * rr))   # never reach the next beat
    beats = np.array([x[p - pre:p + post] for p in peaks if p - pre >= 0 and p + post <= len(x)])
    if len(beats) < 5:
        return np.nan
    tmpl = np.median(beats, axis=0)
    corr = np.array([np.corrcoef(b, tmpl)[0, 1] for b in beats])
    return float(np.mean(corr >= ECG_MIN_TEMPLATE_CORR))


def check_ecg(x: np.ndarray, fs: float, cfg, min_s: float = 20.0) -> Quality:
    x = np.asarray(x, dtype=float)
    issues, m = _basic(x, fs, "ecg", ECG_MIN_PTP_V, min_s)
    if issues:
        return Quality("ecg", False, issues, m)
    x250 = S.resample(x, fs, cfg.ecg_fs)
    peaks = S.ecg_rpeaks(x250, cfg.ecg_fs)
    ibi, frac = S.clean_ibi(S.peaks_to_ibi_ms(peaks, cfg.ecg_fs), cfg)
    m.update(beats=int(ibi.size), clean_frac=round(frac, 2))
    if ibi.size:
        m["hr_bpm"] = round(60000.0 / float(np.median(ibi)), 1)
    clean = S.bandpass(x250, cfg.ecg_fs, 0.5, 40)
    m["good_beats"] = round(_good_beat_fraction(clean, cfg.ecg_fs, peaks), 3)
    if ibi.size and not 30 <= m["hr_bpm"] <= 200:
        issues.append("implausible_hr")
    if ibi.size < MIN_CLEAN_BEATS:
        issues.append("too_few_beats")
    if frac < MIN_CLEAN_FRACTION:
        issues.append("irregular_beats")
    if not np.isfinite(m["good_beats"]) or m["good_beats"] < ECG_MIN_GOOD_BEATS:
        issues.append("beat_shape")
    return Quality("ecg", not issues, issues, m)


def check_ppg(x: np.ndarray, fs: float, ecg_hr_bpm: float | None = None, min_s: float = 20.0) -> Quality:
    x = np.asarray(x, dtype=float)
    issues, m = _basic(x, fs, "ppg", PPG_MIN_PTP_V, min_s)
    if issues:
        return Quality("ppg", False, issues, m)
    b, a = sps.butter(2, [0.7 / (fs / 2), 3.0 / (fs / 2)], btype="band")
    y = sps.filtfilt(b, a, x - x.mean())
    f, pw = sps.welch(y, fs=fs, nperseg=min(len(y), int(fs * 8)))
    band = (f >= 0.7) & (f <= 3.0)
    k = int(np.argmax(pw[band]))
    m["pulse_bpm"] = round(float(f[band][k]) * 60, 1)
    m["peak_ratio"] = round(float(pw[band][k] / np.median(pw[band])), 1)
    if m["peak_ratio"] < PPG_MIN_PEAK_RATIO:
        issues.append("no_rhythm")
    elif ecg_hr_bpm is not None and abs(m["pulse_bpm"] - ecg_hr_bpm) > MAX_HR_DISAGREEMENT_BPM:
        issues.append("ppg_ecg_mismatch")
    return Quality("ppg", not issues, issues, m)
