"""Realtime scorer for the device (runs on the SK-AM62 or a laptop).

Usage:
    scorer = RealtimeScorer.from_file("models/model_a.joblib", input_fs=500)
    for chunk in stream_from_afe4950():          # any size numpy array of samples
        for event in scorer.push(chunk):
            print(event)                         # dicts: baseline_ready / window / ALERT

Flow: first 2 min -> personal baseline. Then every 60 s -> HRV features ->
normalize to baseline -> P(fatigued) -> smoothed -> alert after N high windows.
Works with Model A bundles (HRV features from ECG or PPG).
"""
from __future__ import annotations

import numpy as np

from . import features as F
from . import signals as S
from .models import load_bundle


class RealtimeScorer:
    def __init__(self, bundle: dict, input_fs: float):
        self.b = bundle
        self.cfg = bundle["cfg"]
        self.kind = bundle["kind"]                       # "ecg" or "ppg"
        self.fs_in = input_fs
        self.fs = self.cfg.ecg_fs if self.kind == "ecg" else self.cfg.ppg_fs
        self.buf = np.array([])
        self.baseline = None
        self.ema = None
        self.high_streak = 0
        self.t = 0.0                                      # seconds of signal consumed

    @classmethod
    def from_file(cls, path: str, input_fs: float):
        return cls(load_bundle(path), input_fs)

    def _features(self, seg):
        peaks = S.ecg_rpeaks(seg, self.fs) if self.kind == "ecg" else S.ppg_peaks(seg, self.fs)
        ibi, frac = S.clean_ibi(S.peaks_to_ibi_ms(peaks, self.fs), self.cfg)
        if not S.window_ok(ibi, frac, self.cfg):
            return None, frac
        return F.hrv_features(ibi), frac

    def _normalize(self, f: dict) -> np.ndarray:
        out = []
        for c in self.b["feature_cols"]:
            if not self.cfg.normalize_to_baseline:
                out.append(f[c])
            elif c == "pnn50":
                out.append((f[c] - self.baseline[c]) / 100.0)
            else:
                out.append(np.log(max(f[c], 1e-6) / max(self.baseline[c], 1e-6)))
        return np.asarray(out)[None, :]

    def push(self, samples: np.ndarray):
        events = []
        self.buf = np.r_[self.buf, S.resample(np.asarray(samples, float), self.fs_in, self.fs)]
        need_base = int(self.cfg.baseline_s * self.fs)
        win = int(self.cfg.window_s * self.fs)

        if self.baseline is None and len(self.buf) >= need_base:
            f, frac = self._features(self.buf[:need_base])
            self.buf = self.buf[need_base:]
            self.t += self.cfg.baseline_s
            if f is None:
                events.append({"type": "baseline_failed", "t_s": self.t, "clean_frac": frac,
                               "msg": "Signal too noisy - hold still / touch the pad again"})
            else:
                self.baseline = f
                events.append({"type": "baseline_ready", "t_s": self.t, "baseline": f})

        while self.baseline is not None and len(self.buf) >= win:
            seg, self.buf = self.buf[:win], self.buf[win:]
            self.t += self.cfg.window_s
            f, frac = self._features(seg)
            if f is None:                                  # motion / poor contact: skip, don't guess
                events.append({"type": "window_skipped", "t_s": self.t, "clean_frac": frac})
                continue
            p = float(self.b["model"].predict_proba(self._normalize(f))[0, 1])
            a = self.cfg.ema_alpha
            self.ema = p if self.ema is None else a * p + (1 - a) * self.ema
            self.high_streak = self.high_streak + 1 if self.ema >= self.cfg.alert_threshold else 0
            ev = {"type": "window", "t_s": self.t, "p_fatigued": p, "p_smoothed": self.ema,
                  "hr": f["mean_hr"], "rmssd": f["rmssd"]}
            events.append(ev)
            if self.high_streak == self.cfg.alert_consecutive:
                events.append({"type": "ALERT", "t_s": self.t,
                               "msg": "Signs of fatigue - time for a short breathing break"})
        return events
