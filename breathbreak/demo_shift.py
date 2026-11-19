"""SYNTHETIC 12-hour shift for demonstrating the passive app. NOT real data - never train or report on it.

Story (band on 07:00, off 19:00):
  07:00-07:30  settling in (becomes the worker's normal for today)
  08:10-08:25  band loose -> wear alert
  09:30-09:45  short break (worker taps Start / End break)
  12:00-12:30  lunch with the band taken off (app infers a break)
  14:30 ->     made-up fatigue drift: resting heart rate creeps up, HRV creeps down
  16:45-17:00  break after the app's suggestion, partial recovery, then drift resumes
  19:00        band taken off
Physical work shows up as short bouts of higher heart rate / lower HRV all day (not fatigue).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

SHIFT_MIN = 720
OFF_TAIL_MIN = 10


def _pulse(tau):
    return np.exp(-0.5 * ((tau - 0.15) / 0.09) ** 2) + 0.35 * np.exp(-0.5 * ((tau - 0.45) / 0.12) ** 2)


def make_demo_shift(start: datetime, fs: float = 50.0, seed: int = 1):
    """-> ppg (np.ndarray), fs, tapped breaks [(start_min, end_min)], per-minute truth dict (for tests)."""
    rng = np.random.default_rng(seed)
    n_min = SHIFT_MIN + OFF_TAIL_MIN
    hr = np.full(n_min, 72.0)
    rm = np.full(n_min, 42.0)

    # physical work: short bouts of exertion all day
    for _ in range(30):
        a = int(rng.integers(35, SHIFT_MIN - 10))
        d = int(rng.integers(3, 10))
        hr[a:a + d] += rng.uniform(10, 20)
        rm[a:a + d] *= 0.6

    taps = [(150, 165), (585, 600)]
    lunch = (300, 330)
    fatigue = np.zeros(n_min)
    fatigue[450:585] = np.linspace(0, 1, 135)
    fatigue[585:600] = 0.5
    fatigue[600:SHIFT_MIN] = np.linspace(0.5, 1.1, SHIFT_MIN - 600)
    hr += 12 * fatigue
    rm *= 1 - 0.38 * np.clip(fatigue, 0, 1.2)
    for a, b in taps + [lunch]:            # resting on a break
        hr[a:b] -= 5
        rm[a:b] *= 1.2

    mode = np.array(["good"] * n_min, dtype=object)
    mode[70:85] = "loose"
    mode[lunch[0]:lunch[1]] = "off"
    mode[SHIFT_MIN:] = "off"

    # beat times
    T = n_min * 60.0
    beats, t = [], 0.0
    while t < T + 2:
        m = min(int(t // 60), n_min - 1)
        s = rm[m] / 1000 / np.sqrt(2)
        ibi = 60.0 / hr[m] + 0.3 * rm[m] / 1000 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, s)
        t += max(ibi, 0.3)
        beats.append(t)
    bt = np.asarray(beats)

    ts = np.arange(int(T * fs)) / fs
    i = np.clip(np.searchsorted(bt, ts, "right") - 1, 1, len(bt) - 1)
    y = _pulse(ts - bt[i]) + _pulse(ts - bt[i - 1])
    y = 0.2 + 0.15 * y + 0.02 * np.sin(2 * np.pi * 0.05 * ts) + rng.normal(0, 0.003, len(ts))

    m_of = (ts // 60).astype(int)
    loose = mode[m_of] == "loose"
    if loose.any():
        k = int(loose.sum())
        motion = np.cumsum(rng.normal(0, 0.02, k))
        y[loose] = 0.2 + 0.15 * motion / (np.abs(motion).max() + 1e-9) + rng.normal(0, 0.04, k)
    off = mode[m_of] == "off"
    y[off] = 0.0005 + rng.normal(0, 0.00003, int(off.sum()))
    return y, fs, taps, {"hr": hr, "rmssd": rm, "mode": mode, "fatigue": fatigue}
