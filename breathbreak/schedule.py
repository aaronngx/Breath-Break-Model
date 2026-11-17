"""Schedule-based fatigue prior: a simple two-process model of sleepiness.

Process S (homeostatic sleep pressure) rises exponentially while awake and
decays during sleep; Process C (circadian) makes sleepiness peak in the early
morning. Time constants follow the classic two-process model
(Borbely 1982; Daan, Beersma & Borbely 1984): rise 18.2 h, decay 4.2 h.

The device only needs three inputs per check:
    hours awake, hours slept last night, and the clock time.
These features are the *prior*; the reaction-time test (pvt.py) confirms.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

TAU_RISE_H = 18.2              # sleep pressure build-up while awake
TAU_DECAY_H = 4.2              # sleep pressure dissipation while asleep
CIRCADIAN_PEAK_SLEEPINESS_H = 5.0   # clock hour of maximum circadian sleepiness (~04-06 h)
HABITUAL_WAKE_H, HABITUAL_SLEEP_H = 16.0, 8.0


def _steady_state_at_wake() -> float:
    """Sleep pressure on waking after a long run of regular 16 h awake / 8 h asleep days."""
    s = 0.5
    for _ in range(50):
        s_bed = 1 - (1 - s) * np.exp(-HABITUAL_WAKE_H / TAU_RISE_H)
        s = s_bed * np.exp(-HABITUAL_SLEEP_H / TAU_DECAY_H)
    return float(s)


def sleep_pressure(hours_awake: float, sleep_hours: float = 8.0) -> float:
    """Process S (0..1) after `hours_awake` hours, given last night's sleep length."""
    s_wake0 = _steady_state_at_wake()
    s_bed = 1 - (1 - s_wake0) * np.exp(-HABITUAL_WAKE_H / TAU_RISE_H)   # pressure at last bedtime
    s_wake = s_bed * np.exp(-max(sleep_hours, 0.0) / TAU_DECAY_H)      # after last night's sleep
    return float(1 - (1 - s_wake) * np.exp(-max(hours_awake, 0.0) / TAU_RISE_H))


def circadian_sleepiness(clock_hour: float) -> float:
    """Process C as sleepiness (+1 at ~05:00, -1 at ~17:00)."""
    return float(np.cos(2 * np.pi * (clock_hour - CIRCADIAN_PEAK_SLEEPINESS_H) / 24.0))


def schedule_features(wake_time: datetime, now: datetime, sleep_hours: float = 8.0) -> dict:
    hours_awake = (now - wake_time).total_seconds() / 3600.0
    clock = now.hour + now.minute / 60.0
    return {"hours_awake": hours_awake,
            "sleep_pressure": sleep_pressure(hours_awake, sleep_hours),
            "circadian": circadian_sleepiness(clock)}


SCHEDULE_FEATURES = ("sleep_pressure", "circadian")
