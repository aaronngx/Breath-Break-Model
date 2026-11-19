"""Wrist-band wear check: is the optical (PPG) sensor reading a pulse, and if not, what to adjust.

Used two ways during a shift (band put on at the start, taken off at the end):
  fit_check(x, fs)   first ~20 s after putting the band on -> OK, or the problem + steps to fix it
  scan(x, fs)        the whole shift in 20-s windows -> timeline of good / problem stretches,
                     % of the shift with a usable signal, and when the band was taken off

Window status (from signal_quality.check_ppg):
  good          clear pulse rhythm
  no_contact    flat: band not touching skin / taken off / sensor light off
  too_tight     signal maxed out (clipped): pressed too hard or outside light leaking in
  loose         no clear pulse rhythm or impossible rate: band loose or arm moving
Thresholds come from the HackEdge AFE4950 desk recordings; recheck them on real wrist data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signal_quality import check_ppg

WIN_S, STEP_S = 20.0, 10.0
MIN_BAD_S = 30.0          # problem stretches shorter than this are ignored (a quick arm movement)

STATUS = {
    "flatline": "no_contact",
    "clipped": "too_tight",
    "no_rhythm": "loose",
    "implausible_hr": "loose",
    "ppg_ecg_mismatch": "loose",
}

FIX = {
    "good": ("Signal is good", "The band is reading your pulse. You're set for the shift.", []),
    "no_contact": ("No contact with your skin", "The sensor sees nothing - it is not touching your skin.", [
        "Put the band on the top of your wrist, sensor side against the skin.",
        "Move it 1-2 finger widths up from the wrist bone.",
        "Tighten it one notch so it does not slide.",
        "Still nothing? Restart the band - the sensor light may be off.",
    ]),
    "too_tight": ("Signal is maxed out", "The band is pressing too hard, or outside light is getting in.", [
        "Loosen the band one notch - you should be able to slide a finger under it.",
        "Make sure the sensor is flat on the skin with no gap at the edges.",
        "Keep it out of direct sunlight or strong lamps while checking.",
    ]),
    "loose": ("Pulse not clear", "The band is loose or your arm is moving.", [
        "Tighten the band one notch so it does not move when you turn your wrist.",
        "Rest your arm on the table and keep still for 30 seconds.",
        "Cold hands weaken the signal - warm them up first.",
        "Tattoos or thick hair under the sensor can block it - move the band slightly.",
    ]),
}


@dataclass
class Window:
    start_s: float
    status: str
    pulse_bpm: float | None


def window_status(x, fs) -> tuple[str, float | None]:
    """Status of one stretch of PPG (any length >= ~18 s) -> (status, pulse bpm or None)."""
    return _status(x, fs, min_s=0.9 * min(WIN_S, len(x) / fs))


def _status(x, fs, min_s: float = 0.9 * WIN_S) -> tuple[str, float | None]:
    q = check_ppg(x, fs, min_s=min_s)
    if q.ok:
        return "good", q.metrics.get("pulse_bpm")
    return STATUS.get(q.reason, "loose"), None


def fit_check(x: np.ndarray, fs: float) -> Window:
    """Check the first WIN_S seconds after putting the band on."""
    x = np.asarray(x, float)
    s, bpm = _status(x[: int(WIN_S * fs)], fs)
    return Window(0.0, s, bpm)


def scan(x: np.ndarray, fs: float) -> list[Window]:
    x = np.asarray(x, float)
    n, w, step = len(x), int(WIN_S * fs), int(STEP_S * fs)
    out = []
    for a in range(0, max(n - w, 0) + 1, step):
        s, bpm = _status(x[a:a + w], fs)
        out.append(Window(a / fs, s, bpm))
    return out


def summarize(wins: list[Window], total_s: float) -> dict:
    """-> good %, taken-off time (trailing no_contact), and problem stretches during wear."""
    if not wins:
        return {"good_pct": 0.0, "off_at_s": None, "problems": [], "worn_s": 0.0}
    st = [w.status for w in wins]
    # taken off at the end of the shift = trailing no_contact windows
    k = len(st)
    while k > 0 and st[k - 1] == "no_contact":
        k -= 1
    off_at = wins[k].start_s + STEP_S if 0 < k < len(st) else (0.0 if k == 0 else None)
    worn = wins[:k]
    good_pct = 100.0 * sum(w.status == "good" for w in worn) / max(len(worn), 1)
    problems, i = [], 0
    while i < len(worn):
        if worn[i].status == "good":
            i += 1
            continue
        j = i
        while j + 1 < len(worn) and worn[j + 1].status == worn[i].status:
            j += 1
        a, b = worn[i].start_s, worn[j].start_s + WIN_S
        if b - a >= MIN_BAD_S:
            problems.append({"from_s": a, "to_s": min(b, total_s), "status": worn[i].status})
        i = j + 1
    return {"good_pct": good_pct, "off_at_s": off_at, "problems": problems,
            "worn_s": (off_at if off_at is not None else total_s)}


def make_demo_shift(fs: float = 50.0, seed: int = 0) -> tuple[np.ndarray, float, list]:
    """SYNTHETIC 6-minute 'shift' for showing the timeline: good / loose / good / tight / good / off.
    Not real data; never use it for training or results."""
    rng = np.random.default_rng(seed)
    plan = [("good", 60), ("loose", 60), ("good", 60), ("too_tight", 40), ("good", 80), ("no_contact", 60)]
    parts = []
    for kind, dur in plan:
        t = np.arange(int(dur * fs)) / fs
        beats = np.cumsum(rng.normal(60 / 72, 0.03, int(dur * 72 / 60) + 3))
        pulse = sum(np.exp(-0.5 * ((t - b) / 0.09) ** 2) - 0.35 * np.exp(-0.5 * ((t - b - 0.3) / 0.12) ** 2)
                    for b in beats if b < dur + 1)
        drift = 0.02 * np.sin(2 * np.pi * 0.05 * t + rng.uniform(0, 6))
        if kind == "good":
            y = 0.2 + 0.15 * pulse + drift + rng.normal(0, 0.003, len(t))
        elif kind == "loose":
            motion = np.cumsum(rng.normal(0, 0.02, len(t)))
            motion -= np.convolve(motion, np.ones(25) / 25, "same") * 0.0
            y = 0.2 + 0.01 * pulse + 0.15 * motion / (np.abs(motion).max() + 1e-9) + rng.normal(0, 0.04, len(t))
        elif kind == "too_tight":
            y = np.clip(0.2 + 0.6 * pulse + drift, None, 0.45)
        else:
            y = 0.0005 + rng.normal(0, 0.00003, len(t))
        parts.append(y)
    return np.concatenate(parts), fs, plan
