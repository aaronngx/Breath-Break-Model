"""Passive shift monitor: the worker puts the band on and nothing else. No questions, no tests.

    epochs = epoch_table(ppg, fs_ppg, ecg=None, fs_ecg=None)        # every minute: signal status, HR, HRV
    breaks = infer_breaks(epochs, taps)                               # Start/End-break taps + band-off gaps
    run    = monitor(epochs, breaks, start, schedule_bundle, history_baseline=None)
    run["decisions"]   every 5 min: level (0 green / 1 yellow / 2 red / None), risk, reasons, prompt
    run["summary"]     end-of-shift numbers for the worker and for HR

What is learned and what is a rule
  learned  schedule risk - models/fatigue_schedule.joblib (DROZY, leave-one-person-out AUC 0.79).
           With no questions we ASSUME the worker woke 1 h before putting the band on, after 7 h sleep.
  rule     heart drift vs the worker's own normal. NOT VALIDATED: placeholder until pilot data trains it.
           "Normal" = the first 30 min of this shift (or past shifts if today's start was unreadable).
           Physical work also raises heart rate, so drift uses QUIET moments only: the calmer half of the last
           20 min (lower heart rate), median HR and HRV (fatigue lifts the floor; lifting a box doesn't).
           After a break the window restarts, so a break 'resets' the heart reading.
  rule     2 h without a break -> suggest a short break (accident risk climbs with time since a break,
           Tucker et al. 2003).
All rule constants are in the block below so the pilot can re-tune them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import scipy.signal as sps

from . import CFG
from . import signals as S
from .schedule import schedule_features
from .wear_check import window_status

# ---------------------------------------------------------------- tune on pilot data
EPOCH_S = 60
DECIDE_EVERY_MIN = 5
BASELINE_MIN = 30            # first 30 min of the shift = today's normal
MIN_BASE_EPOCHS = 15
LOOKBACK_MIN = 20
MIN_LOOKBACK_EPOCHS = 8
HRV_DROP_FULL = 0.30         # quiet HRV 30 % below normal  -> drift index 1.0
HR_RISE_FULL = 12.0          # quiet heart rate +12 bpm      -> drift index 1.0
RED, YELLOW = 0.60, 0.45     # same cut-offs as the schedule model's alert threshold
BREAK_DUE_MIN = 120
PROMPT_REPEAT_MIN = 60       # don't nag: repeat a suggestion at most every 60 min, unless it gets worse
PERSIST_MIN = 10             # a level must hold for 10 min before it triggers a suggestion
POST_BREAK_QUIET_MIN = 20    # no suggestion in the first 20 min after a break
BANDOFF_BREAK_MIN = (5, 60)  # band off mid-shift for 5-60 min = a break
RECOVERED_HRV_UP = 0.10
RECOVERED_HR_DOWN = 5.0
ASSUMED_WAKE_BEFORE_H = 1.0
ASSUMED_SLEEP_H = 7.0

BREAK_KINDS = {
    "micro": ("5-minute reset", 5, [
        "Stop and step away from your task.",
        "Slow breathing for 1 minute: breathe in for 4 seconds, out for 6 seconds.",
        "Drink a glass of water.",
        "Stand, stretch your back and shoulders, and look at something far away.",
    ]),
    "recovery": ("15-minute recovery break", 15, [
        "Stop what you are doing. Do not drive or run machines until the break is over.",
        "If you can lie or sit somewhere quiet: nap for 10 minutes (set an alarm, no more than 20).",
        "If you can't nap: sit, close your eyes and do the slow breathing for 5 minutes.",
        "Drink water and eat something light.",
        "Before going back: stand up, walk for 2 minutes in bright light.",
    ]),
}


# ---------------------------------------------------------------- per-minute signal
def _ppg_beat_times(x, fs):
    y = S.bandpass(np.asarray(x, float) - np.mean(x), fs, 0.5, 4.0)
    pk, _ = sps.find_peaks(y, distance=max(1, int(0.33 * fs)), prominence=0.3 * np.std(y))
    pk = pk[(pk > 0) & (pk < len(y) - 1)]
    a, b, c = y[pk - 1], y[pk], y[pk + 1]          # parabolic refinement: sub-sample peak timing
    den = a - 2 * b + c
    off = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    return (pk + np.clip(off, -0.5, 0.5)) / fs


def _ecg_beat_times(x, fs):
    z = S.resample(np.asarray(x, float), fs, CFG.ecg_fs)
    return S.ecg_rpeaks(z, CFG.ecg_fs) / CFG.ecg_fs


def _hr_rmssd(beat_t):
    ibi, frac = S.clean_ibi(np.diff(beat_t) * 1000.0, CFG)
    if ibi.size < 20 or frac < 0.8:
        return None, None
    d = np.diff(ibi)
    d = d[np.abs(d) <= max(5 * np.median(np.abs(d)), 20.0)]   # drop beat-timing glitches (they inflate RMSSD)
    return 60000.0 / ibi.mean(), float(np.sqrt(np.mean(d ** 2)))


def epoch_table(ppg, fs_ppg, ecg=None, fs_ecg=None) -> pd.DataFrame:
    """One row per minute: status (good / loose / too_tight / no_contact), hr, rmssd, source."""
    n = int(len(ppg) / fs_ppg // EPOCH_S)
    rows = []
    for m in range(n):
        a, b = int(m * EPOCH_S * fs_ppg), int((m + 1) * EPOCH_S * fs_ppg)
        status, _ = window_status(ppg[a:b], fs_ppg)
        hr = rm = None
        src = ""
        if ecg is not None:
            ea, eb = int(m * EPOCH_S * fs_ecg), int((m + 1) * EPOCH_S * fs_ecg)
            hr, rm = _hr_rmssd(_ecg_beat_times(ecg[ea:eb], fs_ecg)) if eb <= len(ecg) else (None, None)
            src = "ecg" if hr is not None else ""
        if hr is None and status == "good":
            hr, rm = _hr_rmssd(_ppg_beat_times(ppg[a:b], fs_ppg))
            src = "ppg" if hr is not None else ""
        rows.append({"t_min": m, "status": status, "hr": hr, "rmssd": rm, "source": src})
    cols = ["t_min", "status", "hr", "rmssd", "source"]
    return pd.DataFrame(rows, columns=cols).astype({"hr": float, "rmssd": float})


# ---------------------------------------------------------------- breaks
@dataclass
class Break:
    start_min: int
    end_min: int
    how: str                      # "tap" or "band off"
    after_prompt: bool = False
    recovery: dict = field(default_factory=dict)


def shift_end_min(epochs) -> int:
    """Band taken off for good = trailing no_contact minutes."""
    st = epochs.status.to_numpy()
    k = len(st)
    while k > 0 and st[k - 1] == "no_contact":
        k -= 1
    return k


def infer_breaks(epochs, taps=()) -> list[Break]:
    out = [Break(int(a), int(b), "tap") for a, b in taps]
    st, end = epochs.status.to_numpy(), shift_end_min(epochs)
    i = 0
    while i < end:
        if st[i] != "no_contact":
            i += 1
            continue
        j = i
        while j < end and st[j] == "no_contact":
            j += 1
        if BANDOFF_BREAK_MIN[0] <= j - i <= BANDOFF_BREAK_MIN[1] and i > 0 and \
                not any(b.start_min <= i < b.end_min for b in out):
            out.append(Break(i, j, "band off"))
        i = j
    return sorted(out, key=lambda b: b.start_min)


# ---------------------------------------------------------------- decisions
def _quiet(ep):
    """Quiet-moment heart: the calmer half of the minutes (lower heart rate), median HR and median HRV."""
    q = ep[ep.hr <= ep.hr.median()]
    return float(q.hr.median()), float(q.rmssd.median())


def _usable(epochs, breaks):
    ok = (epochs.status == "good") & epochs.hr.notna()
    for b in breaks:
        ok &= ~epochs.t_min.between(b.start_min, b.end_min - 1)
    return epochs[ok]


def baseline(epochs, breaks, history=None) -> dict | None:
    use = _usable(epochs, breaks)
    first = use[use.t_min < BASELINE_MIN]
    if len(first) >= MIN_BASE_EPOCHS:
        hr, rm = _quiet(first)
        return {"hr": hr, "rmssd": rm, "from": "first 30 min of this shift"}
    early = use[use.t_min < 2 * BASELINE_MIN]
    if len(early) >= MIN_BASE_EPOCHS:
        hr, rm = _quiet(early.head(BASELINE_MIN))
        return {"hr": hr, "rmssd": rm, "from": "first hour of this shift (start was unreadable)"}
    if history:
        return {**history, "from": "your past shifts"}
    return None


def _schedule_risk(bundle, start, t_min):
    from .fusion import predict_fatigue
    now = start + timedelta(minutes=t_min)
    wake = start - timedelta(hours=ASSUMED_WAKE_BEFORE_H)
    return predict_fatigue(bundle, schedule_features(wake, now, ASSUMED_SLEEP_H))


def _level(r):
    return 2 if r >= RED else 1 if r >= YELLOW else 0


def monitor(epochs, breaks, start: datetime, bundle, history_baseline=None) -> dict:
    end = shift_end_min(epochs)
    base = baseline(epochs, breaks, history_baseline)
    use = _usable(epochs, breaks)
    rows, last_prompt = [], None
    for T in range(DECIDE_EVERY_MIN, end + 1, DECIDE_EVERY_MIN):
        on_break = next((b for b in breaks if b.start_min < T <= b.end_min), None)
        done = [b for b in breaks if b.end_min <= T]
        last_end = max((b.end_min for b in done), default=0)
        since_break = T - last_end
        recent = epochs[(epochs.t_min >= T - 5) & (epochs.t_min < T)]
        band_ok = bool((recent.status == "good").mean() >= 0.6) if len(recent) else False
        band_issue = None if band_ok or on_break else recent.status[recent.status != "good"].mode().iat[0] \
            if (recent.status != "good").any() else None

        sched = _schedule_risk(bundle, start, T)
        win = use[(use.t_min >= max(T - LOOKBACK_MIN, last_end)) & (use.t_min < T)]
        drift = hr_q = rm_q = None
        if base is not None and len(win) >= MIN_LOOKBACK_EPOCHS and T > BASELINE_MIN:
            hr_q, rm_q = _quiet(win)
            drift = max(0.0, (1 - rm_q / base["rmssd"]) / HRV_DROP_FULL, (hr_q - base["hr"]) / HR_RISE_FULL)
        heart = None if drift is None else float(np.clip(0.10 + 0.5 * drift, 0, 0.95))
        risk = max(sched, heart if heart is not None else 0.0)
        level = _level(risk)
        reasons = []
        if heart is not None and _level(heart) >= 1:
            parts = []
            if hr_q - base["hr"] >= 3:
                parts.append(f"resting heart rate is {hr_q - base['hr']:.0f} beats per minute higher")
            if rm_q / base["rmssd"] <= 0.9:
                parts.append(f"heart-rate variability is {100 * (1 - rm_q / base['rmssd']):.0f}% lower")
            reasons.append("Your " + " and ".join(parts) + " than at the start of your shift.")
        if _level(sched) >= 1:
            reasons.append(f"{T / 60:.1f} hours on shift at {(start + timedelta(minutes=T)):%H:%M}.")
        if since_break >= BREAK_DUE_MIN and not on_break:
            level = max(level, 1)
            reasons.append(f"{since_break / 60:.1f} hours without a break.")

        prompt = None
        need = PERSIST_MIN // DECIDE_EVERY_MIN - 1          # earlier decisions that must also be >= yellow
        held = need <= 0 or (len(rows) >= need and all(r["level"] >= 1 for r in rows[-need:]))
        settled = T - last_end >= POST_BREAK_QUIET_MIN or last_end == 0
        if level >= 1 and not on_break and held and settled:
            fresh = last_prompt is None or last_prompt["t"] < last_end
            worse = last_prompt is not None and level > last_prompt["level"]
            due = last_prompt is not None and T - last_prompt["t"] >= PROMPT_REPEAT_MIN
            if fresh or worse or due:
                prompt = "recovery" if level == 2 else "micro"
                last_prompt = {"t": T, "level": level, "kind": prompt}
        rows.append({"t_min": T, "time": start + timedelta(minutes=T), "level": level, "risk": round(risk, 3),
                     "schedule_risk": round(sched, 3), "heart_risk": None if heart is None else round(heart, 3),
                     "hr_quiet": hr_q, "rmssd_quiet": rm_q, "since_break_min": since_break,
                     "on_break": on_break is not None, "band_ok": band_ok, "band_issue": band_issue,
                     "prompt": prompt, "reasons": reasons})
    dec = pd.DataFrame(rows)
    if dec.empty:                       # recording shorter than one decision step
        return {"decisions": dec, "breaks": breaks, "baseline": base, "end_min": end, "summary": None}
    for b in breaks:
        prior = dec[(dec.t_min <= b.start_min) & (dec.t_min > b.start_min - 30) & dec.prompt.notna()]
        b.after_prompt = len(prior) > 0
        b.recovery = recovery(epochs, b)
    return {"decisions": dec, "breaks": breaks, "baseline": base, "end_min": end,
            "summary": summarize(dec, breaks, epochs, end, start)}


def recovery(epochs, b: Break) -> dict:
    """Quiet heart before vs at the end of the break (only when the band was worn during it)."""
    if b.how == "band off":                       # no signal during the break: can't tell how it went
        return {"measured": False}
    good = epochs[(epochs.status == "good") & epochs.hr.notna()]
    pre = good[good.t_min.between(b.start_min - 15, b.start_min - 1)]
    post = good[good.t_min.between(b.end_min - 5, b.end_min - 1)]
    if len(pre) < 5 or len(post) < 3:
        return {"measured": False}
    hr0, rm0 = _quiet(pre)
    hr1, rm1 = _quiet(post)
    up, down = rm1 / rm0 - 1, hr0 - hr1
    return {"measured": True, "hrv_change_pct": round(100 * up), "hr_change_bpm": round(-down, 1),
            "recovered": bool(up >= RECOVERED_HRV_UP or down >= RECOVERED_HR_DOWN)}


def summarize(dec, breaks, epochs, end, start) -> dict:
    worn = epochs[epochs.t_min < end]
    prompts = dec[dec.prompt.notna()]
    measured = [b for b in breaks if b.recovery.get("measured")]
    return {
        "shift_start": start.isoformat(timespec="minutes"),
        "hours_worn": round(end / 60, 2),
        "good_signal_pct": round(100 * float((worn.status == "good").mean()) if len(worn) else 0.0),
        "breaks_suggested": int(len(prompts)),
        "breaks_taken": len(breaks),
        "breaks_after_suggestion": sum(b.after_prompt for b in breaks),
        "longest_without_break_h": round(float(dec.since_break_min.max()) / 60, 2) if len(dec) else 0.0,
        "min_yellow": int((dec.level == 1).sum() * DECIDE_EVERY_MIN),
        "min_red": int((dec.level == 2).sum() * DECIDE_EVERY_MIN),
        "peak_level": int(dec.level.max()) if len(dec) else 0,
        "breaks_recovered": sum(b.recovery.get("recovered", False) for b in measured),
        "breaks_measured": len(measured),
    }
