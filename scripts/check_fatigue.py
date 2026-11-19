"""Shift fatigue check for the SK-AM62 (or any PC): schedule check-in -> fatigue risk.

    python scripts/check_fatigue.py --wake 06:30 --slept 7                 # schedule only (default, instant)
    python scripts/check_fatigue.py --wake 06:30 --slept 7 --pvt           # + 3-min PVT (more accurate)
    python scripts/check_fatigue.py --wake 06:30 --slept 7 --pvt-json last_pvt.json   # reuse a saved PVT

Default = schedule only (no PVT, no waiting). On DROZY (leave-one-subject-out) schedule only
scores AUC 0.79 vs 0.89 with the PVT, and it relies entirely on what the user reports.
Output is fatigue risk only (no stress score). Needs models/fatigue_schedule.joblib and, for
--pvt, models/fatigue_fusion.joblib (both made by scripts/evaluate_fusion.py).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from breathbreak.fusion import decide, load_fusion, predict_fatigue  # noqa: E402
from breathbreak.pvt import run_pvt  # noqa: E402
from breathbreak.schedule import schedule_features  # noqa: E402


def parse_wake(hhmm: str, now: datetime) -> datetime:
    h, m = (int(v) for v in hhmm.split(":"))
    wake = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return wake if wake <= now else wake - timedelta(days=1)   # woke "yesterday" (night shift)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wake", required=True, help="wake-up time today, HH:MM")
    ap.add_argument("--slept", type=float, required=True, help="hours slept last night")
    ap.add_argument("--pvt", action="store_true", help="also run the 3-min PVT (more accurate, takes 3 min)")
    ap.add_argument("--minutes", type=float, default=3.0, help="PVT length (with --pvt)")
    ap.add_argument("--pvt-json", help="use a saved PVT result (implies --pvt)")
    ap.add_argument("--model", help="override the model file")
    a = ap.parse_args()

    use_pvt = a.pvt or bool(a.pvt_json)
    bundle = load_fusion(a.model or ("models/fatigue_fusion.joblib" if use_pvt else "models/fatigue_schedule.joblib"))
    now = datetime.now()
    feats = schedule_features(parse_wake(a.wake, now), now, a.slept)
    line = f"\nAwake {feats['hours_awake']:.1f} h | slept {a.slept:g} h | clock {now:%H:%M}"
    if use_pvt:
        if a.pvt_json:
            with open(a.pvt_json) as f:
                pvt = json.load(f)
        else:
            pvt = run_pvt(a.minutes)
        feats.update(pvt)
        line += f" | PVT lapses {pvt['pvt_lapses']} of {pvt['pvt_n']} | median RT {pvt['pvt_median_rt']:.0f} ms"
    p = predict_fatigue(bundle, feats)
    verdict = decide(p, bundle["alert_threshold"])
    print(line)
    print(f"Fatigue risk: {p:.2f}  ->  {verdict}" + ("" if use_pvt else "   (schedule only, from what you reported)"))
    if verdict == "ALERT":
        print("Take a break now" + (", then re-test." if use_pvt else "."))


if __name__ == "__main__":
    main()
