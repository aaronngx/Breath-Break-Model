"""Read a HackEdge-style folder (data_manifest.csv + raw TI exports) and check every recording.

    python scripts/check_recordings.py data/hackedge

Per recording: signal quality, heart rate and HRV, change versus the same person's
OTHER normal recordings, and whether the HRV re-test trigger fires. If the manifest
row also has wake / slept / pvt_json / recorded_at, the full schedule + PVT model
(models/fatigue_fusion.joblib) gives a fatigue risk and OK / RETEST / ALERT.

Short clips (< 60 s) are scored but flagged: HRV from 30 s is not reliable.
Writes reports/recordings_check.csv
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak import signals as S  # noqa: E402
from breathbreak.data.hackedge import load_hackedge  # noqa: E402
from breathbreak.features import hrv_features  # noqa: E402
from breathbreak.fusion import RETEST_RMSSD_DROP, decide, hrv_wants_retest  # noqa: E402

TRIM_S = 2.0            # drop electrode-settling at start/end
MIN_BEATS = 20
FEATS = ["mean_hr", "sdnn", "rmssd", "pnn50"]


def clip_features(rec):
    x = S.resample(rec.signal, rec.fs, CFG.ecg_fs)
    fs = CFG.ecg_fs
    peaks = S.ecg_rpeaks(x, fs)
    peaks = peaks[(peaks >= TRIM_S * fs) & (peaks < len(x) - TRIM_S * fs)]
    ibi, frac = S.clean_ibi(S.peaks_to_ibi_ms(peaks, fs), CFG)
    q = {"beats": int(ibi.size), "clean_frac": round(frac, 2)}
    if ibi.size < MIN_BEATS or frac < CFG.min_clean_fraction:
        return None, q
    f = hrv_features(ibi)
    return {k: f[k] for k in FEATS}, q


def fusion_verdict(rec, bundle):
    e = rec.extra
    need = ("wake", "slept", "pvt_json", "recorded_at")
    if bundle is None or any(pd.isna(e.get(k)) or str(e.get(k)).strip() == "" for k in need):
        return None, None
    from breathbreak.fusion import predict_fatigue
    from breathbreak.schedule import schedule_features
    when = datetime.strptime(str(e["recorded_at"]), "%Y-%m-%d %H:%M")
    h, m = (int(v) for v in str(e["wake"]).split(":"))
    wake = when.replace(hour=h, minute=m)
    if wake > when:
        wake = wake.replace(day=wake.day - 1)
    feats = schedule_features(wake, when, float(e["slept"]))
    with open(os.path.join(e["root"], str(e["pvt_json"]))) as f:
        feats.update(json.load(f))
    p = predict_fatigue(bundle, feats)
    return p, decide(p, bundle["alert_threshold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/hackedge")
    ap.add_argument("--manifest", default="data_manifest.csv")
    ap.add_argument("--model", default="models/fatigue_fusion.joblib")
    a = ap.parse_args()

    recs = load_hackedge(a.root, CFG, a.manifest)
    bundle = None
    if os.path.exists(a.model):
        from breathbreak.fusion import load_fusion
        bundle = load_fusion(a.model)

    rows = []
    for r in recs:
        f, q = clip_features(r)
        row = {"person": r.subject.replace("hackedge_", ""), "file": r.extra["file"],
               "condition": r.extra["condition"], "kss": r.extra["kss"],
               "seconds": round(r.extra["duration_s"]), **q, "usable": f is not None}
        if f:
            row.update({k: round(v, 1) for k, v in f.items()})
        p, verdict = fusion_verdict(r, bundle)
        row["fatigue_risk"] = None if p is None else round(p, 2)
        row["verdict"] = verdict or "needs wake/slept/PVT"
        rows.append(row)
    t = pd.DataFrame(rows)

    # change versus the same person's OTHER normal recordings (median), like a personal baseline
    t["rmssd_change_pct"], t["hrv_retest"] = np.nan, None
    for i, r in t[t.usable].iterrows():
        base = t[(t.person == r.person) & (t.condition == "normal") & t.usable & (t.file != r.file)]
        if base.empty:
            continue
        ratio = r.rmssd / base.rmssd.median()
        t.loc[i, "rmssd_change_pct"] = round(100 * (ratio - 1))
        t.loc[i, "hrv_retest"] = hrv_wants_retest(np.log(ratio))
        t.loc[i, "hr_change_bpm"] = round(r.mean_hr - base.mean_hr.median(), 1)

    pd.set_option("display.width", 220)
    cols = ["person", "file", "condition", "seconds", "beats", "usable", "mean_hr", "rmssd",
            "hr_change_bpm", "rmssd_change_pct", "hrv_retest", "verdict"]
    print(t[[c for c in cols if c in t]].to_string(index=False))

    short = (t.seconds < 60).sum()
    print(f"\n{len(t)} recordings read, {t.usable.sum()} usable; {short} shorter than 60 s (HRV unreliable).")
    print(f"HRV re-test trigger = RMSSD >= {RETEST_RMSSD_DROP:.0%} below the person's other normal recordings.")
    comp = t.dropna(subset=["rmssd_change_pct"])
    if not comp.empty:
        print("\nBy person and condition (median change vs own normals):")
        print(comp.groupby(["person", "condition"]).agg(
            recordings=("file", "size"), hr_change_bpm=("hr_change_bpm", "median"),
            rmssd_change_pct=("rmssd_change_pct", "median"),
            retest_triggered=("hrv_retest", "sum")).to_string())
    if (t.verdict == "needs wake/slept/PVT").all():
        print("\nNo fatigue verdicts: the manifest has no wake / slept / pvt_json / recorded_at columns.\n"
              "Add them for new recordings to get the schedule + PVT fatigue risk.")
    os.makedirs("reports", exist_ok=True)
    t.to_csv("reports/recordings_check.csv", index=False)
    print("Saved reports/recordings_check.csv")


if __name__ == "__main__":
    main()
