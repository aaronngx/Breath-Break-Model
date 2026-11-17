"""Evaluate the fusion design on DROZY (sleep loss) and WESAD (stress).

Pre-declared comparisons (nothing tuned on the results):
  main set  (every DROZY session with a PVT + KSS):
      schedule only | PVT only (first 3 min) | schedule + PVT
  HRV set   (sessions that also have baseline-normalised 3-min HRV):
      schedule + PVT | HRV only | schedule + PVT + HRV
  HRV retest trigger: how often RMSSD drops >= 20 % below baseline
      in DROZY (alert vs fatigued) and WESAD (relaxed vs stressed).

Every score is leave-one-subject-out at the session level; 95 % CIs bootstrap over subjects.
Schedule assumption (DROZY does not record it): woke 07:00 on day 1 after 8 h sleep.

  python scripts/evaluate_fusion.py --drozy data/DROZY --wesad data/WESAD
Outputs reports/fusion_summary.json and models/fatigue_fusion.joblib
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak.data.drozy import load_drozy, load_drozy_pvt  # noqa: E402
from breathbreak.dataset import build_extended_hrv_table  # noqa: E402
from breathbreak.features import EXTENDED_FEATURE_NAMES  # noqa: E402
from breathbreak.fusion import FUSION_FEATURES, RETEST_RMSSD_DROP, make_fusion_model  # noqa: E402
from breathbreak.pvt import PVT_FEATURES, score_pvt  # noqa: E402
from breathbreak.schedule import SCHEDULE_FEATURES, schedule_features  # noqa: E402

HRV_COLS = [f"hrv_{c}" for c in EXTENDED_FEATURE_NAMES]


def session_table(root, pvt_seconds, wake_hour, sleep_hours):
    s = load_drozy_pvt(root)
    rows = []
    for _, r in s.iterrows():
        day1 = r.start.date() if r.session == "1" else (r.start - timedelta(days=1)).date()
        wake = datetime.combine(day1, datetime.min.time()) + timedelta(hours=wake_hour)
        f = schedule_features(wake, r.start, sleep_hours)
        f.update(score_pvt(r.rt_ms[r.stim_s < pvt_seconds]))
        f.update(subject=r.subject, session=r.session, kss=r.kss,
                 label=int(r.kss >= CFG.kss_fatigue_threshold))
        rows.append(f)
    return pd.DataFrame(rows)


def loso_proba(df, cols, C=1.0):
    X, y, g = df[cols].to_numpy(float), df.label.to_numpy(int), df.subject.to_numpy()
    p = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        p[te] = make_fusion_model(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p


def metrics(df, p, n_boot=2000, seed=0):
    d = df.assign(p=p).dropna(subset=["p"])
    auc = roc_auc_score(d.label, d.p)
    rng = np.random.default_rng(seed)
    subs = d.subject.unique()
    boots = []
    for _ in range(n_boot):
        b = pd.concat([d[d.subject == s] for s in rng.choice(subs, len(subs))])
        if b.label.nunique() == 2:
            boots.append(roc_auc_score(b.label, b.p))
    within = [roc_auc_score(x.label, x.p) for _, x in d.groupby("subject") if x.label.nunique() == 2]
    return {"sessions": int(len(d)), "people": int(d.subject.nunique()),
            "fatigued": int(d.label.sum()), "auc": round(auc, 3),
            "auc_ci95": [round(float(np.percentile(boots, 2.5)), 3), round(float(np.percentile(boots, 97.5)), 3)],
            "balanced_acc": round(balanced_accuracy_score(d.label, d.p >= 0.5), 3),
            "within_person_auc": round(float(np.mean(within)), 3), "within_person_n": len(within)}


def show(title, res):
    print(f"\n=== {title} ===")
    t = pd.DataFrame(res).T[["sessions", "people", "fatigued", "auc", "auc_ci95", "balanced_acc",
                             "within_person_auc", "within_person_n"]]
    print(t.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drozy", default="data/DROZY")
    ap.add_argument("--wesad", help="WESAD root (stress check for the HRV retest trigger)")
    ap.add_argument("--pvt-seconds", type=float, default=180.0)
    ap.add_argument("--wake-hour", type=float, default=7.0)
    ap.add_argument("--sleep-hours", type=float, default=8.0)
    a = ap.parse_args()

    # ---- main set: schedule + PVT ----
    st = session_table(a.drozy, a.pvt_seconds, a.wake_hour, a.sleep_hours)
    main_res = {
        "schedule only": metrics(st, loso_proba(st, list(SCHEDULE_FEATURES))),
        "PVT only (3 min)": metrics(st, loso_proba(st, list(PVT_FEATURES))),
        "schedule + PVT": metrics(st, loso_proba(st, list(FUSION_FEATURES))),
    }
    show("DROZY, all PVT sessions (leave-one-subject-out, session level)", main_res)

    # ---- HRV set ----
    print("\nComputing 3-min HRV (beats are cached in cache/beats)...", flush=True)
    hrv_win = build_extended_hrv_table(load_drozy(a.drozy, CFG), CFG)
    hrv_ses = (hrv_win.groupby(["subject", "session"])[list(EXTENDED_FEATURE_NAMES)].median()
               .add_prefix("hrv_").reset_index())
    sh = st.merge(hrv_ses, on=["subject", "session"]).dropna(subset=HRV_COLS).reset_index(drop=True)
    hrv_res = {
        "schedule + PVT": metrics(sh, loso_proba(sh, list(FUSION_FEATURES))),
        "HRV only": metrics(sh, loso_proba(sh, HRV_COLS, C=0.1)),
        "schedule + PVT + HRV": metrics(sh, loso_proba(sh, list(FUSION_FEATURES) + HRV_COLS, C=0.1)),
    }
    show("DROZY, sessions that also have HRV (same sessions for all three)", hrv_res)

    # ---- HRV retest trigger ----
    thr = np.log(1 - RETEST_RMSSD_DROP)
    trig = {"drozy_alert": float((hrv_win[hrv_win.label == 0].rmssd < thr).mean()),
            "drozy_fatigued": float((hrv_win[hrv_win.label == 1].rmssd < thr).mean())}
    if a.wesad:
        from breathbreak.data.wesad import load_wesad
        w = build_extended_hrv_table(load_wesad(a.wesad, kind="ecg"), CFG)
        trig["wesad_relaxed"] = float((w[w.session == "baseline"].rmssd < thr).mean())
        trig["wesad_stressed"] = float((w[w.session == "stress"].rmssd < thr).mean())
    print(f"\n=== HRV retest trigger (RMSSD >= {RETEST_RMSSD_DROP:.0%} below own baseline), share of 3-min windows ===")
    for k, v in trig.items():
        print(f"  {k:16s} {v:.0%}")

    # ---- PVT and KSS, for context ----
    print("\nMedian PVT lapses (first 3 min) by session:",
          st.groupby("session").pvt_lapses.median().to_dict(),
          "| median KSS:", st.groupby("session").kss.median().to_dict())

    # ---- save the deployable model (schedule + PVT, fit on all sessions) ----
    model = make_fusion_model().fit(st[list(FUSION_FEATURES)].to_numpy(float), st.label.to_numpy(int))
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": list(FUSION_FEATURES), "alert_threshold": CFG.alert_threshold,
                 "pvt_seconds": a.pvt_seconds, "trained_on": "DROZY",
                 "schedule_assumption": f"wake {a.wake_hour:g}:00 day 1, {a.sleep_hours:g} h sleep"},
                "models/fatigue_fusion.joblib")
    coefs = dict(zip(FUSION_FEATURES, np.round(model.named_steps["clf"].coef_[0], 3)))
    print("Fusion coefficients (standardised; + pushes toward fatigued):", coefs)

    os.makedirs("reports", exist_ok=True)
    with open("reports/fusion_summary.json", "w") as f:
        json.dump({"main": main_res, "hrv_set": hrv_res, "hrv_retest_trigger": trig,
                   "coefficients": {k: float(v) for k, v in coefs.items()},
                   "assumptions": {"pvt_seconds": a.pvt_seconds, "wake_hour": a.wake_hour,
                                   "sleep_hours": a.sleep_hours}}, f, indent=2)
    print("\nSaved models/fatigue_fusion.joblib and reports/fusion_summary.json")


if __name__ == "__main__":
    main()
