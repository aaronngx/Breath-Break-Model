"""Train + evaluate Model A: time-domain HRV features -> logistic regression.

Examples
  python scripts/train_model_a.py --drozy data/DROZY
  python scripts/train_model_a.py --drozy data/DROZY --wesad data/WESAD --own data/own
  python scripts/train_model_a.py --own data/own                     # your recordings only

Outputs models/model_a.joblib + reports/model_a_windows.csv
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak.dataset import build_feature_table  # noqa: E402
from breathbreak.evaluate import (loso_predict, print_report, session_order_control,  # noqa: E402
                                  stress_false_alarm_check, summarize)
from breathbreak.models import make_model_a, save_bundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drozy", help="DROZY root folder")
    ap.add_argument("--drozy-labels", help="optional CSV subject,session,kss")
    ap.add_argument("--own", help="folder with sessions.csv + your AFE4950 CSVs")
    ap.add_argument("--own-kind", default="ecg", choices=["ecg", "ppg"])
    ap.add_argument("--wesad", help="WESAD root (stress control check)")
    ap.add_argument("--out", default="models/model_a.joblib")
    ap.add_argument("--no-baseline", action="store_true", help="disable personal-baseline normalization")
    a = ap.parse_args()
    cfg = CFG
    if a.no_baseline:
        cfg.normalize_to_baseline = False

    recs = []
    if a.drozy:
        from breathbreak.data.drozy import load_drozy
        recs += load_drozy(a.drozy, cfg, labels_csv=a.drozy_labels)
    if a.own:
        from breathbreak.data.own import load_own
        recs += [r for r in load_own(a.own, cfg, kinds=(a.own_kind,)) if r.label is not None]
    if not recs:
        sys.exit("Give at least --drozy or --own")
    kinds = {r.kind for r in recs}
    if len(kinds) > 1:
        sys.exit(f"Mixing signal kinds {kinds}; train one model per kind")
    kind = kinds.pop()

    table = build_feature_table(recs, cfg)
    cols = list(cfg.model_a_features)
    X, y, g = table[cols].to_numpy(float), table.label.to_numpy(int), table.subject.to_numpy()
    print(f"{len(table)} windows, {len(np.unique(g))} subjects, fatigued={y.mean():.2f}")

    model = make_model_a(cfg)
    proba = loso_predict(model, X, y, g)
    res = summarize(y, proba, g)
    print_report("Model A - leave-one-subject-out", res)
    ctrl = session_order_control(table)
    print_report("Control - session order only (should be WORSE than Model A)", ctrl)

    model.fit(X, y)
    coefs = {c: round(float(v), 3) for c, v in zip(cols, model.named_steps["clf"].coef_[0])}
    print("\nCoefficients (+ pushes toward 'fatigued'):", coefs)

    stress = None
    if a.wesad:
        from breathbreak.data.wesad import load_wesad
        w = build_feature_table(load_wesad(a.wesad, kind=kind), cfg)
        stress = stress_false_alarm_check(model, w[cols].to_numpy(float), w)
        print("\n=== Stress control (WESAD, nobody sleep-deprived) ===\n", stress.round(3))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_bundle(a.out, model, cfg, cols, kind, extra={"loso": res, "control": ctrl, "coefs": coefs})
    os.makedirs("reports", exist_ok=True)
    table.assign(p_fatigued_loso=proba).to_csv("reports/model_a_windows.csv", index=False)
    with open("reports/model_a_summary.json", "w") as f:
        json.dump({"loso": res, "session_control": ctrl, "coefs": coefs,
                   "stress": None if stress is None else stress.to_dict()}, f, indent=2, default=float)
    print(f"\nSaved {a.out} and reports/")


if __name__ == "__main__":
    main()
