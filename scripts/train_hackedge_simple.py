"""Simple normal-vs-fatigue classifier on the HackEdge folders (folder name = label).

What it is: ECG of the whole file -> HRV features -> logistic regression.
Two versions are compared: all features (HR, SDNN, RMSSD, pNN50, CV) and RMSSD only.
(10-s windows were tried first and scored below chance: RMSSD from 10 s is too noisy.) Only people recorded in BOTH conditions are used (otherwise the
model just learns who the person is). Noisy / not-worn files are dropped by the
signal-quality gate.

How it is tested (honest): leave-one-FILE-out. Each file is scored by a model that
never saw it. Note: RMSSD was chosen after looking at this same data, so even the
honest number is optimistic.

    python scripts/train_hackedge_simple.py data/hackedge                 # evaluate + save model (RMSSD only)
    python scripts/train_hackedge_simple.py --predict path/to/new.csv     # score one TI export

Limit: "fatigue" in this data = recorded while sick, one person. This shows the
pipeline can separate the two folders; it is not evidence of sleepiness detection.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak import features as F  # noqa: E402
from breathbreak import signals as S  # noqa: E402
from breathbreak.data.hackedge import load_hackedge, read_ti_export  # noqa: E402

FEATS = ["mean_hr", "sdnn", "rmssd", "pnn50", "cvnn"]
STEP_S, MIN_BEATS = 5.0, 8
MODEL_PATH = "models/hackedge_simple.joblib"


def windows(x, fs, win_s=None):
    """-> list of feature dicts, one per window (win_s=None: one window = whole file)."""
    f = CFG.ecg_fs
    x = S.resample(np.asarray(x, float), fs, f)
    pk = S.ecg_rpeaks(x, f) / f                       # beat times, s
    out, w = [], (win_s or len(x) / f)
    for a in np.arange(0, len(x) / f - w + 1e-9, STEP_S):
        b = pk[(pk >= a) & (pk < a + w)]
        if len(b) < MIN_BEATS:
            continue
        ibi, frac = S.clean_ibi(np.diff(b) * 1000.0, CFG)
        if ibi.size < MIN_BEATS - 1 or frac < 0.8:
            continue
        h = F.hrv_features(ibi)
        out.append({k: float(h[k]) for k in FEATS})
    return out


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


def table(root):
    recs = load_hackedge(root, CFG)
    both = {r.subject for r in recs if r.label == 1} & {r.subject for r in recs if r.label == 0}
    rows = []
    for r in recs:
        if r.subject not in both:
            print(f"[skip] {r.extra['file']}: {r.subject} has only one condition (model would learn the person)")
            continue
        w = windows(r.signal, r.fs)
        if w:
            rows.append(dict(file=r.extra["file"], person=r.subject, label=r.label, **w[0]))
    return pd.DataFrame(rows)


def loo(df, feats):
    """Leave-one-file-out probabilities."""
    X, y, p = df[feats].to_numpy(), df.label.to_numpy(), np.zeros(len(df))
    for i in range(len(df)):
        m = np.arange(len(df)) != i
        p[i] = make_model().fit(X[m], y[m]).predict_proba(X[i:i + 1])[0, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/hackedge")
    ap.add_argument("--predict", help="score one TI export CSV with the saved model")
    a = ap.parse_args()

    if a.predict:
        b = joblib.load(MODEL_PATH)
        sig = read_ti_export(a.predict, CFG)
        if sig["ecg"] is None:
            print(f"ECG not usable: {sig['ecg_quality'].message()}")
            return
        w = windows(*sig["ecg"])
        if not w:
            print("Too few clean heartbeats - hold still and re-record.")
            return
        p = float(b["model"].predict_proba(pd.DataFrame(w)[b["features"]].to_numpy())[0, 1])
        print(f"RMSSD {w[0]['rmssd']:.1f} ms, HR {w[0]['mean_hr']:.0f} bpm  ->  P(fatigue) = {p:.2f}  ->  "
              f"{'FATIGUE' if p >= 0.5 else 'NORMAL'}")
        print("Trained on one person's healthy vs sick recordings; treat as a demo, not a diagnosis.")
        return

    df = table(a.root)
    pd.set_option("display.width", 200)
    print(f"\n{len(df)} files ({int(df.label.sum())} fatigue, {int((df.label == 0).sum())} normal), "
          "whole-file features\n")
    df["truth"] = np.where(df.label == 1, "fatigue", "normal")
    for name, feats in (("all 5 features", FEATS), ("RMSSD only", ["rmssd"])):
        p = loo(df, feats)
        df[f"p ({name})"] = p.round(2)
        ok = (p >= 0.5) == df.label
        print(f"HONEST leave-one-file-out, {name:15s}: accuracy {ok.mean():.2f} ({int(ok.sum())}/{len(df)}), "
              f"AUC {roc_auc_score(df.label, p):.2f}")
    print()
    print(df[["file", "truth", "mean_hr", "rmssd", "p (all 5 features)", "p (RMSSD only)"]].round(1).to_string(index=False))
    y = df.label.to_numpy()
    FEATS[:] = ["rmssd"]                                   # ship the better, simpler version
    m = make_model().fit(df[FEATS].to_numpy(), y)
    coef = dict(zip(FEATS, np.round(m[-1].coef_[0], 2)))
    print(f"\nWeights (standardised; + pushes toward fatigue): {coef}")
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": m, "features": list(FEATS),
                 "note": "HackEdge folder label (healthy vs sick), person A only"}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
