"""DEMO ONLY: a model that labels the existing HackEdge files correctly because it was trained on them.

    python scripts/demo_insample.py ../HackEdge                        # train on all files, show in-sample labels
    python scripts/demo_insample.py ../HackEdge --predict data/fatigue/first_attempt.csv

What this is: logistic regression on whole-file HRV (HR, SDNN, RMSSD, pNN50, CV) of person A,
with almost no regularisation (C=1000), so it memorises the 7 usable files.
  - normal regularisation (C=0.5) gets only 4/7 even on its own training data
  - tested fairly (leave-one-file-out) it gets 0/7

So the in-sample 7/7 shows the pipeline runs end to end. It is NOT evidence that the model
detects fatigue, and a new file / new person will be close to a coin flip.
Say on the slide: "training accuracy (model has seen these files); not tested on new data".
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak.data.hackedge import read_ti_export  # noqa: E402
from train_hackedge_simple import FEATS, table, windows  # noqa: E402

MODEL_PATH = "models/hackedge_insample.joblib"
TAG = "IN-SAMPLE (trained on these files, not tested on new data)"


def make_model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="../HackEdge")
    ap.add_argument("--predict", help="TI export CSV (path relative to root, or absolute)")
    a = ap.parse_args()

    if a.predict:
        b = joblib.load(MODEL_PATH)
        path = a.predict if os.path.isabs(a.predict) else os.path.join(a.root, a.predict)
        sig = read_ti_export(path, CFG)
        if sig["ecg"] is None:
            print(f"ECG not usable: {sig['ecg_quality'].message()}")
            return
        w = windows(*sig["ecg"])
        if not w:
            print("Too few clean heartbeats - re-record.")
            return
        p = float(b["model"].predict_proba(pd.DataFrame(w)[FEATS].to_numpy())[0, 1])
        seen = os.path.normpath(a.predict) in {os.path.normpath(f) for f in b["files"]}
        print(f"{a.predict}: P(fatigue) = {p:.2f} -> {'FATIGUE' if p >= 0.5 else 'NORMAL'}")
        print(TAG if seen else "NEW FILE: model never saw it; expect close to a coin flip.")
        return

    df = table(a.root)
    X, y = df[FEATS].to_numpy(), df.label.to_numpy()
    m = make_model(1000).fit(X, y)
    df["p_fatigue"] = m.predict_proba(X)[:, 1].round(2)
    df["predicted"] = np.where(df.p_fatigue >= 0.5, "fatigue", "normal")
    df["truth"] = np.where(y == 1, "fatigue", "normal")
    print(f"\n{TAG}\n")
    print(df[["file", "truth", "predicted", "p_fatigue", "mean_hr", "rmssd"]].round(1).to_string(index=False))
    print(f"\nTraining accuracy: {(df.predicted == df.truth).sum()}/{len(df)}  <- model has seen every one of these files")
    reg = make_model(0.5).fit(X, y)
    print(f"Same data, normal regularisation (C=0.5): {(reg.predict(X) == y).sum()}/{len(y)} even in-sample")
    print("Fair test (leave-one-file-out, scripts/train_hackedge_simple.py): 0/7.  "
          "Best fair result overall: RMSSD vs own baseline 5/7, p = 0.20.")
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": m, "features": FEATS, "files": list(df.file), "note": TAG}, MODEL_PATH)
    print(f"Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
