"""Feed the current HackEdge ECG files into the HRV models and check: can they tell normal from fatigue?

    python scripts/test_hackedge_models.py data/hackedge

Models tested (all trained on DROZY, never on HackEdge):
  - model_a:            HRV divided by the person's own normal baseline (log ratio)
  - model_a_nobaseline: raw HRV, no baseline
The main fusion model (sleep schedule + PVT) cannot be run: HackEdge has no wake time, sleep hours or PVT.
Baseline for each file = mean of the SAME person's OTHER normal files (the file itself is never in its baseline).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak import features as F  # noqa: E402
from breathbreak import signals as S  # noqa: E402
from breathbreak.data.hackedge import load_hackedge  # noqa: E402

def feats(x, fs, cfg):
    x = S.resample(np.asarray(x, float), fs, cfg.ecg_fs)
    ibi, frac = S.clean_ibi(S.peaks_to_ibi_ms(S.ecg_rpeaks(x, cfg.ecg_fs), cfg.ecg_fs), cfg)
    return F.hrv_features(ibi) if ibi.size >= 10 else None


def norm(f, base, cols):
    return [(f[c] - base[c]) / 100.0 if c == "pnn50" else np.log(max(f[c], 1e-6) / max(base[c], 1e-6)) for c in cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data/hackedge")
    root = ap.parse_args().root
    A = joblib.load("models/model_a.joblib")
    N = joblib.load("models/model_a_nobaseline.joblib")
    recs = load_hackedge(root, CFG)
    rows = [dict(file=r.extra["file"], person=r.subject[-1], truth=r.extra["condition"], f=feats(r.signal, r.fs, A["cfg"]))
            for r in recs]
    df = pd.DataFrame([r for r in rows if r["f"] is not None])
    out = []
    for i, r in df.iterrows():
        normals = df[(df.person == r.person) & (df.truth == "normal") & (df.index != i)]
        base = {c: float(np.mean([f[c] for f in normals.f])) for c in A["feature_cols"]} if len(normals) else None
        pa = float(A["model"].predict_proba([norm(r.f, base, A["feature_cols"])])[0, 1]) if base else np.nan
        pn = float(N["model"].predict_proba([[r.f[c] for c in N["feature_cols"]]])[0, 1])
        out.append(dict(file=r.file, person=r.person, truth=r.truth, hr=round(r.f["mean_hr"], 1), rmssd=round(r.f["rmssd"], 1),
                        p_model_a=round(pa, 2), p_no_baseline=round(pn, 2)))
    o = pd.DataFrame(out)
    for c in ("p_model_a", "p_no_baseline"):
        o[c.replace("p_", "pred_")] = np.where(o[c].isna(), "-", np.where(o[c] >= CFG.alert_threshold, "fatigue", "normal"))
    pd.set_option("display.width", 220)
    print(o.to_string(index=False))
    y = (o.truth == "fatigue").astype(int)
    for c in ("p_model_a", "p_no_baseline"):
        m = o[c].notna() & (o.person == "A")
        ok = ((o[c] >= CFG.alert_threshold).astype(int) == y)[m]
        print(f"{c:14s} person A: correct {int(ok.sum())}/{int(m.sum())}, AUC {roc_auc_score(y[m], o[c][m]):.2f}  (threshold {CFG.alert_threshold})")
    print("\nAUC 0.5 = coin flip. Person B has only normal files, so only person A can test normal vs fatigue.")
    print("Main fusion model (schedule + PVT): not runnable, HackEdge has no wake time / sleep / PVT.")


if __name__ == "__main__":
    main()
