"""MEFAR with a locked person-level test set.

    python scripts/mefar_split_eval.py                      # PRV pulse features (no pretrained model)
    python scripts/mefar_split_eval.py --embedder papagei   # frozen PaPaGei-S (transfer learning)

1. DEVELOP on train + val people only (19): leave-one-person-out, compare a few candidate models,
   pick the best and its alert threshold. The 4 test people are never loaded into this step.
2. TEST once: fit the chosen model on all 19, score the 4 test people, report.
Scores are reported per window and per session (average of the session's windows; 2 sessions/person).
A "session order" control (evening = fatigued, nothing else) is shown next to every number.
Writes reports/mefar_split_eval.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak.data.mefar import load_mefar  # noqa: E402
from breathbreak.dataset import build_raw_windows  # noqa: E402
from breathbreak.embedders import get_embedder  # noqa: E402
from breathbreak.evaluate import loso_predict  # noqa: E402
from breathbreak.models import subtract_baseline_embeddings  # noqa: E402

SPLIT, LABELS, ROOT = "data/mefar_split.csv", "data/mefar_labels.csv", "data/MEFAR_raw"


def model(C):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(class_weight="balanced", C=C, max_iter=2000))


def features(recs, emb):
    X, meta, base_raw = build_raw_windows(recs, CFG)
    E = emb.embed(X, CFG.ppg_fs)
    base = {s: emb.embed(w[None, :], CFG.ppg_fs)[0] for s, w in base_raw.items()}
    D, keep = subtract_baseline_embeddings(E, meta.subject.to_numpy(), base)
    meta = meta.assign(evening=(meta.session == "evening").astype(float))
    return {"raw": E[keep], "delta": D}, meta[keep].reset_index(drop=True)


def session_scores(meta, p):
    d = meta.assign(p=p).dropna(subset=["p"]).groupby(["subject", "session"]).agg(y=("label", "first"), p=("p", "mean"))
    return d.reset_index()


def scores(meta, p, thr=0.5):
    ok = ~np.isnan(p)
    y = meta.label.to_numpy(int)
    s = session_scores(meta, p)
    out = {"window_auc": roc_auc_score(y[ok], p[ok]), "n_windows": int(ok.sum()),
           "session_auc": roc_auc_score(s.y, s.p) if s.y.nunique() == 2 else np.nan,
           "session_bal_acc": balanced_accuracy_score(s.y, (s.p >= thr).astype(int)), "n_sessions": len(s)}
    return {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in out.items()}


def best_threshold(meta, p):
    s = session_scores(meta, p)
    grid = np.unique(np.round(s.p, 3))
    ba = [balanced_accuracy_score(s.y, (s.p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(ba))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="prv", choices=["prv", "papagei"])
    args = ap.parse_args()
    split = pd.read_csv(SPLIT).set_index("subject").split
    split.index = "mefar_" + split.index
    recs_all = load_mefar(ROOT, LABELS, CFG)
    dev_recs = [r for r in recs_all if split[r.subject] in ("train", "val")]
    test_recs = [r for r in recs_all if split[r.subject] == "test"]
    CFG.embedder = args.embedder
    emb = get_embedder(CFG)
    print(f"dev people: {len({r.subject for r in dev_recs})}  test people: {len({r.subject for r in test_recs})} (locked)")

    # ---- 1. develop: LOSO on the 19 dev people only
    Fd, md = features(dev_recs, emb)
    yd, gd = md.label.to_numpy(int), md.subject.to_numpy()
    cands = {f"{args.embedder} {kind}, C={C}": (kind, C) for kind in ("delta", "raw") for C in (0.01, 0.1, 1.0)}
    dev = {}
    print("\n=== DEVELOPMENT: leave-one-person-out on 19 train+val people ===")
    for name, (kind, C) in cands.items():
        p = loso_predict(model(C), Fd[kind], yd, gd)
        dev[name] = {**scores(md, p), "threshold": best_threshold(md, p)}
        print(f"  {name:28s} {dev[name]}")
    p_ctrl = loso_predict(LogisticRegression(class_weight="balanced"), md[["evening"]].to_numpy(), yd, gd)
    dev["control: evening only"] = scores(md, p_ctrl)
    print(f"  {'control: evening only':28s} {dev['control: evening only']}")

    chosen = max(cands, key=lambda n: dev[n]["session_auc"])
    kind, C = cands[chosen]
    thr = dev[chosen]["threshold"]
    print(f"\nChosen on dev: {chosen}  (session AUC {dev[chosen]['session_auc']}), threshold {thr}")

    # ---- 2. test once
    Ft, mt = features(test_recs, emb)
    m = clone(model(C)).fit(Fd[kind], yd)
    pt = m.predict_proba(Ft[kind])[:, 1]
    test = scores(mt, pt, thr)
    ctrl = LogisticRegression(class_weight="balanced").fit(md[["evening"]].to_numpy(), yd)
    test_ctrl = scores(mt, ctrl.predict_proba(mt[["evening"]].to_numpy())[:, 1], 0.5)
    sess = session_scores(mt, pt).assign(predicted=lambda d: np.where(d.p >= thr, "fatigued", "not"),
                                         truth=lambda d: np.where(d.y == 1, "fatigued", "not"))
    print("\n=== TEST (4 locked people, run once) ===")
    print(sess[["subject", "session", "truth", "predicted", "p"]].round(3).to_string(index=False))
    print(f"  model   {test}")
    print(f"  control {test_ctrl}  (evening = fatigued)")

    os.makedirs("reports", exist_ok=True)
    out = "reports/mefar_split_eval.json" if args.embedder == "prv" else f"reports/mefar_split_eval_{args.embedder}.json"
    with open(out, "w") as f:
        json.dump({"split": SPLIT, "embedder": args.embedder, "chosen": chosen, "threshold": thr, "development_loso_19": dev,
                   "test_4_people": test, "test_control_evening": test_ctrl,
                   "test_sessions": sess.round(3).to_dict("records")}, f, indent=2, default=float)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
