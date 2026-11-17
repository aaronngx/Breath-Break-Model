"""Train + evaluate Model B: frozen PPG foundation model (transfer learning) + linear probe.

Examples
  python scripts/train_model_b.py --mefar data/MEFAR_raw --mefar-labels data/mefar_labels.csv --embedder papagei
  python scripts/train_model_b.py --mefar data/MEFAR_raw --mefar-labels data/mefar_labels.csv --embedder prv
  python scripts/train_model_b.py --own data/own --embedder prv --wesad data/WESAD

Embeddings are cached in cache/ so re-runs are fast.
Outputs models/model_b.joblib + reports/model_b_summary.json
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak.dataset import build_raw_windows  # noqa: E402
from breathbreak.embedders import get_embedder  # noqa: E402
from breathbreak.evaluate import (loso_predict, print_report, session_order_control,  # noqa: E402
                                  stress_false_alarm_check, summarize)
from breathbreak.models import make_model_b, save_bundle, subtract_baseline_embeddings  # noqa: E402


def embed_cached(emb, X, fs, tag):
    os.makedirs("cache", exist_ok=True)
    key = hashlib.md5(X.tobytes()[:10_000_000] + f"{X.shape}{tag}".encode()).hexdigest()[:12]
    path = f"cache/emb_{tag}_{key}.npy"
    if os.path.exists(path):
        return np.load(path)
    E = emb.embed(X, fs)
    np.save(path, E)
    return E


def prepare(recs, cfg, emb, tag):
    X, meta, base_raw = build_raw_windows(recs, cfg)
    E = embed_cached(emb, X, cfg.ppg_fs, f"{emb.name}_{tag}")
    base = {s: emb.embed(w[None, :], cfg.ppg_fs)[0] for s, w in base_raw.items()}
    E, keep = subtract_baseline_embeddings(E, meta.subject.to_numpy(), base)
    return E, meta[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mefar", help="MEFAR_raw folder")
    ap.add_argument("--mefar-labels", help="CSV subject,session,cfs")
    ap.add_argument("--own", help="your AFE4950 data folder (uses the ppg column)")
    ap.add_argument("--wesad", help="WESAD root (stress control, wrist BVP)")
    ap.add_argument("--embedder", default=CFG.embedder, choices=["prv", "papagei", "pulseppg"])
    ap.add_argument("--weights", default=CFG.embedder_weights)
    ap.add_argument("--out", default="models/model_b.joblib")
    a = ap.parse_args()
    cfg = CFG
    cfg.embedder, cfg.embedder_weights = a.embedder, a.weights

    recs = []
    if a.mefar:
        from breathbreak.data.mefar import load_mefar
        recs += load_mefar(a.mefar, a.mefar_labels, cfg)
    if a.own:
        from breathbreak.data.own import load_own
        recs += [r for r in load_own(a.own, cfg, kinds=("ppg",)) if r.label is not None]
    if not recs:
        sys.exit("Give at least --mefar or --own")

    emb = get_embedder(cfg)
    E, meta = prepare(recs, cfg, emb, "train")
    y, g = meta.label.to_numpy(int), meta.subject.to_numpy()
    print(f"{len(meta)} windows, {len(np.unique(g))} subjects, embedding dim={E.shape[1]}, fatigued={y.mean():.2f}")

    model = make_model_b(cfg, E.shape[1])
    proba = loso_predict(model, E, y, g)
    res = summarize(y, proba, g)
    print_report(f"Model B ({emb.name}) - leave-one-subject-out", res)
    ctrl = session_order_control(meta)
    print_report("Control - session order only (should be WORSE than Model B)", ctrl)
    model.fit(E, y)

    stress = None
    if a.wesad:
        from breathbreak.data.wesad import load_wesad
        Ew, mw = prepare(load_wesad(a.wesad, kind="ppg"), cfg, emb, "wesad")
        stress = stress_false_alarm_check(model, Ew, mw)
        print("\n=== Stress control (WESAD wrist BVP) ===\n", stress.round(3))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    save_bundle(a.out, model, cfg, [f"emb_{i}" for i in range(E.shape[1])], "ppg",
                extra={"embedder": emb.name, "loso": res, "control": ctrl})
    os.makedirs("reports", exist_ok=True)
    with open("reports/model_b_summary.json", "w") as f:
        json.dump({"embedder": emb.name, "loso": res, "session_control": ctrl,
                   "stress": None if stress is None else stress.to_dict()}, f, indent=2, default=float)
    print(f"\nSaved {a.out} and reports/")


if __name__ == "__main__":
    main()
