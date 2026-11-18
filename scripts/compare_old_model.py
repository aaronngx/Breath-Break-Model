"""Head-to-head on DROZY: HackEdge's old "delta" model vs the schedule + PVT fusion model.

The old model is RE-IMPLEMENTED here from HackEdge fatigue_pipeline.py / train_delta_model.py
(same filters, Pan-Tompkins R-peaks, RR cleaning, stable-range trim, 8 features
HR, HR_range, SDNN, RMSSD, pNN50, R_mV, T_over_R, RT_ms; same within-person delta pairs,
StandardScaler + LogisticRegression(C=0.5, balanced); thresholds 0.5 watch / 0.667 alarm).

Pre-declared test:
  sessions 2 and 3 of every DROZY subject who has a rested session 1 (= the baseline)
  label: KSS >= 7; leave-one-subject-out for BOTH models
  old model (a) features from the whole 10-min session (best case)
            (b) features from a 30-s clip at 60-90 s (how it was used in HackEdge)
  new model: schedule + 3-min PVT, same sessions

  python scripts/compare_old_model.py --drozy data/DROZY
"""
import argparse
import os
import sys
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import signal  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from breathbreak import CFG  # noqa: E402
from breathbreak import signals as S  # noqa: E402
from breathbreak.data.drozy import load_drozy  # noqa: E402
from breathbreak.fusion import FUSION_FEATURES  # noqa: E402

FS = 500
PRE, POST = 0.2, 0.4
FEATURES = ["HR", "HR_range", "SDNN", "RMSSD", "pNN50", "R_mV", "T_over_R", "RT_ms"]


# ---------------- HackEdge pipeline (re-implemented) ----------------
def butter(x, fc, kind, order=2):
    b, a = signal.butter(order, fc, btype=kind, fs=FS)
    return signal.filtfilt(b, a, x)


def clean_ecg(v):
    x = 1000 * (v - np.median(v))
    x = butter(x, 0.5, "highpass")
    b, a = signal.iirnotch(60, 30, FS)
    x = signal.filtfilt(b, a, x)
    return butter(x, 40, "lowpass")


def artifact_mask(v, c, k=2.0):
    sec = np.arange(len(c)) // FS
    peak = pd.Series(np.abs(c)).groupby(sec).max()
    bad = np.isin(sec, peak.index[peak > k * peak.median()]) | (np.abs(v) > 1.19)
    return np.convolve(bad, np.ones(FS), "same") > 0


def pan_tompkins(c, bad):
    x = np.where(bad, 0, c)
    bp = butter(x, [5, 20], "bandpass")
    w = int(.12 * FS)
    mwi = np.convolve(np.gradient(bp) ** 2, np.ones(w) / w, "same")
    ref = mwi[~bad] if (~bad).any() else mwi
    pk, _ = signal.find_peaks(mwi, distance=int(.33 * FS), height=.3 * np.percentile(ref, 99))
    hw = int(.06 * FS)
    pk = np.array([p for p in pk if hw <= p < len(x) - hw], int)
    if not len(pk):
        return pk
    absmax = np.array([p - hw + np.argmax(np.abs(c[p - hw:p + hw])) for p in pk])
    pol = np.sign(np.median(c[absmax])) or 1.0
    pk = np.array([p - hw + np.argmax(pol * c[p - hw:p + hw]) for p in pk], int)
    pk = pk[~bad[pk]]
    amp = pol * c[pk]
    med = np.median(amp) if len(pk) else 0
    return pk[(amp >= .5 * med) & (amp <= 1.6 * med)]


def clean_rr(tp, bad, tol=0.2):
    rr = np.diff(tp)
    t = tp[1:]
    ok = (rr > .33) & (rr < 1.5)
    ok &= ~np.array([bad[int(a * FS):int(b * FS) + 1].any() for a, b in zip(tp[:-1], tp[1:])], bool)
    med = pd.Series(np.where(ok, rr, np.nan)).rolling(5, center=True, min_periods=1).median().values
    ok &= np.abs(rr - med) <= tol * med
    return t, rr, ok


def stable_range(v, c, rp, trim_min=2, run=3):
    mv = 1000 * (v - np.median(v))
    sec = np.arange(len(mv)) // FS
    drift = pd.Series(butter(mv, 1.0, "lowpass")).groupby(sec).agg(np.ptp).values
    peak = pd.Series(np.abs(c)).groupby(sec).max().values
    r_amp = np.median(np.abs(c[rp])) if len(rp) else np.inf
    bad = (drift > .3 * r_amp) | (peak > 1.6 * r_amp) | (peak < .5 * r_amp)
    s, e = trim_min, len(bad) - trim_min
    while s < e and bad[s:s + run].any():
        s += 1
    while e > s and bad[e - run:e].any():
        e -= 1
    return float(s), float(e)


def features_from(rr, beat):
    ms = 1000 * rr
    d = np.diff(ms)
    i0 = int(PRE * FS)
    base = np.median(beat[:int(.08 * FS)])
    R = beat[i0] - base
    tseg = beat[i0 + int(.15 * FS): i0 + int(.38 * FS)] - base
    ti = int(np.argmax(np.abs(tseg)))
    return dict(HR=60000 / ms.mean(), HR_range=60 / rr.min() - 60 / rr.max(), SDNN=ms.std(ddof=1),
                RMSSD=np.sqrt(np.mean(d ** 2)), pNN50=100 * np.mean(np.abs(d) > 50),
                R_mV=R, T_over_R=tseg[ti] / R, RT_ms=(int(.15 * FS) + ti) / FS * 1000)


def analyze(v, segment=None):
    t = np.arange(len(v)) / FS
    c = clean_ecg(v)
    mask = artifact_mask(v, c)
    rp = pan_tompkins(c, mask)
    if len(rp) <= 2:
        return None, None
    t_rr, rr, ok = clean_rr(t[rp], mask)
    keep = stable_range(v, c, rp)
    a, b = segment if segment else keep
    sel = ok & (t_rr > a) & (t_rr < b)
    pk = rp[(t[rp] > a + PRE) & (t[rp] < b - POST)]
    pk = pk[~np.array([mask[p - int(PRE * FS):p + int(POST * FS)].any() for p in pk], bool)]
    if sel.sum() < 15 or len(pk) < 10:
        return None, keep
    beats = np.array([c[p - int(PRE * FS):p + int(POST * FS)] for p in pk])
    return features_from(rr[sel], np.median(beats, 0)), keep


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))


# ---------------- evaluation ----------------
def session_features(recs, clip):
    rows, halves = [], {}
    for r in recs:
        v = S.resample(r.signal, r.fs, FS)
        if clip:
            v = v[60 * FS: 90 * FS]
        f, keep = analyze(v)
        if f is None:
            continue
        rows.append(dict(subject=r.subject, session=r.session, label=r.label, **f))
        if r.label == 0 and keep and keep[1] - keep[0] >= 40:          # split-half "no change" pairs
            mid = (keep[0] + keep[1]) / 2
            h1, _ = analyze(v, (keep[0], mid))
            h2, _ = analyze(v, (mid, keep[1]))
            if h1 and h2:
                halves[(r.subject, r.session)] = (h1, h2)
    return pd.DataFrame(rows), halves


def train_pairs(df, halves, subjects):
    X, y = [], []
    for s in subjects:
        mine = df[df.subject == s]
        normals, fat = mine[mine.label == 0], mine[mine.label == 1]
        for _, f in fat.iterrows():
            for _, n in normals.iterrows():
                X.append([f[c] - n[c] for c in FEATURES]); y.append(1)
        for (_, a), (_, b) in permutations(list(normals.iterrows()), 2):
            X.append([a[c] - b[c] for c in FEATURES]); y.append(0)
    for (s, _), (h1, h2) in halves.items():
        if s in subjects:
            for p, q in ((h2, h1), (h1, h2)):
                X.append([p[c] - q[c] for c in FEATURES]); y.append(0)
    return np.array(X), np.array(y)


def old_model_loso(df, halves):
    out = []
    subjects = df.subject.unique()
    for s in subjects:
        base = df[(df.subject == s) & (df.session == "1")]
        test = df[(df.subject == s) & (df.session.isin(["2", "3"]))]
        if base.empty or test.empty:
            continue
        X, y = train_pairs(df, halves, [x for x in subjects if x != s])
        m = make_model().fit(X, y)
        b = base.iloc[0]
        for _, r in test.iterrows():
            p = m.predict_proba([[r[c] - b[c] for c in FEATURES]])[0, 1]
            out.append(dict(subject=s, session=r.session, label=int(r.label), p=float(p)))
    return pd.DataFrame(out)


def scores(d, name):
    w = [roc_auc_score(x.label, x.p) for _, x in d.groupby("subject") if x.label.nunique() == 2]
    return dict(model=name, sessions=len(d), people=d.subject.nunique(), fatigued=int(d.label.sum()),
                auc=round(roc_auc_score(d.label, d.p), 3),
                bal_acc_at_0_5=round(balanced_accuracy_score(d.label, d.p >= 0.5), 3),
                bal_acc_at_0_667=round(balanced_accuracy_score(d.label, d.p >= 0.667), 3),
                within_person_auc=round(float(np.mean(w)), 3) if w else np.nan, within_n=len(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drozy", default="data/DROZY")
    a = ap.parse_args()
    recs = load_drozy(a.drozy, CFG)

    res = {}
    for clip, name in ((False, "old delta model, whole 10-min session"), (True, "old delta model, 30-s clip")):
        df, halves = session_features(recs, clip)
        res[name] = old_model_loso(df, halves)

    import evaluate_fusion as ef
    st = ef.session_table(a.drozy, 180.0, 7.0, 8.0)
    st["p"] = ef.loso_proba(st, list(FUSION_FEATURES))
    keys = set(map(tuple, res["old delta model, whole 10-min session"][["subject", "session"]].values))
    new = st[[(s, ss) in keys for s, ss in zip(st.subject, st.session)]][["subject", "session", "label", "p"]]
    res["NEW schedule + 3-min PVT"] = new

    common = set.intersection(*[set(map(tuple, d[["subject", "session"]].values)) for d in res.values()])
    rows = []
    for name, d in res.items():
        d = d[[(s, ss) in common for s, ss in zip(d.subject, d.session)]]
        rows.append(scores(d, name))
    pd.set_option("display.width", 200)
    print(f"\nDROZY sessions 2-3 scored against session-1 baseline, leave-one-subject-out "
          f"(same {len(common)} sessions for every model)\n")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
