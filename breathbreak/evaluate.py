"""Honest evaluation: leave-one-subject-out (LOSO) + two sanity controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut


def loso_predict(model, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Out-of-subject probability for every window. A subject's windows are
    only ever scored by a model that never saw that subject."""
    proba = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = clone(model).fit(X[tr], y[tr])
        proba[te] = m.predict_proba(X[te])[:, 1]
    return proba


def summarize(y: np.ndarray, proba: np.ndarray, groups: np.ndarray, thr: float = 0.5) -> dict:
    ok = ~np.isnan(proba)
    y, proba, groups = y[ok], proba[ok], groups[ok]
    pred = (proba >= thr).astype(int)
    per_subj = pd.DataFrame({"g": groups, "y": y, "p": pred}).groupby("g").apply(
        lambda d: (d.y == d.p).mean(), include_groups=False)
    out = {
        "n_windows": int(len(y)),
        "n_subjects": int(len(np.unique(groups))),
        "fatigued_fraction": float(y.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "per_subject_acc_median": float(per_subj.median()),
        "per_subject_acc_min": float(per_subj.min()),
        "per_subject_acc_max": float(per_subj.max()),
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, proba))
    return out


def session_order_control(meta: pd.DataFrame) -> dict | None:
    """Can 'which session is this' alone predict the label?

    If this control scores about as well as your model, your model may be
    learning time-of-day / session order, not fatigue (a known DROZY trap).
    """
    if "session_index" not in meta or meta.session_index.isna().all():
        return None
    X = meta[["session_index"]].to_numpy(float)
    y = meta.label.to_numpy(int)
    g = meta.subject.to_numpy()
    p = loso_predict(LogisticRegression(class_weight="balanced"), X, y, g)
    return summarize(y, p, g)


def stress_false_alarm_check(model, X_ctrl: np.ndarray, meta_ctrl: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """Run a fatigue model on WESAD (nobody is sleep-deprived).
    Good result: P(fatigued) does NOT jump during 'stress' vs 'baseline'."""
    p = model.predict_proba(X_ctrl)[:, 1]
    d = meta_ctrl.assign(p_fatigued=p, flagged=(p >= thr).astype(int))
    return d.groupby("session")[["p_fatigued", "flagged"]].mean().rename(
        columns={"flagged": "flagged_fraction"})


def print_report(title: str, res: dict | None):
    print(f"\n=== {title} ===")
    if not res:
        print("  (not available)")
        return
    for k, v in res.items():
        print(f"  {k:26s} {v:.3f}" if isinstance(v, float) else f"  {k:26s} {v}")
