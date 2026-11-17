"""Fatigue fusion: schedule prior + reaction-time test, with HRV only as a retest trigger.

    schedule (hours awake, sleep, clock)  -> sets the prior
    PVT (2-3 min reaction-time test)      -> confirms impairment
    HRV (AFE4950 ECG)                     -> can only ask for an extra PVT, never raises an alert itself

Why HRV is demoted: it measures arousal, so stress moves it the same way as
sleepiness (see storm-reports/ and reports/fusion_summary.json). The output is
fatigue risk only - never a stress score (EU AI Act Art. 5(1)(f) guidelines).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .pvt import PVT_FEATURES
from .schedule import SCHEDULE_FEATURES

FUSION_FEATURES = SCHEDULE_FEATURES + PVT_FEATURES
RETEST_RMSSD_DROP = 0.20      # HRV trigger: RMSSD >= 20 % below the person's baseline
# On WESAD the trigger fired in 65 % of stressed vs 16 % of relaxed 3-min windows,
# so it is rate-limited: at most one HRV-prompted re-test per cooldown period.
RETEST_COOLDOWN_MIN = 60.0


def make_fusion_model(C: float = 1.0) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(class_weight="balanced", C=C, max_iter=2000))])


def hrv_wants_retest(rmssd_log_ratio: float) -> bool:
    """rmssd_log_ratio = log(RMSSD_now / RMSSD_baseline) (as produced by build_extended_hrv_table)."""
    return bool(np.isfinite(rmssd_log_ratio) and rmssd_log_ratio < np.log(1 - RETEST_RMSSD_DROP))


class RetestLimiter:
    """Allows at most one HRV-prompted re-test per `cooldown_min` minutes."""

    def __init__(self, cooldown_min: float = RETEST_COOLDOWN_MIN):
        self.cooldown_s = cooldown_min * 60.0
        self._last = None

    def allow(self, now_s: float) -> bool:
        if self._last is None or now_s - self._last >= self.cooldown_s:
            self._last = now_s
            return True
        return False


def decide(p_fatigue: float, threshold: float = 0.6, hrv_retest: bool = False) -> str:
    """ALERT -> breathing break + re-test; RETEST -> run another PVT now; OK otherwise.
    HRV can only produce RETEST, never ALERT."""
    if p_fatigue >= threshold:
        return "ALERT"
    if hrv_retest:
        return "RETEST"
    return "OK"


def load_fusion(path: str = "models/fatigue_fusion.joblib") -> dict:
    import joblib
    return joblib.load(path)


def predict_fatigue(bundle: dict, features: dict) -> float:
    x = np.array([[features[c] for c in bundle["features"]]], dtype=float)
    return float(bundle["model"].predict_proba(x)[0, 1])
