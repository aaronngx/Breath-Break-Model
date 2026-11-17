"""Model A (HRV + logistic regression) and Model B (frozen embedder + logistic regression)."""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_model_a(cfg) -> Pipeline:
    """Interpretable: standardize -> logistic regression (balanced classes)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000,
                                   random_state=cfg.random_state)),
    ])


def make_model_b(cfg, n_features: int) -> Pipeline:
    """Linear probe on frozen embeddings. PCA keeps it small for few subjects."""
    steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    if n_features > 32:
        steps.append(("pca", PCA(n_components=32, random_state=cfg.random_state)))
    steps.append(("clf", LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000,
                                            random_state=cfg.random_state)))
    return Pipeline(steps)


def subtract_baseline_embeddings(E: np.ndarray, subjects: np.ndarray, base_emb: dict) -> tuple[np.ndarray, np.ndarray]:
    """Model B personal baseline: embedding minus that person's baseline embedding.
    Returns (E_normalized, keep_mask) - subjects without a baseline are dropped."""
    keep = np.array([s in base_emb for s in subjects])
    E2 = E.copy()
    for s, b in base_emb.items():
        m = subjects == s
        E2[m] = E[m] - b
    return E2[keep], keep


def save_bundle(path: str, model, cfg, feature_cols, kind: str, extra: dict | None = None):
    joblib.dump({"model": model, "cfg": cfg, "feature_cols": list(feature_cols),
                 "kind": kind, "extra": extra or {}}, path)


def load_bundle(path: str) -> dict:
    return joblib.load(path)
