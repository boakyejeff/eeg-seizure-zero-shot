"""Baseline classifiers on windowed features.

Seizure data is heavily imbalanced (seconds of ictal vs hours of
interictal), so all estimators use class_weight="balanced" by default.
Standardize features with a StandardScaler fit on the training patient only.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_model(name: str = "logreg"):
    """Return a (scaler -> classifier) pipeline."""
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            class_weight="balanced",
            random_state=42,
        )
    else:
        raise ValueError(f"unknown model: {name}")
    return make_pipeline(StandardScaler(), clf)


def train_model(X: np.ndarray, y: np.ndarray, name: str = "logreg"):
    """Fit the pipeline on windowed features/labels from one patient."""
    if y.sum() == 0:
        raise ValueError("training labels contain no seizures")
    model = build_model(name)
    model.fit(X, y)
    return model


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    """Seizure probability per window."""
    return model.predict_proba(X)[:, 1]
