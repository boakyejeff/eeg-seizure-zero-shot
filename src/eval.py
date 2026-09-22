"""Cross-patient (zero-shot) evaluation harness.

The centerpiece of this project: given a dict of patient_id -> (X, y)
windowed datasets, hold out each patient in turn, train on the rest, and
report per-patient metrics. Works for 2 patients (train on A, test on B)
or leave-one-patient-out with more.

Metrics (window level, then mapped to clinically meaningful rates):
- AUC (ROC) over window seizure probabilities
- sensitivity = TP / (TP + FN)
- specificity = TN / (TN + FP)
- false-alarm rate per hour = FP / total recording hours
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve

from model import build_model, predict_proba


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    window_s: float = 4.0,
    threshold: float = 0.5,
) -> dict:
    """AUC, sensitivity, specificity, and false alarms/hour at a threshold."""
    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    try:
        auc_score = float(roc_auc_score(y_true, y_score))
    except ValueError:  # only one class present in y_true
        auc_score = float("nan")

    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    hours = len(y_true) * window_s / 3600.0
    fa_per_hour = fp / hours if hours else float("nan")

    return {
        "auc": auc_score,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "false_alarms_per_hour": fa_per_hour,
        "n_windows": int(len(y_true)),
        "n_seizure_windows": int(y_true.sum()),
        "recording_hours": hours,
        "threshold": threshold,
    }


def cross_patient_eval(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    model_name: str = "logreg",
    window_s: float = 4.0,
) -> dict[str, dict]:
    """Leave-one-patient-out: train on all patients except one, test on it.

    datasets: {patient_id: (X, y)}. Returns {patient_id: metrics}.
    """
    results: dict[str, dict] = {}
    patient_ids = sorted(datasets)
    for held_out in patient_ids:
        train_ids = [p for p in patient_ids if p != held_out]
        X_train = np.vstack([datasets[p][0] for p in train_ids])
        y_train = np.concatenate([datasets[p][1] for p in train_ids])
        model = build_model(model_name)
        model.fit(X_train, y_train)

        X_test, y_test = datasets[held_out]
        scores = predict_proba(model, X_test)
        results[held_out] = compute_metrics(y_test, scores, window_s=window_s)
        results[held_out]["trained_on"] = train_ids
    return results


def print_report(results: dict[str, dict]) -> None:
    """Human-readable console report of a cross-patient evaluation."""
    print(f"{'held-out':<12}{'AUC':>8}{'sens':>8}{'spec':>8}{'FA/h':>10}")
    print("-" * 46)
    for patient, m in results.items():
        print(
            f"{patient:<12}"
            f"{m['auc']:>8.3f}"
            f"{m['sensitivity']:>8.3f}"
            f"{m['specificity']:>8.3f}"
            f"{m['false_alarms_per_hour']:>10.2f}"
        )
        print(
            f"  trained_on={m['trained_on']} "
            f"n={m['n_windows']} seizure_windows={m['n_seizure_windows']} "
            f"hours={m['recording_hours']:.2f}"
        )
