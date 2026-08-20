"""Metrics aligned with the competition's accuracy objective."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def binary_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    labels = probabilities >= threshold
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "threshold": float(threshold),
    }


def best_accuracy_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    low: float = 0.35,
    high: float = 0.65,
    step: float = 0.0025,
) -> tuple[float, float]:
    """Find a deterministic OOF threshold, preferring the value closest to 0.5 on ties."""
    candidates = np.arange(low, high + step / 2, step)
    scored = [
        (float(accuracy_score(y_true, probabilities >= threshold)), float(threshold))
        for threshold in candidates
    ]
    score, threshold = max(scored, key=lambda item: (item[0], -abs(item[1] - 0.5)))
    return threshold, score
