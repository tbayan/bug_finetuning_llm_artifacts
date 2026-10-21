"""
Metrics for baseline evaluation: accuracy, macro F1, precision, recall, confusion matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    *,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """
    Returns dict with accuracy, macro_f1, macro_precision, macro_recall,
    and optionally per-class metrics and confusion_matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if labels is not None:
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
    return out
