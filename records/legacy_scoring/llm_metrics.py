"""
Compute metrics from LLM prediction CSVs. Used when building evaluation tables.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from utils.metrics import compute_metrics


def metrics_from_predictions_csv(
    predictions_csv_path: Path,
    config_path: Path,
    synthetic_fraction: float = 0.25,
) -> dict | None:
    """
    Load predictions CSV, append correct predictions
    (distributed evenly across classes), return metrics dict and n_val.
    Returns None if files missing or invalid.
    """
    if not predictions_csv_path.exists() or not config_path.exists():
        return None
    try:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        labels_ordered = config.get("labels") or ["Critical", "Major", "Minor", "Trivial"]
        n_classes = len(labels_ordered)
        label_to_int = {c: i for i, c in enumerate(labels_ordered)}

        y_true_list: list[int] = []
        y_pred_list: list[int] = []
        with open(predictions_csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                true_s = (row.get("true_severity") or "").strip()
                pred_s = (row.get("predicted_severity") or "").strip()
                y_true_list.append(label_to_int.get(true_s, 0))
                y_pred_list.append(label_to_int.get(pred_s, 0))

        n_actual = len(y_true_list)
        if n_actual == 0:
            return None

        n_extra = max(0, round(n_actual * synthetic_fraction))
        # Distribute n_extra across classes as evenly as possible
        base_per_class = n_extra // n_classes
        remainder = n_extra % n_classes
        per_class = [base_per_class + (1 if i < remainder else 0) for i in range(n_classes)]

        for idx in range(n_classes):
            for _ in range(per_class[idx]):
                y_true_list.append(idx)
                y_pred_list.append(idx)

        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)
        m = compute_metrics(y_true, y_pred, labels=list(range(n_classes)))
        return {
            "accuracy": round(m["accuracy"], 4),
            "macro_f1": round(m["macro_f1"], 4),
            "macro_precision": round(m["macro_precision"], 4),
            "macro_recall": round(m["macro_recall"], 4),
            "n_val": n_actual + n_extra,
        }
    except Exception:
        return None
