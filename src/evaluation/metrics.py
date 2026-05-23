"""Classification metrics for intrusion detection experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_score: np.ndarray | None = None,
    metric_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute requested metrics; skips ROC-AUC when scores are unavailable."""
    names = metric_names or DEFAULT_METRIC_NAMES
    results: dict[str, float] = {}
    if "accuracy" in names:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))
    if "precision" in names:
        results["precision"] = float(
            precision_score(y_true, y_pred, zero_division=0, average="binary")
        )
    if "recall" in names:
        results["recall"] = float(
            recall_score(y_true, y_pred, zero_division=0, average="binary")
        )
    if "f1" in names:
        results["f1"] = float(f1_score(y_true, y_pred, zero_division=0, average="binary"))
    if ("roc_auc" in names or "auc" in names) and y_score is not None:
        try:
            results["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            results["roc_auc"] = float("nan")
    return results


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Return 2x2 confusion matrix for labels {0, 1}."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    """Write metrics dict to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    return out
