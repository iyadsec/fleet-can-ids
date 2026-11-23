"""Vehicle-level evaluation from retrained predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from src.experiments.data_splits import is_benign_attack_type


def _latency_proxy(df: pd.DataFrame) -> float:
    if "frame_count" in df.columns:
        return float(df["frame_count"].mean() * 0.001)
    return float(df.get("window_size", pd.Series([100])).mean() * 0.001)


def compute_vehicle_level_metrics(predictions: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    test = predictions[predictions["split"] == "test"].copy()
    if test.empty:
        raise ValueError("No test predictions for vehicle-level evaluation")

    rows: list[dict] = []
    for vm, sub in test.groupby("vehicle_model"):
        y = sub["label"].astype(int).to_numpy()
        scores = sub["anomaly_score"].astype(float).to_numpy()
        pred = sub["local_event_alert"].astype(int).to_numpy()
        try:
            roc = float(roc_auc_score(y, scores))
        except ValueError:
            roc = float("nan")
        try:
            pr = float(average_precision_score(y, scores))
        except ValueError:
            pr = float("nan")
        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        rows.append(
            {
                "vehicle_model": vm,
                "roc_auc": roc,
                "pr_auc": pr,
                "precision": float(precision_score(y, pred, zero_division=0)),
                "recall": float(recall_score(y, pred, zero_division=0)),
                "f1": float(f1_score(y, pred, zero_division=0)),
                "false_positive_rate": fpr,
                "false_negative_rate": fnr,
                "latency_sec": _latency_proxy(sub),
                "test_windows": len(sub),
            }
        )

    pooled = test.copy()
    y = pooled["label"].astype(int).to_numpy()
    pred = pooled["local_event_alert"].astype(int).to_numpy()
    scores = pooled["anomaly_score"].astype(float).to_numpy()
    rows.append(
        {
            "vehicle_model": "pooled",
            "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "false_positive_rate": float(((y == 0) & (pred == 1)).sum() / max((y == 0).sum(), 1)),
            "false_negative_rate": float(((y == 1) & (pred == 0)).sum() / max((y == 1).sum(), 1)),
            "latency_sec": _latency_proxy(pooled),
            "test_windows": len(pooled),
        }
    )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "vehicle_level_metrics.csv", index=False)

    by_attack = test.groupby("attack_type").apply(
        lambda g: pd.Series(
            {
                "recall": float(recall_score(g["label"], g["local_event_alert"], zero_division=0)),
                "precision": float(precision_score(g["label"], g["local_event_alert"], zero_division=0)),
                "count": len(g),
            }
        ),
        include_groups=False,
    ).reset_index()
    by_attack.to_csv(out_dir / "by_attack_type.csv", index=False)
    metrics[metrics.vehicle_model != "pooled"].to_csv(out_dir / "by_vehicle_model.csv", index=False)

    th_rows = []
    for vm, sub in test.groupby("vehicle_model"):
        th_rows.append(
            {
                "vehicle_model": vm,
                "weak_threshold": float(sub["weak_threshold"].iloc[0]),
                "strong_threshold": float(sub["strong_threshold"].iloc[0]),
            }
        )
    pd.DataFrame(th_rows).to_csv(out_dir / "threshold_comparison.csv", index=False)
    return metrics
