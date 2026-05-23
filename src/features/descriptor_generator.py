"""Compact anomaly descriptor generation from window features and IDS predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.models.vehicle_ids import (
    VEHICLE_MODELS,
    generate_window_predictions,
    load_feature_dataset,
    load_window_predictions,
    save_window_predictions,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

DESCRIPTOR_COLUMNS = [
    "event_id",
    "vehicle_model",
    "attack_type",
    "anomaly_score",
    "predicted_label",
    "ground_truth_label",
    "behavioural_feature_vector",
]

VEHICLE_PREFIX = {
    "Hyundai": "HYU",
    "Kia": "KIA",
    "Chevrolet": "CHV",
}

PRIMARY_MODEL = "random_forest"


def make_event_id(vehicle_model: str, window_id: int) -> str:
    """Build a compact unique event identifier."""
    prefix = VEHICLE_PREFIX.get(vehicle_model, "UNK")
    return f"EVT-{prefix}-{int(window_id):06d}"


def behavioural_vector_to_json(row: pd.Series) -> str:
    """Serialize behavioural features to a compact JSON array string."""
    values = [float(row[c]) if pd.notna(row[c]) else None for c in BEHAVIOURAL_FEATURE_COLUMNS]
    return json.dumps(values, separators=(",", ":"))


def aggregate_predictions(
    predictions: pd.DataFrame,
    *,
    primary_model: str = PRIMARY_MODEL,
) -> pd.DataFrame:
    """
    Collapse long-format model predictions to one row per window.

    ``predicted_label``: attack if **any** model flags the window.
    ``anomaly_score``: score from the primary model (RF probability by default).
    ``ids_model``: model used for the reported score.
    """
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "window_id",
                "vehicle_model",
                "predicted_label",
                "anomaly_score",
                "ids_model",
                "any_model_flag",
            ]
        )

    primary = predictions[predictions["model"] == primary_model].copy()
    if primary.empty:
        logger.warning("Primary model %s missing; using max score across models.", primary_model)
        primary = (
            predictions.groupby(["window_id", "vehicle_model"], as_index=False)
            .agg(anomaly_score=("anomaly_score", "max"))
        )
        primary["ids_model"] = "ensemble_max"
    else:
        primary = primary.drop(columns=["model"], errors="ignore")
        primary["ids_model"] = primary_model

    flags = (
        predictions.groupby(["window_id", "vehicle_model"], as_index=False)["predicted_label"]
        .max()
        .rename(columns={"predicted_label": "any_model_flag"})
    )

    agg = primary[["window_id", "vehicle_model", "anomaly_score", "ids_model"]].merge(
        flags, on=["window_id", "vehicle_model"], how="left"
    )
    agg["predicted_label"] = agg["any_model_flag"].astype(int)
    return agg


def filter_suspicious_windows(
    merged: pd.DataFrame,
    *,
    score_threshold: float = 0.5,
    require_any_model: bool = True,
) -> pd.DataFrame:
    """
    Keep only suspicious / anomalous windows.

    A window is suspicious if any IDS model predicted attack, or the primary
    anomaly score meets *score_threshold*.
    """
    if require_any_model:
        mask = (merged["predicted_label"] == 1) | (merged["anomaly_score"] >= score_threshold)
    else:
        mask = merged["anomaly_score"] >= score_threshold
    return merged[mask].copy()


def generate_anomaly_descriptors(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    primary_model: str = PRIMARY_MODEL,
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    """Build compact anomaly descriptor rows for suspicious windows only."""
    agg = aggregate_predictions(predictions, primary_model=primary_model)
    merged = features.merge(agg, on=["window_id", "vehicle_model"], how="inner")
    suspicious = filter_suspicious_windows(merged, score_threshold=score_threshold)

    if suspicious.empty:
        logger.warning("No suspicious windows found.")
        return pd.DataFrame(columns=DESCRIPTOR_COLUMNS)

    descriptors = pd.DataFrame(
        {
            "event_id": suspicious.apply(
                lambda r: make_event_id(r["vehicle_model"], r["window_id"]), axis=1
            ),
            "vehicle_model": suspicious["vehicle_model"],
            "attack_type": suspicious["attack_type"],
            "anomaly_score": suspicious["anomaly_score"].round(6),
            "predicted_label": suspicious["predicted_label"].astype(int),
            "ground_truth_label": suspicious["label"].astype(int),
            "behavioural_feature_vector": suspicious.apply(behavioural_vector_to_json, axis=1),
        }
    )
    return descriptors[DESCRIPTOR_COLUMNS]


def load_or_generate_predictions(
    features: pd.DataFrame,
    predictions_path: Path | str | None,
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    include_autoencoder: bool = True,
    regenerate: bool = False,
) -> pd.DataFrame:
    """Load cached window predictions or generate them from trained IDS models."""
    if predictions_path is not None:
        path = Path(predictions_path)
        if path.exists() and not regenerate:
            logger.info("Loading predictions from %s", path)
            return load_window_predictions(path)

    logger.info("Generating window-level IDS predictions...")
    predictions = generate_window_predictions(
        features,
        random_state=random_state,
        test_size=test_size,
        include_autoencoder=include_autoencoder,
    )
    if predictions_path is not None:
        save_window_predictions(predictions, predictions_path)
    return predictions


def save_anomaly_descriptors(descriptors: pd.DataFrame, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    descriptors.to_csv(out, index=False)
    logger.info("Saved %d anomaly descriptors to %s", len(descriptors), out)
    return out


def descriptor_statistics(descriptors: pd.DataFrame) -> dict[str, Any]:
    if descriptors.empty:
        return {"n_descriptors": 0}
    return {
        "n_descriptors": int(len(descriptors)),
        "by_vehicle": descriptors["vehicle_model"].value_counts().to_dict(),
        "by_attack_type": descriptors["attack_type"].value_counts().to_dict(),
        "mean_anomaly_score": float(descriptors["anomaly_score"].mean()),
        "predicted_attack_rate": float(descriptors["predicted_label"].mean()),
    }


def print_descriptor_summary(descriptors: pd.DataFrame) -> None:
    stats = descriptor_statistics(descriptors)
    print("\n=== Anomaly Descriptor Summary ===")
    print(f"Total descriptors: {stats.get('n_descriptors', 0):,}")
    if stats.get("by_vehicle"):
        print("  By vehicle:")
        for k, v in sorted(stats["by_vehicle"].items()):
            print(f"    {k}: {v:,}")
    if stats.get("by_attack_type"):
        print("  By attack type:")
        for k, v in sorted(stats["by_attack_type"].items()):
            print(f"    {k}: {v:,}")
    if "mean_anomaly_score" in stats:
        print(f"  Mean anomaly score: {stats['mean_anomaly_score']:.4f}")
    print("==================================\n")
