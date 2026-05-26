"""Final fleet-level anomaly outcome decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

ISOLATED = "Isolated anomaly"
WEAK_ISOLATED = "Weak isolated signal"
COORDINATED = "Fleet-level coordinated behavioural pattern"


def load_cluster_results(path: Path | str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fleet cluster results not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_fleet_edges(path: Path | str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fleet edges not found: {csv_path}")
    return pd.read_csv(csv_path)


def classify_final_outcomes(
    cluster_results: pd.DataFrame,
    fleet_edges: pd.DataFrame | None = None,
    *,
    similarity_threshold: float = 0.85,
    minimum_cluster_size: int = 2,
) -> pd.DataFrame:
    """
    Classify each anomaly event as isolated or coordinated.

    A coordinated pattern requires a cluster containing anomalies from more
    than one vehicle and high behavioural similarity. Temporal proximity is
    intentionally not used.
    """
    required = {
        "event_id",
        "window_id",
        "vehicle_model",
        "attack_type",
        "anomaly_score",
        "evidence_level",
        "local_alert",
        "weak_signal",
        "cluster_id",
        "cluster_size",
        "num_unique_vehicles",
        "mean_cluster_similarity",
    }
    missing = required - set(cluster_results.columns)
    if missing:
        raise ValueError(f"Cluster results missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for _, row in cluster_results.iterrows():
        local_alert = int(row["local_alert"])
        weak_signal = int(row["weak_signal"])
        num_vehicles = int(row["num_unique_vehicles"])
        cluster_size = int(row["cluster_size"])
        mean_sim = float(row["mean_cluster_similarity"])
        is_fleet = (
            int(row["cluster_id"]) != -1
            and num_vehicles > 1
            and cluster_size >= minimum_cluster_size
            and mean_sim >= similarity_threshold
        )
        if is_fleet:
            outcome = COORDINATED
        elif local_alert == 1 and num_vehicles == 1:
            outcome = ISOLATED
        elif weak_signal == 1 and num_vehicles == 1:
            outcome = WEAK_ISOLATED
        else:
            outcome = ISOLATED if local_alert == 1 else WEAK_ISOLATED

        upgraded = row["evidence_level"] == "weak_suspicious_signal" and outcome == COORDINATED

        rows.append(
            {
                "event_id": row["event_id"],
                "window_id": int(row["window_id"]),
                "vehicle_model": row["vehicle_model"],
                "attack_type": row["attack_type"],
                "anomaly_score": float(row["anomaly_score"]),
                "evidence_level": row["evidence_level"],
                "local_alert": local_alert,
                "weak_signal": weak_signal,
                "cluster_id": int(row["cluster_id"]),
                "cluster_size": cluster_size,
                "num_unique_vehicles": num_vehicles,
                "mean_cluster_similarity": mean_sim,
                "final_outcome": outcome,
                "was_upgraded_by_fleet": bool(upgraded),
            }
        )

    return pd.DataFrame(rows)


def summarize_final_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate final outcome counts for reporting."""
    if outcomes.empty:
        return pd.DataFrame(
            columns=["final_outcome", "event_count", "vehicle_count", "attack_types"]
        )
    return (
        outcomes.groupby("final_outcome", as_index=False)
        .agg(
            event_count=("event_id", "count"),
            vehicle_count=("vehicle_model", "nunique"),
            attack_types=("attack_type", lambda s: ",".join(sorted(s.dropna().unique()))),
        )
        .sort_values("event_count", ascending=False)
    )


def save_final_outcomes(
    outcomes: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    outcomes_path: Path | str,
    summary_path: Path | str,
) -> tuple[Path, Path]:
    outcomes_out = Path(outcomes_path)
    summary_out = Path(summary_path)
    outcomes_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(outcomes_out, index=False)
    summary.to_csv(summary_out, index=False)
    logger.info("Saved final detection outcomes to %s", outcomes_out)
    logger.info("Saved final outcome summary to %s", summary_out)
    return outcomes_out, summary_out

