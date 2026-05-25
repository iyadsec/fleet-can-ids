"""Final fleet-level anomaly outcome decisions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

ISOLATED = "Isolated anomaly"
COORDINATED = "Fleet-level coordinated behavioural pattern"


def load_cluster_results(path: Path | str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fleet cluster results not found: {csv_path}")
    return pd.read_csv(csv_path)


def classify_final_outcomes(
    cluster_results: pd.DataFrame,
    *,
    similarity_threshold: float = 0.85,
    min_vehicles: int = 2,
) -> pd.DataFrame:
    """
    Classify each anomaly event as isolated or coordinated.

    A coordinated pattern requires a cluster containing anomalies from more
    than one vehicle and high behavioural similarity. Temporal proximity is
    intentionally not used.
    """
    required = {
        "event_id",
        "vehicle_model",
        "attack_type",
        "algorithm",
        "cluster_id",
        "n_vehicles_in_cluster",
        "mean_behavioural_similarity",
    }
    missing = required - set(cluster_results.columns)
    if missing:
        raise ValueError(f"Cluster results missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for event_id, group in cluster_results.groupby("event_id", sort=False):
        candidates = group[
            (group["cluster_id"].astype(int) != -1)
            & (group["n_vehicles_in_cluster"].astype(int) >= min_vehicles)
            & (group["mean_behavioural_similarity"].astype(float) >= similarity_threshold)
        ].copy()
        if "is_suspicious_campaign" in candidates.columns:
            candidates = candidates[candidates["is_suspicious_campaign"].astype(bool)]

        if candidates.empty:
            best = group.sort_values("mean_behavioural_similarity", ascending=False).iloc[0]
            outcome = ISOLATED
        else:
            # Prefer the strongest behavioural cluster supporting the fleet decision.
            best = candidates.sort_values(
                ["mean_behavioural_similarity", "n_vehicles_in_cluster"],
                ascending=[False, False],
            ).iloc[0]
            outcome = COORDINATED

        rows.append(
            {
                "event_id": event_id,
                "vehicle_model": best["vehicle_model"],
                "attack_type": best["attack_type"],
                "algorithm": best["algorithm"],
                "cluster_id": int(best["cluster_id"]),
                "n_vehicles_in_cluster": int(best["n_vehicles_in_cluster"]),
                "mean_behavioural_similarity": float(best["mean_behavioural_similarity"]),
                "final_classification": outcome,
            }
        )

    return pd.DataFrame(rows)


def summarize_final_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Aggregate final outcome counts for reporting."""
    if outcomes.empty:
        return pd.DataFrame(
            columns=["final_classification", "event_count", "vehicle_count", "attack_types"]
        )
    return (
        outcomes.groupby("final_classification", as_index=False)
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

