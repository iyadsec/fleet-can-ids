"""Research evidence tables and figures for fleet-aware CAN IDS."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _count_csv_rows(path: Path | str) -> int:
    """Count data rows in a CSV without assuming particular column names."""
    return int(len(pd.read_csv(path)))


def save_raw_vs_descriptor_size(
    window_features_path: Path | str,
    descriptors_path: Path | str,
    output_path: Path | str,
    figure_path: Path | str,
    *,
    window_size: int = 100,
    estimated_frame_bytes: int = 16,
) -> pd.DataFrame:
    """Compare estimated raw CAN window payload size with descriptor table size."""
    total_raw_windows = _count_csv_rows(window_features_path)
    total_descriptors = _count_csv_rows(descriptors_path)
    estimated_raw_bytes = int(total_raw_windows * window_size * estimated_frame_bytes)
    desc_path = Path(descriptors_path)
    estimated_descriptor_bytes = int(desc_path.stat().st_size if desc_path.exists() else 0)
    compression_ratio = (
        estimated_raw_bytes / estimated_descriptor_bytes if estimated_descriptor_bytes else 0.0
    )
    percentage_reduction = (
        100.0 * (1.0 - (estimated_descriptor_bytes / estimated_raw_bytes))
        if estimated_raw_bytes
        else 0.0
    )
    out = pd.DataFrame(
        [
            {
                "total_raw_windows": total_raw_windows,
                "total_anomaly_descriptors": total_descriptors,
                "estimated_raw_bytes": estimated_raw_bytes,
                "estimated_descriptor_bytes": estimated_descriptor_bytes,
                "compression_ratio": round(compression_ratio, 6),
                "percentage_reduction": round(percentage_reduction, 6),
            }
        ]
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    fig_out = Path(figure_path)
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(["Raw windows", "Descriptors"], [estimated_raw_bytes, estimated_descriptor_bytes])
    ax.set_ylabel("Estimated bytes")
    ax.set_title("Raw CAN window size vs descriptor size")
    fig.tight_layout()
    fig.savefig(fig_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved raw-vs-descriptor size comparison to %s", output)
    return out


def build_fleet_value_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    total = int(len(outcomes))
    weak = int((outcomes["evidence_level"] == "weak_suspicious_signal").sum())
    upgraded = int(outcomes["was_upgraded_by_fleet"].astype(bool).sum())
    fleet = int((outcomes["final_outcome"] == "Fleet-level coordinated behavioural pattern").sum())
    row = {
        "total_events": total,
        "total_strong_local_anomalies": int((outcomes["evidence_level"] == "strong_local_anomaly").sum()),
        "total_weak_suspicious_signals": weak,
        "total_isolated_anomalies": int((outcomes["final_outcome"] == "Isolated anomaly").sum()),
        "total_weak_isolated_signals": int((outcomes["final_outcome"] == "Weak isolated signal").sum()),
        "total_fleet_level_patterns": fleet,
        "percentage_events_fleet_level": round((fleet / total) * 100.0, 6) if total else 0.0,
        "percentage_weak_signals_upgraded": round((upgraded / weak) * 100.0, 6) if weak else 0.0,
    }
    return pd.DataFrame([row])


def build_weak_signal_upgrade_summary(outcomes: pd.DataFrame) -> pd.DataFrame:
    weak_df = outcomes[outcomes["evidence_level"] == "weak_suspicious_signal"].copy()
    upgraded = weak_df[weak_df["was_upgraded_by_fleet"].astype(bool)]
    total_weak = int(len(weak_df))
    row = {
        "total_weak_suspicious_signals": total_weak,
        "weak_signals_not_alerted_locally": int((weak_df["local_alert"].astype(int) == 0).sum()),
        "weak_signals_upgraded_by_fleet": int(len(upgraded)),
        "upgrade_percentage": round((len(upgraded) / total_weak) * 100.0, 6) if total_weak else 0.0,
        "vehicles_involved": ",".join(sorted(upgraded["vehicle_model"].dropna().unique())) or "none",
        "attack_types_involved": ",".join(sorted(upgraded["attack_type"].dropna().unique())) or "none",
    }
    return pd.DataFrame([row])


def build_cross_vehicle_cluster_summary(clusters: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    cross = clusters[clusters["is_cross_vehicle_cluster"].astype(bool)].copy()
    if cross.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_size",
                "num_unique_vehicles",
                "vehicles_in_cluster",
                "dominant_attack_type",
                "mean_cluster_similarity",
                "num_strong_local_anomalies",
                "num_weak_suspicious_signals",
                "num_upgraded_weak_signals",
            ]
        )
    upgraded_ids = set(outcomes.loc[outcomes["was_upgraded_by_fleet"].astype(bool), "event_id"])
    rows: list[dict[str, Any]] = []
    for cid, group in cross.groupby("cluster_id", sort=True):
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_size": int(group["cluster_size"].iloc[0]),
                "num_unique_vehicles": int(group["num_unique_vehicles"].iloc[0]),
                "vehicles_in_cluster": group["vehicles_in_cluster"].iloc[0],
                "dominant_attack_type": group["dominant_attack_type"].iloc[0],
                "mean_cluster_similarity": float(group["mean_cluster_similarity"].iloc[0]),
                "num_strong_local_anomalies": int(group["local_alert"].astype(int).sum()),
                "num_weak_suspicious_signals": int(group["weak_signal"].astype(int).sum()),
                "num_upgraded_weak_signals": int(group["event_id"].isin(upgraded_ids).sum()),
            }
        )
    return pd.DataFrame(rows)


def save_research_evidence_summaries(
    outcomes_path: Path | str,
    clusters_path: Path | str,
    *,
    metrics_dir: Path | str,
) -> dict[str, Path]:
    outcomes = pd.read_csv(outcomes_path)
    clusters = pd.read_csv(clusters_path)
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "fleet_value_summary": build_fleet_value_summary(outcomes),
        "weak_signal_upgrade_summary": build_weak_signal_upgrade_summary(outcomes),
        "cross_vehicle_cluster_summary": build_cross_vehicle_cluster_summary(clusters, outcomes),
    }
    written: dict[str, Path] = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        written[name] = path
    return written


def save_research_figures(
    outcomes_path: Path | str,
    clusters_path: Path | str,
    *,
    figures_dir: Path | str,
) -> dict[str, Path]:
    outcomes = pd.read_csv(outcomes_path)
    clusters = pd.read_csv(clusters_path)
    fig_dir = Path(figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def savefig(name: str) -> Path:
        path = fig_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        written[name] = path
        return path

    plt.figure(figsize=(8, 4))
    outcomes["final_outcome"].value_counts().plot(kind="bar")
    plt.title("Local vs fleet final outcomes")
    plt.ylabel("Event count")
    savefig("local_vs_fleet_outcomes.png")

    plt.figure(figsize=(6, 4))
    weak = outcomes[outcomes["evidence_level"] == "weak_suspicious_signal"]
    weak_counts = weak["was_upgraded_by_fleet"].value_counts().rename(
        {True: "Upgraded", False: "Not upgraded"}
    )
    if weak_counts.empty:
        weak_counts = pd.Series({"Upgraded": 0, "Not upgraded": 0})
    weak_counts.plot(kind="bar")
    plt.title("Weak suspicious signal upgrades")
    plt.ylabel("Event count")
    savefig("weak_signal_upgrade_chart.png")

    plt.figure(figsize=(8, 4))
    cross = clusters[clusters["is_cross_vehicle_cluster"].astype(bool)]
    if not cross.empty:
        cross.drop_duplicates("cluster_id")["dominant_attack_type"].value_counts().plot(kind="bar")
    plt.title("Cross-vehicle clusters by attack type")
    plt.ylabel("Cluster count")
    savefig("cross_vehicle_clusters_by_attack_type.png")

    plt.figure(figsize=(8, 4))
    if not cross.empty:
        cross.drop_duplicates("cluster_id")["num_unique_vehicles"].value_counts().sort_index().plot(kind="bar")
    plt.title("Fleet cluster vehicle distribution")
    plt.xlabel("Vehicles in cluster")
    plt.ylabel("Cluster count")
    savefig("fleet_cluster_vehicle_distribution.png")
    return written
