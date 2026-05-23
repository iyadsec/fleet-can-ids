"""Generate a consolidated markdown report for the full experiment pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _section(title: str, body: str) -> str:
    return f"## {title}\n\n{body.strip()}\n\n"


def _df_summary_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df.empty:
        return "_No data._\n"
    sub = df.head(max_rows)
    cols = sub.columns.tolist()
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_({len(df) - max_rows} more rows omitted)_")
    return "\n".join(lines) + "\n"


def generate_pipeline_report(
    *,
    config: dict[str, Any],
    step_results: dict[str, dict[str, Any]],
    output_path: Path | str,
) -> Path:
    """Write a markdown summary of pipeline artefacts and step outcomes."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    artifacts = config.get("pipeline", {}).get("artifacts", {})
    root = Path(config.get("_project_root", "."))

    lines = [
        "# Fleet CAN-IDS — Pipeline Report",
        "",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        f"**Project:** {config.get('project', {}).get('name', 'fleet-can-ids')}",
        f"**Seed:** {config.get('project', {}).get('seed', '—')}",
        "",
    ]

    # Step timeline
    timeline = []
    for name, info in step_results.items():
        status = info.get("status", "unknown")
        elapsed = info.get("elapsed_sec")
        extra = f" ({elapsed:.1f}s)" if isinstance(elapsed, (int, float)) else ""
        timeline.append(f"- **{name}**: {status}{extra}")
    lines.append(_section("Pipeline steps", "\n".join(timeline)))

    # Dataset / windows / features
    clean = root / artifacts.get("clean_can_data", "data/processed/clean_can_data.csv")
    if clean.exists():
        try:
            n_frames = sum(1 for _ in open(clean, encoding="utf-8")) - 1
            lines.append(_section("Dataset", f"- CAN frames: **{n_frames:,}** (`{clean.name}`)"))
        except OSError:
            pass

    for key, title in [
        ("window_metadata", "Windows"),
        ("window_features", "Features"),
        ("anomaly_descriptors", "Anomaly descriptors"),
    ]:
        p = root / artifacts.get(key, "")
        if p.exists():
            try:
                n = sum(1 for _ in open(p, encoding="utf-8")) - 1
                lines.append(_section(title, f"- Rows: **{n:,}** (`{p}`)"))
            except OSError:
                pass

    # Vehicle IDS
    ids_path = root / artifacts.get("vehicle_results", "outputs/metrics/vehicle_level_results.csv")
    if ids_path.exists():
        ids_df = pd.read_csv(ids_path)
        lines.append(_section("Vehicle-level IDS", _df_summary_table(ids_df)))

    # Graph
    graph_stats = _read_json(
        root / artifacts.get("graph_stats", "outputs/metrics/fleet_graph_stats.json")
    )
    if graph_stats:
        body = "\n".join(f"- **{k}**: {v}" for k, v in graph_stats.items() if k != "history")
        lines.append(_section("Fleet graph", body))

    # GNN
    gnn_metrics = _read_json(
        root / artifacts.get("gnn_metrics", "outputs/metrics/gnn_training_metrics.json")
    )
    if gnn_metrics:
        body = "\n".join(
            f"- **{k}**: {v}"
            for k, v in gnn_metrics.items()
            if k != "history"
        )
        lines.append(_section("GNN training", body))

    # Clustering
    cluster_path = root / artifacts.get("campaign_clusters", "outputs/metrics/campaign_clusters.csv")
    if cluster_path.exists():
        cdf = pd.read_csv(cluster_path)
        sus = cdf[cdf["is_suspicious_campaign"]].drop_duplicates(
            subset=["algorithm", "cluster_id"]
        )
        lines.append(
            _section(
                "Campaign clustering",
                f"- Total assignment rows: **{len(cdf):,}**\n"
                f"- Suspicious multi-vehicle clusters:\n\n{_df_summary_table(sus)}",
            )
        )

    # Configuration snapshot
    lines.append(_section("Configuration", f"```yaml\n# See configs/default.yaml\n```"))

    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote pipeline report to %s", out)
    return out
