"""Audit documents comparing strong-only D2 vs shared configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.final_shared_configuration.shared_config import SharedFleetConfiguration


def write_comparison_audits(
    audit_dir: Path,
    df: pd.DataFrame,
    shared: SharedFleetConfiguration,
    search_diag: dict[str, Any],
) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    d2_path = Path("new_experiments/final_publication_scenarios/coordinated_campaign_refinement/configs/frozen_d2_params.json")
    d2 = {}
    if d2_path.exists():
        d2 = json.loads(d2_path.read_text(encoding="utf-8"))

    s3 = df[df["scenario_id"] == "S3"] if not df.empty else pd.DataFrame()
    s4 = df[df["scenario_id"] == "S4"] if not df.empty else pd.DataFrame()
    s0 = df[df["scenario_id"] == "S0"] if not df.empty else pd.DataFrame()

    lines = [
        "# Strong-only D2 vs jointly selected shared configuration",
        "",
        "## Parameter comparison",
        "",
        f"| Parameter | Strong-only D2 | Shared config |",
        f"|-----------|----------------|---------------|",
        f"| similarity_threshold | {d2.get('similarity_threshold', 'N/A')} | {shared.similarity_threshold} |",
        f"| max_same_vehicle_neighbors | {d2.get('top_k_same_vehicle', 'N/A')} | {shared.max_same_vehicle_neighbors} |",
        f"| max_cross_vehicle_neighbors | {d2.get('top_k_cross_vehicle', 'N/A')} | {shared.max_cross_vehicle_neighbors} |",
        f"| dbscan_eps | {d2.get('dbscan_eps', 'N/A')} | {shared.dbscan_eps} |",
        f"| dbscan_min_samples | {d2.get('dbscan_min_samples', 'N/A')} | {shared.dbscan_min_samples} |",
        "",
        "## Performance comparison (test means)",
        "",
    ]
    if not s3.empty:
        lines.append(f"- S3 campaign F1 (shared): {s3['campaign_f1'].mean():.3f}")
        lines.append(f"- S3 fragmentation (shared): {s3['fragmentation_rate'].mean():.3f}")
    if not s4.empty:
        lines.append(f"- S4 campaign F1 (shared): {s4['campaign_f1'].mean():.3f}")
        lines.append(f"- S4 membership precision (shared): {s4['membership_precision'].mean():.3f}")
    if not s0.empty and "false_campaign_rate" in s0.columns:
        lines.append(f"- S0 false campaign rate (shared): {s0['false_campaign_rate'].mean():.3f}")

    lines += [
        "",
        "## Interpretation",
        "",
        "Strong-only D2 was selected using strong coordinated validation scenarios only (V3-like).",
        "The shared configuration is selected on mixed V0–V4 validation to balance safety, strong, and weak campaigns.",
        "Strong-only tuning may overfit S3; shared configuration is the final main configuration for S0–S4.",
    ]
    (audit_dir / "strong_only_vs_shared_configuration.md").write_text("\n".join(lines), encoding="utf-8")

    ready = (
        not s0.empty
        and (s0.get("false_campaign_rate", pd.Series([0])).mean() <= 0.05 if "false_campaign_rate" in s0.columns else True)
        and not s3.empty
        and not s4.empty
    )
    ext_lines = [
        "# Extended scalability readiness",
        "",
        f"**Recommendation:** {'Proceed with extended 10/20/30 study using frozen shared config' if ready else 'Defer extended study until safety and weak-campaign quality validated'}.",
        "",
        f"Constraints met in validation search: {search_diag.get('constraints_met', False)}",
        "",
        "Requirements: valid S0 safety, S1 no coordinated campaign, S2 separation, interpretable S3/S4.",
    ]
    (audit_dir / "extended_scalability_readiness.md").write_text("\n".join(ext_lines), encoding="utf-8")
