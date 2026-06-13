"""Audit existing experiment roots and select authoritative sources."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
BASE = Path("new_experiments/final_validated_runs")


def _run_info(metrics_path: Path) -> dict:
    if not metrics_path.exists():
        return {"run_count": 0, "seed_count": 0, "seeds": []}
    df = pd.read_csv(metrics_path)
    seeds = sorted(df["seed"].dropna().unique().tolist()) if "seed" in df.columns else []
    nodes = df["graph_nodes"].dropna().unique().tolist() if "graph_nodes" in df.columns else []
    return {
        "run_count": len(df),
        "seed_count": len(seeds),
        "seeds": seeds,
        "node_counts": nodes,
    }


def build_source_inventory(project_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    entries = [
        ("results/S0_benign_control", "S0", "final", "validation/final_validation_report.md", True, ""),
        ("results/S1_isolated", "S1", "final", "validation/final_validation_report.md", True, ""),
        ("results/S2_non_coordinated", "S2", "final", "validation/final_validation_report.md", True, ""),
        ("results/S3_strong_campaign", "S3", "final", "validation/final_validation_report.md", True, ""),
        ("results/S4_weak_campaign", "S4", "final", "validation/final_validation_report.md", True, ""),
        ("results/campaign_size", "campaign_size", "preliminary", "validation/phase3_campaign_size_validation.md", False, "variable node counts; superseded"),
        ("results/campaign_size_corrected", "campaign_size", "corrected_final", "validation/campaign_size_corrected/corrected_phase3_validation.md", True, ""),
        ("hierarchical_alignment", "hierarchical_S0_S4_campaign", "final", "validation/hierarchical_alignment_validation.md", True, ""),
        ("framework_ablation", "framework_ablation", "final", "validation/framework_ablation_validation.md", False, "supplementary; not primary scenario tables"),
        ("evaluation_correction", "evaluation_correction", "corrected_metrics", "validation/evaluation_correction_report.md", False, "campaign-size only; layered correction"),
        ("model_diversity", "model_diversity", "preliminary", "", False, "split leakage; excluded"),
        ("model_diversity_corrected", "model_diversity", "diagnostic", "", False, "superseded by model_diversity_final"),
        ("model_diversity_final", "model_diversity", "final", "validation/final_phase4_validation.md", False, "vehicle-model diversity; out of scope"),
        ("model_diversity_final_tuned", "model_diversity", "diagnostic", "", False, "gate tuning provisional; out of scope"),
        ("quick_test", "quick_test", "diagnostic", "", False, "smoke test seed 11 only"),
    ]
    for rel, experiment, status, val_rel, eligible, exclusion in entries:
        root = project_root / BASE / rel
        metrics = root / "run_level_metrics.csv"
        if experiment.startswith("S") and metrics.exists():
            info = _run_info(metrics)
        elif rel == "hierarchical_alignment":
            m = root / "results/fleet_campaign_metrics.csv"
            info = {"run_count": len(pd.read_csv(m)) if m.exists() else 0, "seed_count": 10, "seeds": REQUIRED_SEEDS, "node_counts": ["variable", 200]}
        elif rel.endswith("campaign_size_corrected"):
            info = _run_info(metrics)
            info["node_counts"] = [200]
        elif rel.endswith("campaign_size"):
            info = _run_info(metrics)
        else:
            info = _run_info(metrics) if metrics.exists() else {"run_count": 0, "seed_count": 0, "seeds": [], "node_counts": []}

        rows.append(
            {
                "artifact_path": str(BASE / rel),
                "experiment": experiment,
                "scenario": experiment if experiment.startswith("S") else "",
                "status": status,
                "validation_report": val_rel,
                "run_count": info["run_count"],
                "seed_count": info["seed_count"],
                "node_count_controlled": 200 if eligible and experiment == "campaign_size" else "variable",
                "local_fleet_separated": True if eligible else "",
                "eligible_for_final_publication": eligible,
                "exclusion_reason": exclusion,
            }
        )
    return pd.DataFrame(rows)


def write_source_selection_report(path: Path, inventory: pd.DataFrame) -> None:
    eligible = inventory[inventory["eligible_for_final_publication"] == True]  # noqa: E712
    lines = [
        "# Source selection report",
        "",
        "## Authoritative sources",
        "",
        "### S0–S4 scenario evaluation",
        "- Primary run metrics: `new_experiments/final_validated_runs/results/S{0-4}_*/run_level_metrics.csv`",
        "- Hierarchical aligned metrics: `new_experiments/final_validated_runs/hierarchical_alignment/results/`",
        "",
        "### Campaign-size sensitivity (corrected)",
        "- `new_experiments/final_validated_runs/results/campaign_size_corrected/run_level_metrics.csv`",
        "- Per-run scenario records: `.../campaign_size_corrected/runs/*/selected_source_records.csv` (200 nodes each)",
        "",
        "### Edge connectivity sensitivity",
        "- New authoritative runs under `new_experiments/final_publication_scenarios/results/edge_sensitivity/`",
        "- Fixed 200-node scenario records reused from `campaign_size_corrected` fcgnn runs (strong/weak, n=5)",
        "",
        "## Excluded roots",
        "",
    ]
    for _, row in inventory[inventory["eligible_for_final_publication"] == False].iterrows():  # noqa: E712
        lines.append(f"- **{row['artifact_path']}** ({row['status']}): {row['exclusion_reason']}")
    lines += [
        "",
        "## Policy",
        "",
        "- No aggregation across preliminary and corrected campaign-size versions.",
        "- Vehicle-model-diversity experiments excluded from this package.",
        "- Local (C1) and fleet (C2/C3) metrics remain separated.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
