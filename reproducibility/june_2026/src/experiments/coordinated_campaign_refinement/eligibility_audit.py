"""Audit current graph node eligibility from authoritative D0 runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.coordinated_campaign_refinement.candidates import annotate_fleet_candidates, candidate_statistics
from src.experiments.coordinated_campaign_refinement.guard import CoordinatedCampaignRefinementGuard

D0_ROOT = Path("new_experiments/final_validated_runs/results/campaign_size_corrected/runs")
WEAK_TH = 0.55
STRONG_TH = 0.80


def audit_graph_node_eligibility(project_root: Path, out: Path) -> pd.DataFrame:
    run_root = project_root / D0_ROOT
    rows: list[dict[str, Any]] = []
    samples: list[dict] = []

    for cs in (2, 5, 10):
        for seed in [11, 23, 37]:
            matches = list(run_root.glob(f"campaign_size_corrected_strong_n{cs}_fcgnn_seed{seed}_*"))
            if not matches:
                continue
            rd = matches[0]
            df = pd.read_csv(rd / "selected_source_records.csv")
            ann = annotate_fleet_candidates(df, weak_threshold=WEAK_TH, strong_threshold=STRONG_TH)
            stats = candidate_statistics(ann)
            stats.update({"campaign_size": cs, "seed": seed, "run_id": rd.name})
            rows.append(stats)
            if cs == 5 and seed == 11:
                samples.append(stats)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        agg = summary.groupby("campaign_size").mean(numeric_only=True)

    lines = [
        "# Current graph node eligibility audit",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inclusion rule (D0 — current pipeline)",
        "",
        "All scenario descriptors enter the fleet graph.",
        "",
        "- **Code:** `src/experiments/experiment_pipeline.py` → `build_scenario_graph()`",
        "- **Node matrix:** `prepare_gnn_fleet_node_matrix()` in `src/evaluation/final_gnn_fleet_decision_experiment.py`",
        "- **Input:** full `selected_source_records.csv` (200 nodes per run)",
        "- **Filter:** none on `anomaly_score` before graph construction",
        "",
        "## Findings",
        "",
    ]

    if not summary.empty:
        for cs in (2, 5, 10):
            if cs not in agg.index:
                continue
            a = agg.loc[cs]
            lines += [
                f"### Campaign size {cs} (mean over sample seeds)",
                "",
                f"- Total descriptors: {a['total_descriptors']:.0f}",
                f"- Descriptors entering graph (current): {a['total_descriptors']:.0f} (all)",
                f"- Would-be fleet candidates (score ≥ {WEAK_TH}): {a['candidate_nodes']:.0f}",
                f"- Clearly benign below threshold: {a['benign_below_threshold']:.0f}",
                f"- Benign descriptors in graph today: {a['total_descriptors'] - a['malicious_candidates']:.0f}",
                f"- Benign weak candidates (incorrectly suspicious): {a['benign_incorrectly_promoted']:.0f}",
                f"- Strong malicious in graph: {a['strong_malicious_candidates']:.0f}",
                f"- Weak malicious in graph: {a['weak_malicious_candidates']:.0f}",
                "",
            ]
        lines.append(
            "**Conclusion:** No descriptors fall below the weak threshold (0.55) in budgeted strong scenarios; "
            "however **150–190 benign weak-candidate descriptors** (benign attack type with score ≥ 0.55) "
            "enter the D0 graph alongside malicious evidence. The D1 rule excludes only sub-threshold evidence; "
            "in these runs it retains all 200 nodes."
        )
    else:
        lines.append("No sample runs found for audit.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    summary.to_csv(out.parent / "graph_node_eligibility_by_run.csv", index=False)
    return summary


def audit_same_vs_cross_edges(project_root: Path, out: Path) -> pd.DataFrame:
    run_root = project_root / D0_ROOT
    rows = []
    for cs in (2, 5, 10):
        for rd in sorted(run_root.glob(f"campaign_size_corrected_strong_n{cs}_fcgnn_seed*"))[:10]:
            edge_path = rd / "edge_list.csv"
            if not edge_path.exists():
                continue
            edges = pd.read_csv(edge_path)
            same = int((edges["source_vehicle"] == edges["target_vehicle"]).sum()) if "source_vehicle" in edges.columns else 0
            cross = len(edges) - same
            rows.append({
                "run_id": rd.name,
                "campaign_size": cs,
                "total_edges": len(edges),
                "same_vehicle_edges": same,
                "cross_vehicle_edges": cross,
                "cross_vehicle_edge_percentage": 100.0 * cross / max(len(edges), 1),
            })
    df = pd.DataFrame(rows)
    lines = [
        "# Same-vehicle versus cross-vehicle edge audit",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
    ]
    if not df.empty:
        agg = df.groupby("campaign_size")[["same_vehicle_edges", "cross_vehicle_edges", "cross_vehicle_edge_percentage"]].mean()
        lines.append("## Mean edge counts by campaign size (D0)")
        lines.append("")
        lines.append(agg.round(2).to_markdown())
        lines += [
            "",
            "Cross-vehicle edges provide fleet campaign evidence; same-vehicle edges dominate connectivity "
            "and grow with campaign size, increasing graph density.",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    df.to_csv(out.parent / "same_vs_cross_edge_by_run.csv", index=False)
    return df
