"""Generate audit markdown documents."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.campaign_evaluation import compute_campaign_metrics
from src.experiments.model_diversity_final.campaign_gate import CampaignGateConfig
from src.experiments.model_diversity_final_tuned.false_campaign_metrics import legacy_false_campaign_rate_explanation


def write_false_campaign_metric_definition(path: Path) -> None:
    lines = [
        "# False campaign metric definition",
        "",
        "Four error types are reported separately. They must not be combined into one ambiguous rate.",
        "",
        "## A. False campaign alert",
        "",
        "A predicted multi-vehicle campaign exists when no ground-truth campaign exists (`n_gt == 0` and `n_accepted > 0`).",
        "",
        "## B. Benign membership contamination",
        "",
        "A real campaign is detected, but benign vehicles are incorrectly included in fleet campaign membership.",
        "",
        "## C. Extra false campaign cluster",
        "",
        "A valid campaign is detected, but one or more unrelated campaign clusters are also generated: `max(n_accepted - n_gt, 0)`.",
        "",
        "## D. Incorrect merging",
        "",
        "Unrelated incidents are merged into one campaign (`n_gt > 1` and `n_accepted == 1`).",
        "",
        "## Legacy metric bug (provisional Phase 4)",
        "",
        "In `compute_campaign_metrics`, when `spec_expects_campaign=True`:",
        "",
        "```",
        "false_campaign_alert_rate = n_detected / max(n_detected, 1)",
        "```",
        "",
        "Therefore any run with at least one qualifying DBSCAN cluster reports `false_campaign_alert_rate ≈ 1.0`,",
        "regardless of ground-truth campaign presence. This is a **metric semantics** issue, not purely gate failure.",
        "",
        f"Example when n_detected=2, expect_campaign=True: {legacy_false_campaign_rate_explanation(2, True)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_current_campaign_gate_audit(path: Path) -> None:
    gate = CampaignGateConfig()
    lines = [
        "# Current campaign gate audit (provisional)",
        "",
        "## CampaignGateConfig defaults",
        "",
        f"- min_distinct_vehicles: {gate.min_distinct_vehicles}",
        f"- min_anomalous_member_ratio: {gate.min_anomalous_member_ratio}",
        f"- max_weak_only_ratio: {gate.max_weak_only_ratio}",
        f"- min_membership_confidence: {gate.min_membership_confidence}",
        f"- min_cluster_cohesion: {gate.min_cluster_cohesion}",
        f"- min_cross_vehicle_edges: {gate.min_cross_vehicle_edges}",
        f"- require_cross_model_path: {gate.require_cross_model_path}",
        f"- max_benign_vehicle_inclusion: {gate.max_benign_vehicle_inclusion}",
        "",
        "## Decision flow (provisional)",
        "",
        "1. DBSCAN clusters on similarity (C2) or GraphSAGE embeddings (C3).",
        "2. `_qualify_clusters_ieee` marks multi-vehicle clusters as qualifying.",
        "3. `gate_qualifying_clusters` filters clusters by vehicle count, anomalous ratio, cohesion, cross-model edges.",
        "4. `_assign_gated_decisions` sets coordinated membership when cluster passes gate AND (local_strong OR score >= min_membership_confidence).",
        "",
        "**Issue:** Campaign acceptance and member acceptance are combined in one step.",
        "",
        "## Root cause of high provisional false campaign rate",
        "",
        "Primary: **metric implementation** — legacy `false_campaign_alert_rate` equals 1.0 whenever any qualifying cluster exists.",
        "",
        "Secondary contributors:",
        "- Permissive campaign acceptance (min_distinct_vehicles=3, min_anomalous_ratio=0.40)",
        "- Member promotion via weak_signal + coordinated decision path",
        "- No separate benign-support ratio at campaign level",
        "",
        "## Tuned pipeline changes",
        "",
        "- Separate `accept_campaign_clusters` and `accept_cluster_members`",
        "- Corrected false-campaign semantics (A–D)",
        "- Validation-only grid search with constrained objectives",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def recompute_provisional_false_campaigns(provisional_root: Path, output_csv: Path) -> pd.DataFrame:
    from src.experiments.model_diversity_final_tuned.false_campaign_metrics import compute_false_campaign_breakdown

    rows = []
    for run_dir in sorted(provisional_root.rglob("runs")):
        if not run_dir.is_dir() or "dry_" in run_dir.name:
            continue
        ep = run_dir / "event_predictions.csv"
        cd = run_dir / "cluster_assignments.csv"
        if not ep.exists():
            ep = run_dir / "event_predictions.csv"
        sr = run_dir / "selected_source_records.csv"
        if not ep.exists() or not sr.exists():
            continue
        events = pd.read_csv(ep)
        scenario = pd.read_csv(sr)
        from src.experiments.campaign_analysis_corrected import _build_membership

        seed = int(run_dir.name.split("seed")[-1].split("_")[0]) if "seed" in run_dir.name else 0
        strength = "strong" if "strong" in run_dir.name else "weak"
        membership = _build_membership(scenario, seed, 5, 1.0, strength)
        if "cluster_id" in events.columns:
            cluster_df = (
                events[events["cluster_id"] >= 0]
                .groupby("cluster_id", as_index=False)
                .agg(
                    vehicles_in_cluster=("vehicles_in_cluster", "max"),
                    behavioral_cohesion=("behavioral_cohesion", "max"),
                )
            )
            cluster_df["is_qualifying_campaign_cluster"] = cluster_df["vehicles_in_cluster"] >= 3
            cluster_df["campaign_accepted"] = events.groupby("cluster_id")["campaign_gate_passed"].max().reindex(cluster_df["cluster_id"]).fillna(0).astype(bool).to_numpy() if "campaign_gate_passed" in events.columns else cluster_df["is_qualifying_campaign_cluster"]
        else:
            cluster_df = pd.DataFrame()
        legacy = compute_campaign_metrics(events, membership, cluster_df, True)
        corrected = compute_false_campaign_breakdown(events, membership, cluster_df, expect_campaign=True)
        rows.append({
            "run_id": run_dir.name,
            "method": events["method"].iloc[0] if "method" in events.columns else "",
            "legacy_false_campaign_alert_rate": legacy["false_campaign_alert_rate"],
            "corrected_false_campaign_alert_rate": corrected["false_campaign_alert_rate"],
            **{k: corrected[k] for k in (
                "false_campaign_alert_indicator", "false_campaign_cluster_count",
                "benign_vehicles_included", "extra_cluster_count", "incorrect_merging", "fragmentation",
            )},
        })
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df
