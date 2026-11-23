"""Tables CR1–CR6."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _bundle(df: pd.DataFrame, stem: Path, title: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(f"# {title}\n\n{df.to_markdown(index=False)}", encoding="utf-8")
    try:
        stem.with_suffix(".tex").write_text(df.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
    except Exception:
        pass


def generate_tables(out: Path, all_df: pd.DataFrame, stats: pd.DataFrame, frozen: dict) -> list[str]:
    cr1 = pd.DataFrame([
        {"Design": "D0", "Graph node rule": "All 200 descriptors", "Connectivity rule": "default k_same=10, k_cross=5, tau=0.95",
         "Same-vehicle edges": "yes", "Cross-vehicle edges": "yes", "Clustering rule": "default DBSCAN eps=0.8, min_samples=10"},
        {"Design": "D1", "Graph node rule": "anomaly_score >= weak_threshold (0.55)", "Connectivity rule": "same as D0",
         "Same-vehicle edges": "yes", "Cross-vehicle edges": "yes", "Clustering rule": "default DBSCAN eps=0.8, min_samples=10"},
        {"Design": "D2", "Graph node rule": "candidate-only", "Connectivity rule": f"tau={frozen.get('similarity_threshold')}, k_cross={frozen.get('top_k_cross_vehicle')}, k_same={frozen.get('top_k_same_vehicle')}",
         "Same-vehicle edges": "capped", "Cross-vehicle edges": "capped", "Clustering rule": f"eps={frozen.get('dbscan_eps')}, min_samples={frozen.get('dbscan_min_samples')}, consolidation={frozen.get('consolidate_fragments')}"},
    ])
    _bundle(cr1, out / "table_CR1_graph_design_definitions", "CR1 Graph design definitions")

    agg = all_df.groupby(["campaign_size", "design"]).mean(numeric_only=True).reset_index()
    cr2 = agg[[c for c in [
        "campaign_size", "design", "candidate_nodes", "benign_candidates", "campaign_precision",
        "campaign_recall", "campaign_f1", "membership_precision", "membership_recall",
        "benign_vehicles_included", "fragmentation_rate",
    ] if c in agg.columns]].rename(columns={
        "campaign_size": "Campaign size", "design": "Design", "candidate_nodes": "Candidate nodes",
        "benign_candidates": "Benign candidates", "campaign_precision": "Campaign precision",
        "campaign_recall": "Campaign recall", "campaign_f1": "Campaign F1",
        "membership_precision": "Membership precision", "membership_recall": "Membership recall",
        "benign_vehicles_included": "Benign vehicles included", "fragmentation_rate": "Fragmentation",
    })
    _bundle(cr2, out / "table_CR2_candidate_filtering_effect", "CR2 Candidate filtering effect")

    cr3_cols = ["campaign_size", "design", "unique_edges", "average_degree", "cross_vehicle_edge_percentage",
                "campaign_f1", "membership_f1", "fragmentation_rate", "incorrect_merging_rate"]
    cr3 = agg[[c for c in cr3_cols if c in agg.columns]].rename(columns={
        "campaign_size": "Campaign size", "design": "Design", "unique_edges": "Unique edges",
        "average_degree": "Average degree", "cross_vehicle_edge_percentage": "Cross-vehicle edge percentage",
        "campaign_f1": "Campaign F1", "membership_f1": "Membership F1",
        "fragmentation_rate": "Fragmentation", "incorrect_merging_rate": "Incorrect merging",
    })
    _bundle(cr3, out / "table_CR3_controlled_connectivity_effect", "CR3 Controlled connectivity")

    cr4_cols = ["campaign_size", "campaign_detection_rate", "attacked_vehicles_correctly_included",
                "membership_precision", "membership_recall", "cross_vehicle_edges",
                "false_campaign_cluster_count"]
    cr4 = agg[[c for c in cr4_cols if c in agg.columns]].rename(columns={
        "campaign_size": "Campaign size", "campaign_detection_rate": "Campaign detection rate",
        "attacked_vehicles_correctly_included": "Distinct attacked vehicles recovered",
        "membership_precision": "Membership precision", "membership_recall": "Membership recall",
        "cross_vehicle_edges": "Cross-vehicle support", "false_campaign_cluster_count": "False campaign clusters",
    })
    _bundle(cr4, out / "table_CR4_coordinated_campaign_detection", "CR4 Coordinated campaign detection")

    cr5_cols = ["campaign_size", "design", "candidate_nodes", "unique_edges", "graph_build_time",
                "gnn_inference_time", "end_to_end_latency", "peak_memory"]
    cr5 = agg[[c for c in cr5_cols if c in agg.columns]].rename(columns={
        "campaign_size": "Campaign size", "design": "Design", "candidate_nodes": "Candidate nodes",
        "unique_edges": "Unique edges", "graph_build_time": "Graph-build time",
        "gnn_inference_time": "Inference time", "end_to_end_latency": "End-to-end latency",
        "peak_memory": "Peak memory",
    })
    _bundle(cr5, out / "table_CR5_computational_cost", "CR5 Computational cost")

    if not stats.empty:
        _bundle(stats, out / "table_CR6_statistical_tests", "CR6 Statistical tests")

    return ["CR1", "CR2", "CR3", "CR4", "CR5", "CR6"]
