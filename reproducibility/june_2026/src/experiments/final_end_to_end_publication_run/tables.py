"""Publication tables P1–P12."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.final_shared_configuration.shared_config import SharedFleetConfiguration


def _bundle(df: pd.DataFrame, stem: Path, title: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(f"# {title}\n\n{df.to_markdown(index=False)}", encoding="utf-8")
    try:
        stem.with_suffix(".tex").write_text(df.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
    except Exception:
        pass


def generate_tables(
    out_dir: Path,
    *,
    df: pd.DataFrame,
    shared: SharedFleetConfiguration,
    trace_df: pd.DataFrame,
    vehicle_metrics: pd.DataFrame,
    descriptor_metrics: pd.DataFrame,
    master_hash: str,
) -> list[str]:
    out_dir = Path(out_dir)
    generated: list[str] = []

    p1 = pd.DataFrame(
        [
            {"objective": "Vehicle-level anomaly detection", "metric_family": "ROC-AUC, PR-AUC, F1, FPR"},
            {"objective": "Descriptor compactness and privacy", "metric_family": "compression, bandwidth, exposure"},
            {"objective": "Benign-fleet safety", "metric_family": "false campaign rate"},
            {"objective": "Coordinated campaign detection", "metric_family": "campaign F1, membership F1"},
            {"objective": "Graph connectivity sensitivity", "metric_family": "F1 vs unique edges"},
        ]
    )
    _bundle(p1, out_dir / "table_P1_experimental_objectives", "P1 Experimental objectives")
    generated.append("P1")

    split_summary = trace_df.groupby(["vehicle_model", "split"]).agg(
        trace_count=("source_trace", "count"),
        windows=("window_count", "sum"),
    ).reset_index()
    split_summary["config_hash"] = master_hash
    _bundle(split_summary, out_dir / "table_P2_dataset_and_split_summary", "P2 Dataset and split")
    generated.append("P2")

    p3 = pd.DataFrame([{"parameter": k, "value": v} for k, v in shared.to_dict().items()])
    p3.loc[len(p3)] = {"parameter": "master_config_hash", "value": master_hash}
    _bundle(p3, out_dir / "table_P3_final_experimental_configuration", "P3 Configuration")
    generated.append("P3")

    if not vehicle_metrics.empty:
        _bundle(vehicle_metrics, out_dir / "table_P4_vehicle_level_results", "P4 Vehicle-level results")
        generated.append("P4")

    if not descriptor_metrics.empty:
        _bundle(descriptor_metrics, out_dir / "table_P5_descriptor_compactness_and_privacy", "P5 Descriptor analysis")
        generated.append("P5")

    if not df.empty and "test_condition" in df.columns:
        safety = df[df["test_condition"].isin(
            ["Benign-Fleet Control", "Isolated Single-Vehicle Attack", "Unrelated Multi-Vehicle Incidents"]
        )]
        if not safety.empty:
            p6 = safety.groupby("test_condition").mean(numeric_only=True).reset_index()
            p6_cols = [c for c in ["test_condition", "local_detection_rate", "false_campaign_rate", "incorrect_merging"] if c in p6.columns]
            _bundle(p6[p6_cols], out_dir / "table_P6_benign_isolated_unrelated_results", "P6 Safety scenarios")
            generated.append("P6")

        strong = df[df["test_condition"] == "Strong Coordinated Campaign"]
        if not strong.empty:
            p7 = strong.groupby("campaign_size").mean(numeric_only=True).reset_index()
            _bundle(p7, out_dir / "table_P7_strong_campaign_results", "P7 Strong campaign")
            generated.append("P7")

        weak = df[df["test_condition"] == "Weak Coordinated Campaign"]
        if not weak.empty:
            p8 = weak.groupby("campaign_size").mean(numeric_only=True).reset_index()
            _bundle(p8, out_dir / "table_P8_weak_campaign_results", "P8 Weak campaign")
            generated.append("P8")

        cs = df[df["test_condition"].str.contains("Campaign")]
        if not cs.empty:
            p9_cols = [c for c in cs.columns if c in (
                "campaign_size", "graph_nodes", "unique_edges", "graph_build_time", "end_to_end_latency", "peak_memory"
            )]
            if len(p9_cols) > 1:
                _bundle(cs.groupby(["test_condition", "campaign_size"])[p9_cols[1:]].mean().reset_index(), out_dir / "table_P9_campaign_size_graph_and_cost", "P9 Campaign size cost")
                generated.append("P9")

    stats_path = out_dir.parent / "statistics" / "primary_statistical_tests.csv"
    if stats_path.exists() and stats_path.stat().st_size > 10:
        _bundle(pd.read_csv(stats_path), out_dir / "table_P12_primary_statistical_tests", "P12 Statistical tests")
        generated.append("P12")

    edge_path = out_dir.parent / "results" / "edge_sensitivity" / "run_level_metrics.csv"
    if edge_path.exists():
        edge = pd.read_csv(edge_path)
        if not edge.empty:
            _bundle(edge.groupby("unique_edges").mean(numeric_only=True).reset_index(), out_dir / "table_P10_edge_connectivity_performance", "P10 Edge performance")
            _bundle(edge[["graph_build_time", "gnn_inference_time", "end_to_end_latency"]].describe().reset_index(), out_dir / "table_P11_edge_connectivity_cost", "P11 Edge cost")
            generated.extend(["P10", "P11"])

    return generated
