"""Publication tables FS1–FS8."""

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


def generate_tables(out: Path, df: pd.DataFrame, shared: SharedFleetConfiguration, cfg: dict) -> list[str]:
    fs1 = pd.DataFrame([
        {"Scenario": "S0", "Description": "Benign fleet", "Expected campaign": "none"},
        {"Scenario": "S1", "Description": "Isolated attack", "Expected campaign": "none"},
        {"Scenario": "S2", "Description": "Unrelated incidents", "Expected campaign": "none (separate)"},
        {"Scenario": "S3", "Description": "Strong coordinated", "Expected campaign": "one coordinated"},
        {"Scenario": "S4", "Description": "Weak coordinated", "Expected campaign": "one coordinated"},
    ])
    _bundle(fs1, out / "table_FS1_scenario_definitions", "FS1 Scenario definitions")

    fs2 = pd.DataFrame([
        {"Parameter": k, "Shared value": v, "Validation scenarios used": "V0–V4 mixed",
         "Selection rationale": "Joint validation objective on strong+weak campaign F1 and membership"}
        for k, v in shared.to_dict().items()
    ])
    _bundle(fs2, out / "table_FS2_shared_fleet_configuration", "FS2 Shared fleet configuration")

    if df.empty:
        return ["FS1", "FS2"]

    agg = df.groupby(["scenario_id", "campaign_size"]).mean(numeric_only=True).reset_index()
    fs3_cols = [c for c in ["scenario_id", "campaign_size", "source_windows", "candidate_nodes",
                            "transmission_percentage", "malicious_candidates", "benign_candidates",
                            "candidate_precision", "candidate_recall"] if c in agg.columns]
    fs3 = agg[fs3_cols].rename(columns={"scenario_id": "Scenario", "campaign_size": "Campaign size",
                                        "source_windows": "Source windows", "candidate_nodes": "Fleet candidates",
                                        "transmission_percentage": "Transmission rate"})
    _bundle(fs3, out / "table_FS3_local_filtering_and_transmission", "FS3 Local filtering")

    safety = df[df["scenario_id"].isin(["S0", "S1", "S2"])]
    if not safety.empty:
        fs4_cols = [c for c in ["scenario_id", "local_detection_rate", "false_campaign_rate", "incorrect_merging"] if c in safety.columns]
        if len(fs4_cols) > 1:
            fs4 = safety.groupby("scenario_id").mean(numeric_only=True).reset_index()[fs4_cols].rename(columns={"scenario_id": "Scenario"})
            fs4["Correct final decision rate"] = 1.0 - fs4.get("false_campaign_rate", 0)
            _bundle(fs4, out / "table_FS4_S0_S2_safety_results", "FS4 Safety results")

    s3 = df[df["scenario_id"] == "S3"]
    if not s3.empty:
        fs5 = s3.groupby("campaign_size").mean(numeric_only=True).reset_index()
        _bundle(fs5, out / "table_FS5_strong_campaign_results", "FS5 Strong campaign")

    s4 = df[df["scenario_id"] == "S4"]
    if not s4.empty:
        fs6 = s4.groupby("campaign_size").mean(numeric_only=True).reset_index()
        _bundle(fs6, out / "table_FS6_weak_campaign_results", "FS6 Weak campaign")

    fs7_cols = [c for c in ["scenario_id", "campaign_size", "graph_nodes", "unique_edges",
                            "cross_vehicle_edge_percentage", "graph_build_time", "gnn_inference_time",
                            "end_to_end_latency", "peak_memory"] if c in df.columns]
    fs7 = df.groupby(["scenario_id", "campaign_size"])[fs7_cols[2:]].mean().reset_index() if len(fs7_cols) > 2 else pd.DataFrame()
    if not fs7.empty:
        _bundle(fs7, out / "table_FS7_graph_and_cost", "FS7 Graph and cost")

    stats_path = out.parent / "statistical_analysis" / "statistical_tests.csv"
    if stats_path.exists() and stats_path.stat().st_size > 0:
        _bundle(pd.read_csv(stats_path), out / "table_FS8_statistical_tests", "FS8 Statistical tests")

    return ["FS1", "FS2", "FS3", "FS4", "FS5", "FS6", "FS7", "FS8"]
