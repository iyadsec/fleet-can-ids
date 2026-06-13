"""Publication tables T1–T10."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _bundle(df: pd.DataFrame, stem: Path, title: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    (stem.with_suffix(".md")).write_text(f"# {title}\n\n{df.to_markdown(index=False)}", encoding="utf-8")
    try:
        stem.with_suffix(".tex").write_text(df.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
    except Exception:
        pass


def generate_all_tables(
    out: Path,
    *,
    safety: pd.DataFrame,
    fleet: pd.DataFrame,
    weak: pd.DataFrame,
    campaign_strong: pd.DataFrame,
    campaign_weak: pd.DataFrame,
    campaign_cost: pd.DataFrame,
    edge_perf: pd.DataFrame,
    edge_cost: pd.DataFrame,
    stats: pd.DataFrame,
) -> list[str]:
    generated = []
    t1 = pd.DataFrame([
        {"Scenario": "S0", "Attacked vehicles": 0, "Coordination": "none", "Local evidence": "none",
         "Expected local result": "no alert", "Expected fleet result": "no campaign"},
        {"Scenario": "S1", "Attacked vehicles": 1, "Coordination": "none", "Local evidence": "strong",
         "Expected local result": "local alert", "Expected fleet result": "isolated incident"},
        {"Scenario": "S2", "Attacked vehicles": "≥2", "Coordination": "behaviourally unrelated",
         "Local evidence": "strong", "Expected local result": "local alerts", "Expected fleet result": "separate incidents"},
        {"Scenario": "S3", "Attacked vehicles": "variable", "Coordination": "behavioural campaign",
         "Local evidence": "strong", "Expected local result": "local alerts", "Expected fleet result": "coordinated campaign"},
        {"Scenario": "S4", "Attacked vehicles": "variable", "Coordination": "behavioural campaign",
         "Local evidence": "weak", "Expected local result": "weak/inconclusive", "Expected fleet result": "fleet-correlated campaign"},
    ])
    _bundle(t1, out / "tables/table_T1_scenario_definitions", "Scenario definitions")
    generated.append("table_T1_scenario_definitions")

    if not safety.empty:
        t2 = safety[safety["scenario_id"].isin(["S0", "S1", "S2"])].copy()
        _bundle(t2, out / "tables/table_T2_S0_S2_safety_controls", "Safety controls S0–S2")
        generated.append("table_T2_S0_S2_safety_controls")

    for sid, name in (("S3", "T3_S3_strong_campaign"), ("S4", "T4_S4_weak_campaign")):
        sub = fleet[fleet["scenario_id"] == sid] if "scenario_id" in fleet.columns else pd.DataFrame()
        if not sub.empty:
            agg_cols = {c: (c, "mean") for c in (
                "campaign_detection_rate", "campaign_precision", "campaign_recall", "campaign_f1", "membership_purity",
            ) if c in sub.columns}
            if "benign_vehicles_incorrectly_included" in sub.columns:
                agg_cols["benign_vehicles_included"] = ("benign_vehicles_incorrectly_included", "mean")
            agg = sub.groupby("campaign_size").agg(**agg_cols).reset_index() if agg_cols else sub.head(0)
            _bundle(agg, out / f"tables/table_{name}", name)
            generated.append(f"table_{name}")

    if not campaign_strong.empty:
        _bundle(campaign_strong, out / "tables/table_T5_campaign_size_strong", "Campaign size strong")
        generated.append("table_T5_campaign_size_strong")
    if not campaign_weak.empty:
        _bundle(campaign_weak, out / "tables/table_T6_campaign_size_weak", "Campaign size weak")
        generated.append("table_T6_campaign_size_weak")
    if not campaign_cost.empty:
        _bundle(campaign_cost, out / "tables/table_T7_campaign_size_cost", "Campaign size cost")
        generated.append("table_T7_campaign_size_cost")
    if not edge_perf.empty:
        _bundle(edge_perf, out / "tables/table_T8_edge_connectivity_performance", "Edge connectivity performance")
        generated.append("table_T8_edge_connectivity_performance")
    if not edge_cost.empty:
        _bundle(edge_cost, out / "tables/table_T9_edge_connectivity_cost", "Edge connectivity cost")
        generated.append("table_T9_edge_connectivity_cost")
    if not stats.empty:
        _bundle(stats, out / "tables/table_T10_primary_statistical_tests", "Primary statistical tests")
        generated.append("table_T10_primary_statistical_tests")
    return generated
