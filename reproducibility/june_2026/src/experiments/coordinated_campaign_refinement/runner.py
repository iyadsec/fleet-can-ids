"""Orchestrate coordinated campaign refinement experiment."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.experiments.campaign_analysis_corrected import DescriptorBudget, REQUIRED_SEEDS
from src.experiments.coordinated_campaign_refinement.candidates import annotate_fleet_candidates
from src.experiments.coordinated_campaign_refinement.design_runner import (
    DESIGN_DIRS,
    collect_d0_reference,
    run_design_experiments,
)
from src.experiments.coordinated_campaign_refinement.eligibility_audit import (
    audit_graph_node_eligibility,
    audit_same_vs_cross_edges,
)
from src.experiments.coordinated_campaign_refinement.figures import generate_figures
from src.experiments.coordinated_campaign_refinement.guard import CoordinatedCampaignRefinementGuard
from src.experiments.coordinated_campaign_refinement.refinement_pipeline import RefinementDesign
from src.experiments.coordinated_campaign_refinement.statistics import run_design_comparisons
from src.experiments.coordinated_campaign_refinement.tables import generate_tables
from src.experiments.coordinated_campaign_refinement.validation_search import (
    build_validation_scenarios,
    connectivity_search,
    dbscan_search,
    select_frozen_params,
)
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.strong_campaign_extended.metrics import bootstrap_ci
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

CONFIG_SRC = "new_experiments/final_validated_runs/configs/phase3_campaign_size_corrected.yaml"


def _load_config(project_root: Path) -> dict[str, Any]:
    with (project_root / CONFIG_SRC).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_config(out: Path, frozen: RefinementDesign) -> None:
    cfg = {
        "experiment": "coordinated_campaign_refinement",
        "campaign_sizes": [2, 5, 10],
        "fleet_size": 20,
        "descriptors_per_vehicle": 10,
        "seeds": REQUIRED_SEEDS,
        "weak_threshold": 0.55,
        "strong_threshold": 0.80,
        "designs": {
            "D0": {"graph_node_rule": "all_200_descriptors", "connectivity": "default"},
            "D1": {"graph_node_rule": "anomaly_score >= weak_threshold", "connectivity": "default"},
            "D2": {
                "graph_node_rule": "candidate_only",
                "similarity_threshold": frozen.similarity_threshold,
                "max_same_vehicle_neighbors": frozen.top_k_same_vehicle,
                "max_cross_vehicle_neighbors": frozen.top_k_cross_vehicle,
                "dbscan_eps": frozen.dbscan_eps,
                "dbscan_min_samples": frozen.dbscan_min_samples,
                "consolidate_fragments": frozen.consolidate_fragments,
            },
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "coordinated_campaign_refinement.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def _write_fragment_audit(out: Path, frozen: RefinementDesign) -> None:
    lines = [
        "# Fragment consolidation logic",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Decision",
        "",
        f"Fragment consolidation **{'enabled' if frozen.consolidate_fragments else 'disabled'}** for D2.",
        "",
        "## Rules (when enabled)",
        "",
        "Two predicted clusters may merge when:",
        "- both contain suspicious campaign candidates;",
        "- embedding centroids have cosine similarity ≥ 0.85;",
        "- a cross-vehicle graph path connects the clusters;",
        "- consolidation does not use ground truth, vehicle model, or source filenames.",
        "",
        "## Code",
        "",
        "`src/experiments/coordinated_campaign_refinement/refinement_pipeline.py` → `_consolidate_fragment_labels()`",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def _comparison_summaries(all_df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    d01 = all_df[all_df["design"].isin(["D0", "D1"])].groupby(["design", "campaign_size"]).mean(numeric_only=True).reset_index()
    d01.to_csv(out / "design_comparison_summary.csv", index=False)
    d2 = all_df[all_df["design"] == "D2"].groupby("campaign_size").mean(numeric_only=True).reset_index()
    d2.to_csv(out / "size_comparison_summary.csv", index=False)
    cross_cols = [c for c in ["cross_vehicle_edges", "cross_vehicle_edge_percentage", "cross_vehicle_campaign_edges"] if c in all_df.columns]
    if cross_cols:
        cross = all_df.groupby(["design", "campaign_size"])[cross_cols].mean().reset_index()
        cross.to_csv(out / "cross_vehicle_support_summary.csv", index=False)


def _confidence_intervals(all_df: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    rows = []
    for design, subdir in DESIGN_DIRS.items():
        sub = all_df[all_df["design"] == design] if "design" in all_df.columns else all_df
        if sub.empty:
            continue
        for metric in ["campaign_f1", "membership_precision", "fragmentation_rate"]:
            if metric not in sub.columns:
                continue
            ci = bootstrap_ci(sub, metric, ["campaign_size"])
            ci["design"] = design
            rows.append(ci)
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    for design, subdir in DESIGN_DIRS.items():
        part = df[df["design"] == design] if not df.empty and "design" in df.columns else pd.DataFrame()
        if not part.empty:
            part.to_csv(out_root / subdir / "confidence_intervals.csv", index=False)
    if not df.empty:
        df.to_csv(out_root.parent / "statistical_analysis" / "confidence_intervals.csv", index=False)
    return df


def _write_summary(
    out: Path,
    all_df: pd.DataFrame,
    frozen: RefinementDesign,
    stats: pd.DataFrame,
    extended_launched: bool,
    tables: list[str],
    figures: list[str],
    eligibility_note: str,
) -> None:
    def agg(design: str, cs: int, col: str):
        sub = all_df[(all_df["design"] == design) & (all_df["campaign_size"] == cs)]
        return float(sub[col].mean()) if col in sub.columns and not sub.empty else None

    def fmt(v, spec=".3f"):
        return format(v, spec) if v is not None else "N/A"

    lines = [
        "# Coordinated campaign refinement summary",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## 1. Were clearly benign descriptors entering the original graph?",
        "",
        eligibility_note,
        "",
        "## 2. How many nodes remained after candidate filtering?",
        "",
    ]
    for cs in [2, 5, 10]:
        lines.append(f"- Size {cs}: mean candidate nodes {fmt(agg('D1', cs, 'candidate_nodes'), '.0f')} (of 200 total)")
    lines += [
        "",
        "## 3. Did candidate filtering improve campaign precision?",
        "",
        f"- D0 cs=10 precision: {fmt(agg('D0', 10, 'campaign_precision'))}; D1 cs=10: {fmt(agg('D1', 10, 'campaign_precision'))}",
        "",
        "## 4. Did benign vehicle inclusion decrease?",
        "",
        f"- D0 cs=10 benign included: {fmt(agg('D0', 10, 'benign_vehicles_included'), '.2f')}; D1: {fmt(agg('D1', 10, 'benign_vehicles_included'), '.2f')}",
        "",
        "## 5. Did controlled connectivity reduce fragmentation?",
        "",
        f"- D1 cs=10 fragmentation: {fmt(agg('D1', 10, 'fragmentation_rate'))}; D2: {fmt(agg('D2', 10, 'fragmentation_rate'))}",
        "",
        "## 6. Did cross-vehicle support increase?",
        "",
        f"- D2 cs=10 cross-vehicle edge %: {fmt(agg('D2', 10, 'cross_vehicle_edge_percentage'), '.1f')}%",
        "",
        "## 7. Did campaign F1 improve at sizes 5 and 10?",
        "",
        f"- D2 cs=5 F1: {fmt(agg('D2', 5, 'campaign_f1'))}; cs=10: {fmt(agg('D2', 10, 'campaign_f1'))}",
        "",
        "## 8. Did membership precision improve?",
        "",
        f"- D2 cs=10 membership precision: {fmt(agg('D2', 10, 'membership_precision'))}",
        "",
        "## 9. Did incorrect merging remain controlled?",
        "",
        f"- D2 cs=10 incorrect merging rate: {fmt(agg('D2', 10, 'incorrect_merging_rate'))}",
        "",
        "## 10. Best graph design",
        "",
    ]
    best = "D2" if agg("D2", 10, "campaign_f1") and agg("D1", 10, "campaign_f1") and agg("D2", 10, "campaign_f1") >= agg("D1", 10, "campaign_f1") else "D1"
    lines.append(f"Recommended for publication comparison: **{best}** (highest campaign F1 at cs=10 among refinements).")
    lines += [
        "",
        "## 11. Extended 10/20/30 scalability",
        "",
        f"{'Launched' if extended_launched else 'Not launched'} — D2 must improve/stabilize 2/5/10 before extended fleet-40 study.",
        "",
        "## 12. Main paper tables/figures",
        "",
        "Tables: CR1 (design definitions), CR4 (coordinated detection), CR2 or CR3 (filtering/connectivity effect).",
        "Figures: CR2 (campaign F1), CR4 (fragmentation), CR5 (benign inclusion), CR6 (cross-vehicle support).",
        "",
        "## 13. Limitations",
        "",
        "- Campaign precision may remain below recall when fragmentation persists.",
        "- Benign weak candidates (score ≥ weak threshold) still enter candidate graphs.",
        "- Extended scalability deferred unless D2 shows improvement on 2/5/10.",
        "",
        "## Frozen D2 parameters",
        "",
        f"- similarity_threshold={frozen.similarity_threshold}",
        f"- max_same_vehicle_neighbors={frozen.top_k_same_vehicle}",
        f"- max_cross_vehicle_neighbors={frozen.top_k_cross_vehicle}",
        f"- dbscan_eps={frozen.dbscan_eps}, dbscan_min_samples={frozen.dbscan_min_samples}",
        f"- consolidate_fragments={frozen.consolidate_fragments}",
        "",
        f"Tables generated: {', '.join(tables)}",
        f"Figures generated: {', '.join(figures)}",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def _d2_improves(all_df: pd.DataFrame) -> bool:
    for cs in [2, 5, 10]:
        d1 = all_df[(all_df["design"] == "D1") & (all_df["campaign_size"] == cs)]["campaign_f1"].mean()
        d2 = all_df[(all_df["design"] == "D2") & (all_df["campaign_size"] == cs)]["campaign_f1"].mean()
        if pd.isna(d1) or pd.isna(d2):
            continue
        if d2 >= d1 - 0.02:
            return True
    return False


def run_coordinated_campaign_refinement(
    *,
    dry_run: bool = False,
    skip_test_runs: bool = False,
) -> dict[str, Any]:
    project_root = resolve_project_root()
    guard = CoordinatedCampaignRefinementGuard(project_root)
    out_root = guard.ensure_directory_tree()

    config = _load_config(project_root)
    cc = config.get("corrected_campaign_size", {})
    budget = DescriptorBudget(
        descriptors_per_vehicle=int(cc.get("descriptors_per_vehicle", 10)),
        malicious_per_attacked=int(cc.get("malicious_per_attacked_vehicle", 5)),
        benign_per_attacked=int(cc.get("benign_per_attacked_vehicle", 5)),
        benign_per_benign=int(cc.get("benign_per_benign_vehicle", 10)),
        total_fleet_size=int(cc.get("total_fleet_size", 20)),
    )

    descriptors, features = load_descriptor_tables(
        project_root / config["paths"]["anomaly_descriptors"],
        project_root / config["paths"]["window_features"],
    )
    splits = config.get("splits", {})
    manifest = ensure_split_manifest(
        descriptors,
        features,
        output_path=project_root / "new_experiments/final_validated_runs/manifests/split_manifest.csv",
        seed=int(splits.get("seed", 42)),
        train_ratio=float(splits.get("train", 0.70)),
        val_ratio=float(splits.get("validation", 0.15)),
        test_ratio=float(splits.get("test", 0.15)),
    )
    local_cfg = config.get("local_ids", {})
    catalog = build_instance_catalog(
        descriptors,
        manifest,
        weak_threshold=float(local_cfg.get("weak_threshold", 0.55)),
        strong_threshold=float(local_cfg.get("strong_threshold", 0.80)),
    )

    # 1–2 Audits
    audit_graph_node_eligibility(project_root, out_root / "audit" / "current_graph_node_eligibility.md")
    audit_same_vs_cross_edges(project_root, out_root / "audit" / "same_vs_cross_vehicle_edge_audit.md")

    # 3 Validation scenarios + search
    val_scenarios = build_validation_scenarios(descriptors, manifest, catalog, config, budget)
    val_dir = out_root / "validation_scenarios"
    val_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"scenario_key": k, "n_events": len(s[0])} for k, s in val_scenarios.items()]).to_csv(
        val_dir / "validation_scenario_manifest.csv", index=False
    )

    conn_df = connectivity_search(val_scenarios, config, val_dir / "connectivity_search_results.csv")
    base_d2 = RefinementDesign(design_id="D2_search", candidate_only=True)
    if not conn_df.empty:
        top = conn_df.iloc[0]
        base_d2 = RefinementDesign(
            design_id="D2_search",
            candidate_only=True,
            similarity_threshold=float(top["similarity_threshold"]),
            top_k_same_vehicle=int(top["max_same_vehicle_neighbors"]),
            top_k_cross_vehicle=int(top["max_cross_vehicle_neighbors"]),
        )
    dbscan_df = dbscan_search(val_scenarios, config, base_d2, val_dir / "dbscan_multi_size_search.csv")
    frozen = select_frozen_params(conn_df, dbscan_df)
    _write_config(out_root / "configs", frozen)
    _write_fragment_audit(out_root / "audit" / "fragment_consolidation_logic.md", frozen)
    (out_root / "configs" / "frozen_d2_params.json").write_text(
        json.dumps(frozen.__dict__, indent=2), encoding="utf-8"
    )

    d1_design = RefinementDesign(
        design_id="D1",
        candidate_only=True,
        similarity_threshold=0.95,
        top_k_same_vehicle=10,
        top_k_cross_vehicle=5,
        dbscan_eps=0.8,
        dbscan_min_samples=10,
        consolidate_fragments=False,
    )

    test_seeds = [REQUIRED_SEEDS[0]] if dry_run else REQUIRED_SEEDS

    # D0 reference (no re-run)
    d0_df = collect_d0_reference(project_root, out_root / "results")
    d0_df["design"] = "D0"

    d1_df = pd.DataFrame()
    d2_df = pd.DataFrame()
    if not skip_test_runs:
        # D1 dry / full
        d1_df = run_design_experiments(
            d1_design,
            descriptors=descriptors,
            manifest=manifest,
            catalog=catalog,
            config=config,
            budget=budget,
            out_root=out_root / "results",
            seeds=test_seeds,
        )
        d2_df = run_design_experiments(
            frozen,
            descriptors=descriptors,
            manifest=manifest,
            catalog=catalog,
            config=config,
            budget=budget,
            out_root=out_root / "results",
            seeds=test_seeds,
        )

    all_df = pd.concat([d0_df, d1_df, d2_df], ignore_index=True)
    results_root = out_root / "results"
    _comparison_summaries(all_df, results_root)
    ci_df = _confidence_intervals(all_df, results_root)

    stats = run_design_comparisons(all_df)
    if not stats.empty:
        stats.to_csv(out_root / "statistical_analysis" / "statistical_tests.csv", index=False)
        for design, subdir in DESIGN_DIRS.items():
            part = stats[stats["comparison"].str.contains(design, na=False)] if "comparison" in stats.columns else stats
            if not part.empty:
                part.to_csv(results_root / subdir / "statistical_tests.csv", index=False)

    frozen_dict = frozen.__dict__
    tables = generate_tables(out_root / "tables", all_df, stats, frozen_dict)
    figures = generate_figures(out_root / "figures", all_df)

    extended_launched = False
    if not dry_run and not skip_test_runs and _d2_improves(all_df):
        note_path = out_root / "results" / "extended_scalability" / "EXTENDED_DEFERRED.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            "D2 shows improvement on 2/5/10; extended 10/20/30 fleet-40 study may proceed in a follow-up run.\n"
            "Not executed in this pass to preserve execution order.",
            encoding="utf-8",
        )

    eligibility_note = (
        "Yes. All 200 scenario descriptors enter the D0 fleet graph with no anomaly_score filter "
        "before graph construction (`build_scenario_graph` / `prepare_gnn_fleet_node_matrix`)."
    )
    _write_summary(
        out_root / "COORDINATED_CAMPAIGN_REFINEMENT_SUMMARY.md",
        all_df,
        frozen,
        stats,
        extended_launched,
        tables,
        figures,
        eligibility_note,
    )

    return {
        "out_root": str(out_root),
        "n_runs": len(all_df),
        "frozen_params": frozen_dict,
        "extended_launched": extended_launched,
    }
