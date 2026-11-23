"""Run D1/D2 test experiments and collect D0 reference."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.experiments.campaign_analysis_corrected import DescriptorBudget, REQUIRED_SEEDS, generate_corrected_campaign_scenario
from src.experiments.campaign_evaluation import aggregate_run_metrics
from src.experiments.coordinated_campaign_refinement.candidates import annotate_fleet_candidates, candidate_statistics
from src.experiments.coordinated_campaign_refinement.refinement_pipeline import RefinementDesign, run_refinement_fcgnn
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config
from src.experiments.strong_campaign_extended.metrics import compute_diagnostic_row, load_run_diagnostics

SIZES = [2, 5, 10]
D0_RUN_ROOT = Path("new_experiments/final_validated_runs/results/campaign_size_corrected/runs")

DESIGN_DIRS = {
    "D0": "current_all_node_graph",
    "D1": "candidate_only_graph",
    "D2": "candidate_only_controlled_connectivity",
}


def collect_d0_reference(project_root: Path, out_root: Path) -> pd.DataFrame:
    rows = []
    base_df = pd.read_csv(
        project_root / "new_experiments/final_validated_runs/results/campaign_size_corrected/run_level_metrics.csv"
    )
    sub = base_df[
        (base_df["method"] == "fcgnn")
        & (base_df["attack_strength"] == "strong")
        & (base_df["campaign_size"].isin(SIZES))
    ]
    for _, base in sub.iterrows():
        rd = project_root / D0_RUN_ROOT / str(base["run_id"])
        if not rd.exists():
            continue
        diag = load_run_diagnostics(rd, base_metrics={**base.to_dict(), "experiment_group": "D0"})
        diag["design"] = "D0"
        cand = candidate_statistics(annotate_fleet_candidates(pd.read_csv(rd / "selected_source_records.csv")))
        for k, v in cand.items():
            if k not in diag:
                diag[k] = v
        rows.append(diag)
    df = pd.DataFrame(rows)
    _export_design(out_root / DESIGN_DIRS["D0"], df)
    return df


def run_design_experiments(
    design: RefinementDesign,
    *,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict,
    budget: DescriptorBudget,
    out_root: Path,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    ensure_fleet_scaler_in_config(config, descriptors, manifest)
    seeds = seeds or REQUIRED_SEEDS
    rows: list[dict] = []
    run_dir = out_root / DESIGN_DIRS[design.design_id] / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log = out_root.parent / "logs" / f"{design.design_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"

    for cs in SIZES:
        for seed in seeds:
            try:
                scenario_df, _, membership, _, _ = generate_corrected_campaign_scenario(
                    attack_strength="strong",
                    seed=seed,
                    descriptors=descriptors,
                    manifest=manifest,
                    catalog=catalog,
                    config=config,
                    campaign_size=cs,
                    coordination_strength=1.0,
                    budget=budget,
                )
                outputs, cand_stats, annotated, consol = run_refinement_fcgnn(
                    scenario_df, membership, config, seed, design
                )
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                rid = f"refinement_{design.design_id}_strong_n{cs}_fcgnn_seed{seed}_{ts}"
                rd = run_dir / rid
                rd.mkdir(parents=True, exist_ok=True)
                scenario_df.to_csv(rd / "selected_source_records.csv", index=False)
                annotated.to_csv(rd / "annotated_source_records.csv", index=False)
                membership.to_csv(rd / "vehicle_membership.csv", index=False)
                outputs.event_predictions.to_csv(rd / "event_predictions.csv", index=False)
                outputs.graph_stats.to_csv(rd / "graph_statistics.csv", index=False)
                outputs.edge_list.to_csv(rd / "edge_list.csv", index=False)
                if not consol.empty:
                    consol.to_csv(rd / "fragment_consolidation_log.csv", index=False)

                metrics = aggregate_run_metrics(
                    method="fcgnn",
                    seed=seed,
                    scenario_key=f"refinement_{design.design_id}_strong",
                    campaign_size=cs,
                    coordination_strength=1.0,
                    event_predictions=outputs.event_predictions,
                    vehicle_predictions=outputs.vehicle_predictions,
                    membership=membership,
                    cluster_df=outputs.cluster_df,
                    expect_campaign=True,
                    runtime=outputs.runtime,
                )
                metrics["run_id"] = rid
                metrics["design"] = design.design_id
                pd.DataFrame([metrics]).to_csv(rd / "run_level_metrics.csv", index=False)

                diag = compute_diagnostic_row(
                    run_id=rid,
                    seed=seed,
                    campaign_size=cs,
                    fleet_size=20,
                    total_nodes=200,
                    experiment_group=design.design_id,
                    method="fcgnn",
                    event_predictions=outputs.event_predictions,
                    vehicle_predictions=outputs.vehicle_predictions,
                    membership=membership,
                    cluster_df=outputs.cluster_df,
                    graph_stats=outputs.graph_stats.iloc[0].to_dict(),
                    runtime=outputs.runtime,
                    base_metrics=metrics,
                )
                diag["design"] = design.design_id
                diag.update(cand_stats)
                if not outputs.edge_list.empty and "source_vehicle" in outputs.edge_list.columns:
                    cross_mask = outputs.edge_list["source_vehicle"] != outputs.edge_list["target_vehicle"]
                    cross = int(cross_mask.sum())
                    diag["cross_vehicle_campaign_edges"] = cross
                    diag["same_vehicle_edges"] = len(outputs.edge_list) - cross
                rows.append(diag)
                with log.open("a", encoding="utf-8") as fh:
                    fh.write(f"OK {design.design_id} cs={cs} seed={seed}\n")
            except Exception as exc:
                with log.open("a", encoding="utf-8") as fh:
                    fh.write(f"FAIL cs={cs} seed={seed}: {exc}\n{traceback.format_exc()}\n")

    df = pd.DataFrame(rows)
    _export_design(out_root / DESIGN_DIRS[design.design_id], df)
    return df


def _export_design(out_dir: Path, diag: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if diag.empty:
        return
    diag.to_csv(out_dir / "run_level_metrics.csv", index=False)
    diag.to_csv(out_dir / "campaign_metrics.csv", index=False)
    diag.to_csv(out_dir / "membership_metrics.csv", index=False)
    diag.to_csv(out_dir / "fragmentation_metrics.csv", index=False)
    diag.to_csv(out_dir / "edge_statistics.csv", index=False)
    diag.to_csv(out_dir / "candidate_statistics.csv", index=False)
    diag.to_csv(out_dir / "runtime_memory.csv", index=False)
    diag.groupby(["design", "campaign_size"]).mean(numeric_only=True).reset_index().to_csv(
        out_dir / "summary_mean_std.csv", index=False
    )
