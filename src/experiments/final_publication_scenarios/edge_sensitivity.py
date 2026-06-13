"""Edge-connectivity sensitivity with fixed scenario records."""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.campaign_evaluation import aggregate_run_metrics
from src.experiments.experiment_runner import _write_run_outputs
from src.experiments.final_publication_scenarios.inventory import REQUIRED_SEEDS
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config
from src.experiments.method_fcgnn import run_fcgnn_method
from src.experiments.result_writer import ExperimentRunContext, load_experiment_config
from src.experiments.scenario_registry import get_scenario
from src.experiments.hierarchical_alignment.collect import process_run
from src.experiments.hierarchical_alignment.transform import LocalThresholds

PHASE2 = Path("new_experiments/final_validated_runs/results")
PHASE3 = Path("new_experiments/final_validated_runs/results/campaign_size_corrected")
THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
MAX_NEIGHBORS = [5, 10, 25, 50]


def _reference_run(project_root: Path, attack_strength: str, seed: int) -> Path | None:
    """Fixed 200-node scenario records from corrected campaign-size runs (cs=5)."""
    metrics = project_root / PHASE3 / "run_level_metrics.csv"
    if not metrics.exists():
        return None
    df = pd.read_csv(metrics)
    sub = df[
        (df["method"] == "fcgnn")
        & (df["seed"] == seed)
        & (df["attack_strength"] == attack_strength)
        & (df["campaign_size"] == 5)
        & (df["coordination_strength"] == 1.0)
    ]
    if sub.empty:
        return None
    run_id = str(sub.iloc[0].get("run_id", ""))
    if run_id:
        cand = project_root / PHASE3 / "runs" / run_id
        if cand.exists():
            return cand
    for d in (project_root / PHASE3 / "runs").glob(f"*_{attack_strength}_n5_fcgnn_seed{seed}_*"):
        return d
    return None


def _graph_stat(stats: dict[str, Any], *keys: str, default: Any = 0) -> Any:
    for k in keys:
        if k in stats and pd.notna(stats[k]):
            return stats[k]
    return default


def _graph_audit(stats: dict[str, Any]) -> tuple[bool, str]:
    n = int(_graph_stat(stats, "nodes", "graph_nodes", default=0))
    edges = int(_graph_stat(stats, "unique_undirected_edges", "graph_unique_undirected_edges", default=0))
    iso_pct = float(_graph_stat(stats, "isolated_node_percentage", "graph_isolated_node_percentage", default=0))
    density = float(_graph_stat(stats, "graph_density", "graph_graph_density", default=0))
    if edges == 0:
        return False, "zero_edges"
    if n > 0 and density > 0.95:
        return False, "nearly_complete"
    if n > 0 and iso_pct > 0.8 and edges < n:
        return False, "mostly_isolated"
    return True, ""


def dry_audit_edge_grid(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for attack_strength, scenario_id in (("strong", "S3"), ("weak", "S4")):
        ref = _reference_run(project_root, attack_strength, REQUIRED_SEEDS[0])
        if ref is None:
            continue
        scenario_df = pd.read_csv(ref / "selected_source_records.csv")
        mem_path = ref / "scenario_membership.csv"
        if not mem_path.exists():
            mem_path = ref / "vehicle_membership.csv"
        membership = pd.read_csv(mem_path)
        for tau in THRESHOLDS:
            for k in MAX_NEIGHBORS:
                try:
                    from src.experiments.experiment_pipeline import run_graph_method

                    out = run_graph_method(
                        scenario_df, membership, config, REQUIRED_SEEDS[0], "fcgnn",
                        similarity_threshold=tau, max_neighbors=k,
                    )
                    st = out.graph_stats.iloc[0].to_dict() if not out.graph_stats.empty else {}
                    ok, reason = _graph_audit(st)
                    rows.append({
                        "scenario": scenario_id,
                        "similarity_threshold": tau,
                        "max_neighbors": k,
                        "unique_edges": _graph_stat(st, "unique_undirected_edges", "graph_unique_undirected_edges"),
                        "pyg_stored_edges": _graph_stat(st, "pyg_stored_edges", "graph_pyg_stored_edges"),
                        "graph_density": _graph_stat(st, "graph_density", "graph_graph_density"),
                        "isolated_node_percentage": _graph_stat(st, "isolated_node_percentage", "graph_isolated_node_percentage"),
                        "eligible": ok,
                        "exclusion_reason": reason,
                    })
                except Exception as exc:
                    rows.append({
                        "scenario": scenario_id,
                        "similarity_threshold": tau,
                        "max_neighbors": k,
                        "eligible": False,
                        "exclusion_reason": str(exc),
                    })
    return pd.DataFrame(rows)


def run_edge_sensitivity(
    project_root: Path,
    output_root: Path,
    config: dict[str, Any],
    *,
    dry_only: bool = False,
) -> pd.DataFrame:
    audit = dry_audit_edge_grid(project_root, config)
    audit_path = output_root / "results/edge_sensitivity/excluded_configurations.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    excluded = audit[~audit["eligible"]].copy()
    excluded.to_csv(audit_path, index=False)
    eligible = audit[audit["eligible"]]
    if dry_only:
        return pd.DataFrame()

    rows: list[dict] = []
    thresholds = LocalThresholds()
    for attack_strength, scenario_id in (("strong", "S3"), ("weak", "S4")):
        scenario_key = "S3_strong_campaign" if attack_strength == "strong" else "S4_weak_campaign"
        spec = get_scenario(scenario_key)
        for seed in REQUIRED_SEEDS:
            ref = _reference_run(project_root, attack_strength, seed)
            if ref is None:
                continue
            scenario_df = pd.read_csv(ref / "selected_source_records.csv")
            mem_path = ref / "scenario_membership.csv"
            if not mem_path.exists():
                mem_path = ref / "vehicle_membership.csv"
            membership = pd.read_csv(mem_path)
            for _, cfg_row in eligible[eligible["scenario"] == scenario_id].iterrows():
                tau = float(cfg_row["similarity_threshold"])
                k = int(cfg_row["max_neighbors"])
                run_id = f"edge_{scenario_id}_seed{seed}_tau{tau:.2f}_k{k}".replace(".", "p")
                run_dir = output_root / "results/edge_sensitivity/runs" / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                t0 = time.perf_counter()
                try:
                    ctx = ExperimentRunContext(
                        run_id=run_id,
                        scenario_key=scenario_key,
                        method="fcgnn",
                        seed=seed,
                        campaign_size=5,
                        coordination_strength=1.0,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        output_root=output_root,
                        run_dir=run_dir,
                    )
                    outputs = run_fcgnn_method(
                        ctx, scenario_df, membership, config,
                        seed=seed, similarity_threshold=tau, max_neighbors=k,
                    )
                    metrics = aggregate_run_metrics(
                        method="fcgnn", seed=seed, scenario_key=scenario_key,
                        campaign_size=5, coordination_strength=1.0,
                        event_predictions=outputs.event_predictions,
                        vehicle_predictions=outputs.vehicle_predictions,
                        membership=membership,
                        cluster_df=outputs.cluster_df,
                        expect_campaign=True,
                        runtime={**outputs.runtime, "total_sec": time.perf_counter() - t0},
                    )
                    _write_run_outputs(ctx, spec, scenario_df, membership, outputs, metrics)
                    gs = outputs.graph_stats.iloc[0].to_dict() if not outputs.graph_stats.empty else {}
                    result = process_run(run_dir, thresholds=thresholds)
                    fleet_m = result[2] if result else {}
                    row = {
                        "run_id": run_id,
                        "seed": seed,
                        "scenario": scenario_id,
                        "attack_strength": attack_strength,
                        "campaign_size": 5,
                        "similarity_threshold": tau,
                        "max_neighbors": k,
                        "unique_edges": _graph_stat(gs, "unique_undirected_edges", "graph_unique_undirected_edges"),
                        "pyg_stored_edges": _graph_stat(gs, "pyg_stored_edges", "graph_pyg_stored_edges"),
                        "average_degree": _graph_stat(gs, "average_degree", "graph_average_degree"),
                        "cross_vehicle_edge_percentage": _graph_stat(gs, "cross_vehicle_edge_percentage", "graph_cross_vehicle_edge_percentage"),
                        "graph_build_time": outputs.runtime.get("graph_construction_sec", gs.get("graph_construction_time_sec", np.nan)),
                        "inference_time": outputs.runtime.get("gnn_inference_sec", np.nan),
                        "peak_memory": np.nan,
                        **{k: fleet_m.get(k, metrics.get(k)) for k in (
                            "campaign_precision", "campaign_recall", "campaign_f1",
                            "false_campaign_alert_rate", "campaign_membership_precision",
                            "campaign_membership_recall", "incorrect_campaign_merging",
                            "benign_vehicles_incorrectly_included",
                        )},
                    }
                    rows.append(row)
                except Exception as exc:
                    (output_root / "logs" / f"edge_fail_{run_id}.log").write_text(traceback.format_exc())
    return pd.DataFrame(rows)
