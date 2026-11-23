"""Comprehensive strong-campaign diagnostic metrics from run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.campaign_evaluation import _safe_div
from src.experiments.evaluation_correction.metrics import reconstruct_cluster_df
from src.experiments.experiment_pipeline import resolve_vehicle_id_column
from src.experiments.model_diversity_final_tuned.false_campaign_metrics import compute_false_campaign_breakdown

REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]


def _graph_val(row: dict[str, Any], *keys: str, default: Any = np.nan) -> Any:
    for k in keys:
        if k in row and pd.notna(row[k]):
            return row[k]
    return default


def compute_diagnostic_row(
    *,
    run_id: str,
    seed: int,
    campaign_size: int,
    fleet_size: int,
    total_nodes: int,
    experiment_group: str,
    method: str,
    event_predictions: pd.DataFrame,
    vehicle_predictions: pd.DataFrame,
    membership: pd.DataFrame,
    cluster_df: pd.DataFrame,
    graph_stats: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    base_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fleet-layer diagnostics; local event metrics are not overwritten."""
    runtime = runtime or {}
    graph_stats = graph_stats or {}
    base = base_metrics or {}

    breakdown = compute_false_campaign_breakdown(
        event_predictions, membership, cluster_df, expect_campaign=True
    )
    veh_col = resolve_vehicle_id_column(membership)
    gt_attacked = membership.groupby(veh_col)["ground_truth_campaign_member"].max()
    if "fleet_campaign_member" in event_predictions.columns:
        pred_mem = event_predictions.groupby(veh_col)["fleet_campaign_member"].max()
    else:
        from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED

        pred_mem = event_predictions.groupby(veh_col)["final_decision"].apply(
            lambda s: int((s == DECISION_COORDINATED).any())
        )

    attacked_ids = gt_attacked[gt_attacked == 1].index
    attacked_included = int((pred_mem.reindex(attacked_ids).fillna(0) == 1).sum())
    attacked_missed = int(len(attacked_ids) - attacked_included)

    pred_campaign_veh = pred_mem[pred_mem == 1].index
    predicted_campaign_size = int(len(pred_campaign_veh))
    true_campaign_size = int(campaign_size)

    n_gt = int(breakdown.get("n_ground_truth_campaigns", 1))
    n_qual = int(breakdown.get("n_detected_campaign_clusters", 0))
    fragmentation = int(breakdown.get("fragmentation", 0))
    frag_rate = _safe_div(fragmentation, max(n_gt, 1))
    fragments_per_campaign = _safe_div(n_qual, max(n_gt, 1)) if n_gt else float(n_qual)

    cluster_ids = event_predictions.get("cluster_id", pd.Series(-1, index=event_predictions.index))
    noise_pct = float((cluster_ids == -1).mean() * 100) if len(cluster_ids) else 0.0
    largest_cluster = int(cluster_ids[cluster_ids >= 0].value_counts().max()) if (cluster_ids >= 0).any() else 0

    mem_prec = float(breakdown.get("campaign_membership_precision", np.nan))
    mem_rec = float(breakdown.get("campaign_membership_recall", np.nan))
    mem_f1 = _safe_div(2 * mem_prec * mem_rec, mem_prec + mem_rec)

    graph_build = float(
        runtime.get("graph_construction_sec")
        or graph_stats.get("graph_construction_time_sec")
        or base.get("runtime_graph_construction_sec", np.nan)
    )
    inference = float(runtime.get("gnn_inference_sec") or base.get("runtime_gnn_inference_sec", np.nan))
    clustering = float(runtime.get("clustering_sec") or base.get("runtime_clustering_sec", np.nan))
    e2e = float(runtime.get("total_sec") or base.get("runtime_total_sec", np.nan))
    peak_mem = runtime.get("peak_memory_mb", base.get("peak_memory_mb", np.nan))

    nodes = int(_graph_val(graph_stats, "nodes", "graph_nodes", default=total_nodes))
    unique_edges = float(
        _graph_val(
            graph_stats,
            "unique_undirected_edges",
            "graph_unique_undirected_edges",
            default=base.get("graph_unique_undirected_edges", np.nan),
        )
    )

    row: dict[str, Any] = {
        "run_id": run_id,
        "seed": seed,
        "campaign_size": campaign_size,
        "fleet_size": fleet_size,
        "total_nodes": total_nodes,
        "experiment_group": experiment_group,
        "method": method,
        "framework_config": "C3" if method == "fcgnn" else ("C2" if method == "descriptor_clustering" else method),
        "campaign_detection_rate": float(
            base.get("campaign_detection_rate", breakdown.get("campaign_recall", np.nan))
        ),
        "campaign_precision": float(base.get("campaign_precision", breakdown["campaign_precision"])),
        "campaign_recall": float(base.get("campaign_recall", breakdown["campaign_recall"])),
        "campaign_f1": float(base.get("campaign_f1", breakdown["campaign_f1"])),
        "n_true_campaign_clusters": n_gt,
        "n_predicted_campaign_clusters": n_qual,
        "false_campaign_cluster_count": int(breakdown.get("false_campaign_cluster_count", 0)),
        "missed_campaign_count": int(n_gt > 0 and n_qual == 0),
        "membership_precision": mem_prec,
        "membership_recall": mem_rec,
        "membership_f1": mem_f1,
        "membership_purity": mem_prec,
        "completeness": float(breakdown.get("campaign_recall", mem_rec)),
        "attacked_vehicles_correctly_included": attacked_included,
        "attacked_vehicles_missed": attacked_missed,
        "benign_vehicles_included": int(breakdown.get("benign_vehicles_included", 0)),
        "predicted_campaign_size": predicted_campaign_size,
        "true_campaign_size": true_campaign_size,
        "fragmentation_rate": frag_rate,
        "fragments_per_true_campaign": fragments_per_campaign,
        "incorrect_merging_rate": float(breakdown.get("incorrect_merging", 0)),
        "unrelated_clusters_merged": int(breakdown.get("incorrect_merging", 0)),
        "dbscan_noise_percentage": noise_pct,
        "largest_cluster_size": largest_cluster,
        "connected_component_count": int(
            _graph_val(graph_stats, "connected_components", "graph_connected_components", default=np.nan)
        ),
        "unique_edges": unique_edges,
        "pyg_stored_edges": float(
            _graph_val(graph_stats, "pyg_stored_edges", "graph_pyg_stored_edges", default=np.nan)
        ),
        "average_degree": float(
            _graph_val(graph_stats, "average_degree", "graph_average_degree", default=np.nan)
        ),
        "median_degree": float(
            _graph_val(graph_stats, "median_degree", "graph_median_degree", default=np.nan)
        ),
        "graph_density": float(
            _graph_val(graph_stats, "graph_density", "graph_graph_density", default=np.nan)
        ),
        "isolated_node_percentage": float(
            _graph_val(
                graph_stats,
                "isolated_node_percentage",
                "graph_isolated_node_percentage",
                default=np.nan,
            )
        ),
        "cross_vehicle_edges": float(
            _graph_val(graph_stats, "cross_vehicle_edges", "graph_cross_vehicle_edges", default=np.nan)
        ),
        "cross_vehicle_edge_percentage": float(
            _graph_val(
                graph_stats,
                "cross_vehicle_edge_percentage",
                "graph_cross_vehicle_edge_percentage",
                default=np.nan,
            )
        ),
        "largest_connected_component": int(
            _graph_val(
                graph_stats,
                "largest_component_size",
                "graph_largest_component_size",
                default=np.nan,
            )
        ),
        "graph_build_time": graph_build,
        "gnn_inference_time": inference,
        "clustering_time": clustering,
        "end_to_end_latency": e2e,
        "peak_memory": peak_mem,
    }
    return row


def load_run_diagnostics(run_dir: Path, *, base_metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load per-run CSVs and compute diagnostics."""
    run_dir = Path(run_dir)
    events = pd.read_csv(run_dir / "event_predictions.csv")
    vehicles = pd.read_csv(run_dir / "vehicle_predictions.csv")
    mem_path = run_dir / "vehicle_membership.csv"
    if not mem_path.exists():
        mem_path = run_dir / "scenario_membership.csv"
    membership = pd.read_csv(mem_path)
    cluster_df = reconstruct_cluster_df(events)

    graph_stats: dict[str, Any] = {}
    gs_path = run_dir / "graph_statistics.csv"
    if gs_path.exists():
        graph_stats = pd.read_csv(gs_path).iloc[0].to_dict()

    runtime: dict[str, Any] = {}
    rt_path = run_dir / "runtime_memory.json"
    if rt_path.exists():
        runtime = json.loads(rt_path.read_text(encoding="utf-8"))
        runtime = {
            "graph_construction_sec": runtime.get("runtime_graph_construction_sec"),
            "gnn_inference_sec": runtime.get("runtime_gnn_inference_sec"),
            "clustering_sec": runtime.get("runtime_clustering_sec"),
            "total_sec": runtime.get("runtime_total_sec"),
        }

    if base_metrics is None and (run_dir / "run_level_metrics.csv").exists():
        base_metrics = pd.read_csv(run_dir / "run_level_metrics.csv").iloc[0].to_dict()

    base = base_metrics or {}
    run_id = str(base.get("run_id", run_dir.name))
    return compute_diagnostic_row(
        run_id=run_id,
        seed=int(base.get("seed", 0)),
        campaign_size=int(base.get("campaign_size", 0)),
        fleet_size=int(base.get("total_fleet_size", 20)),
        total_nodes=int(base.get("graph_nodes", base.get("expected_total_nodes", 200))),
        experiment_group=str(base.get("experiment_group", "existing_fleet20")),
        method=str(base.get("method", "fcgnn")),
        event_predictions=events,
        vehicle_predictions=vehicles,
        membership=membership,
        cluster_df=cluster_df,
        graph_stats=graph_stats,
        runtime=runtime,
        base_metrics=base,
    )


def summary_mean_std(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    skip = set(group_cols + ["seed"])
    agg_cols = [c for c in numeric if c not in skip]
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        for col in agg_cols:
            rec[f"{col}_mean"] = float(g[col].mean())
            rec[f"{col}_std"] = float(g[col].std(ddof=1)) if len(g) > 1 else 0.0
        rec["n_seeds"] = len(g)
        rows.append(rec)
    return pd.DataFrame(rows)


def bootstrap_ci(df: pd.DataFrame, metric: str, group_cols: list[str], n_boot: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for keys, g in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        vals = g[metric].dropna().to_numpy()
        if len(vals) < 2:
            continue
        boots = []
        for _ in range(n_boot):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boots.append(float(np.mean(sample)))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rec = dict(zip(group_cols, keys))
        rec.update({"metric": metric, "mean": float(vals.mean()), "ci_low": float(lo), "ci_high": float(hi)})
        rows.append(rec)
    return pd.DataFrame(rows)
