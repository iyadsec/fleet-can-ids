"""Aggregate Phase 4 model diversity run outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.hierarchical_alignment.metrics import (
    compute_fleet_campaign_run_metrics,
    compute_local_event_metrics,
    compute_membership_errors,
    compute_weak_campaign_support,
)
from src.experiments.hierarchical_alignment.transform import LocalThresholds, align_event_predictions, validate_local_not_overwritten


def _load_membership(run_dir: Path) -> pd.DataFrame:
    for name in ("scenario_membership.csv", "vehicle_membership.csv"):
        p = run_dir / name
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"No membership in {run_dir}")


def collect_run_metrics(output_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for strength in ("strong", "weak"):
        runs_dir = output_root / "results" / strength / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir() and (run_dir / "run_level_metrics.csv").exists():
                row = pd.read_csv(run_dir / "run_level_metrics.csv")
                row["run_id"] = run_dir.name
                rows.append(row)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_confidence_intervals(fleet_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = ["campaign_f1", "campaign_detection_rate", "membership_purity", "false_campaign_alert_rate"]
    for (strength, dl, cfg), g in fleet_df.groupby(["attack_strength", "diversity_level", "framework_config"]):
        if cfg not in ("C2", "C3"):
            continue
        for metric in metrics:
            if metric not in g.columns:
                continue
            x = pd.to_numeric(g[metric], errors="coerce").dropna()
            if x.empty:
                continue
            rows.append(
                {
                    "attack_strength": strength,
                    "diversity_level": int(dl),
                    "framework_config": cfg,
                    "metric": metric,
                    "mean": float(x.mean()),
                    "std": float(x.std()),
                    "n_seeds": int(len(x)),
                    "ci95_low": float(x.mean() - 1.96 * x.std() / max(len(x) ** 0.5, 1)),
                    "ci95_high": float(x.mean() + 1.96 * x.std() / max(len(x) ** 0.5, 1)),
                }
            )
    return pd.DataFrame(rows)


def collect_from_runs(output_root: Path) -> dict[str, pd.DataFrame]:
    thresholds = LocalThresholds()
    local_rows, fleet_rows, weak_rows, err_rows = [], [], [], []
    sim_rows, graph_rows, map_rows, comp_rows, runtime_rows, membership_rows = [], [], [], [], [], []
    run_df = collect_run_metrics(output_root)

    for strength in ("strong", "weak"):
        runs_dir = output_root / "results" / strength / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir() or not (run_dir / "event_predictions.csv").exists():
                continue
            raw = pd.read_csv(run_dir / "event_predictions.csv")
            m = pd.read_csv(run_dir / "run_level_metrics.csv").iloc[0]
            method = str(m["method"])
            if method not in ("local_ids", "descriptor_clustering", "fcgnn"):
                continue
            membership = _load_membership(run_dir)
            aligned = align_event_predictions(raw, method=method, thresholds=thresholds)
            validate_local_not_overwritten(raw, aligned)
            eval_df = raw.copy()
            eval_df["local_event_alert"] = aligned["local_event_alert"]
            eval_df["fleet_campaign_member"] = aligned["fleet_campaign_member"]
            eval_df["fleet_decision"] = aligned["fleet_decision"]
            eval_df["local_evidence_level"] = aligned["local_evidence_level"]

            seed = int(m["seed"])
            dl = int(m.get("diversity_level", 0))
            local_m = compute_local_event_metrics(aligned, run_id=run_dir.name, scenario_id=f"D{dl}", seed=seed)
            local_m["attack_strength"] = strength
            local_m["diversity_level"] = dl
            local_m["latency_sec"] = float(m.get("runtime_total_sec", float("nan")))
            fleet_m = compute_fleet_campaign_run_metrics(
                eval_df,
                membership,
                run_id=run_dir.name,
                method=method,
                scenario_id=f"D{dl}",
                seed=seed,
                campaign_size=5,
                coordination_strength=1.0,
                expect_campaign=True,
            )
            fleet_m["attack_strength"] = strength
            fleet_m["diversity_level"] = dl
            fleet_m["composition_label"] = str(m.get("composition_label", ""))
            weak_m = compute_weak_campaign_support(
                eval_df, membership, run_id=run_dir.name, method=method, seed=seed, campaign_size=5,
            )
            weak_m["attack_strength"] = strength
            weak_m["diversity_level"] = dl
            err_m = compute_membership_errors(eval_df, run_id=run_dir.name, method=method)
            local_rows.append(local_m)
            fleet_rows.append(fleet_m)
            weak_rows.append(weak_m)
            err_rows.append(err_m)
            if (run_dir / "descriptor_similarity.csv").exists():
                s = pd.read_csv(run_dir / "descriptor_similarity.csv")
                s["run_id"] = run_dir.name
                s["diversity_level"] = dl
                s["attack_strength"] = strength
                sim_rows.append(s)
            if (run_dir / "graph_statistics.csv").exists():
                g = pd.read_csv(run_dir / "graph_statistics.csv")
                g["run_id"] = run_dir.name
                graph_rows.append(g)
            if (run_dir / "scenario_membership.csv").exists():
                membership_rows.append(
                    pd.read_csv(run_dir / "scenario_membership.csv").assign(run_id=run_dir.name)
                )
            if (run_dir / "scenario_vehicle_mapping.csv").exists():
                map_rows.append(pd.read_csv(run_dir / "scenario_vehicle_mapping.csv").assign(run_id=run_dir.name))
            if (run_dir / "vehicle_composition.csv").exists():
                comp_rows.append(pd.read_csv(run_dir / "vehicle_composition.csv"))
            row = {
                    "run_id": run_dir.name,
                    "method": method,
                    "framework_config": m.get("framework_config", ""),
                    "diversity_level": dl,
                    "attack_strength": strength,
                    "runtime_graph_construction_sec": m.get("runtime_graph_construction_sec"),
                    "runtime_clustering_sec": m.get("runtime_clustering_sec"),
                    "runtime_gnn_inference_sec": m.get("runtime_gnn_inference_sec"),
                    "runtime_total_sec": m.get("runtime_total_sec"),
                    "graph_unique_undirected_edges": m.get("graph_unique_undirected_edges"),
                    "graph_nodes": m.get("graph_nodes"),
                    "cross_model_edges": m.get("cross_model_edges"),
                    "cross_model_edge_percentage": m.get("cross_model_edge_percentage"),
                }
            runtime_rows.append(row)
            if (run_dir / "edge_list.csv").exists():
                el = pd.read_csv(run_dir / "edge_list.csv")
                if not el.empty and "source_vehicle" in el.columns and "target_vehicle" in el.columns:
                    cross = int((el["source_vehicle"] != el["target_vehicle"]).sum())
                    total = len(el)
                    runtime_rows[-1]["cross_model_edges"] = cross
                    runtime_rows[-1]["cross_model_edge_percentage"] = 100.0 * cross / total if total else 0.0
                    if sim_rows:
                        sim_rows[-1]["cross_model_edges"] = cross
                        sim_rows[-1]["same_model_edges"] = total - cross
                        sim_rows[-1]["cross_model_edge_percentage"] = 100.0 * cross / total if total else 0.0

    return {
        "run_level_metrics": run_df,
        "local_event_metrics": pd.DataFrame(local_rows),
        "fleet_campaign_metrics": pd.DataFrame(fleet_rows),
        "weak_campaign_support": pd.DataFrame(weak_rows),
        "campaign_membership_errors": pd.DataFrame(err_rows),
        "descriptor_similarity": pd.concat(sim_rows, ignore_index=True) if sim_rows else pd.DataFrame(),
        "graph_statistics": pd.concat(graph_rows, ignore_index=True) if graph_rows else pd.DataFrame(),
        "scenario_vehicle_mapping": pd.concat(map_rows, ignore_index=True) if map_rows else pd.DataFrame(),
        "vehicle_composition": pd.concat(comp_rows, ignore_index=True) if comp_rows else pd.DataFrame(),
        "runtime_memory": pd.DataFrame(runtime_rows),
        "campaign_membership": pd.concat(membership_rows, ignore_index=True) if membership_rows else pd.DataFrame(),
    }
