"""Full cross-dataset publication stage (per-set and pooled outputs)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ctt.constants import OUTPUT_ROOT, SETS
from src.ctt.evaluation import aggregate_scenario_results, generate_figures, generate_publication_tables
from src.ctt.progress_logger import ProgressLogger
from src.ctt.run_config import RunConfig
from src.ctt.set_pilot import (
    SET_PILOT_SCENARIOS,
    estimate_set_scope,
    generate_set_figures,
    generate_set_summary,
    generate_set_tables,
    run_stage_set_pilot,
)
from src.ctt.utils import ensure_dir, write_markdown


def full_work_root(base_output: Path, set_id: str) -> Path:
    return base_output / "full" / set_id


def pooled_work_root(base_output: Path) -> Path:
    return base_output / "full" / "pooled"


def check_full_prerequisites(base_output: Path) -> tuple[bool, str]:
    from scripts.validate_can_train_and_test_cross_dataset import validate
    from scripts.validate_can_train_and_test_set_pilot import validate_set_pilot

    for prior in ("audit", "pilot"):
        vr = validate(base_output, prior)
        if vr.critical_failures:
            return False, f"full blocked: {prior} validation failed ({vr.critical_failures})"

    sp_vr = validate_set_pilot(base_output, "set_01")
    if sp_vr.critical_failures:
        return False, f"full blocked: set_pilot set_01 validation failed ({sp_vr.critical_failures})"
    return True, ""


def run_stage_full_set(cfg: RunConfig, progress: ProgressLogger) -> dict:
    """Run one full publication set under full/{set_id}/."""
    return run_stage_set_pilot(cfg, progress)


def generate_full_cross_dataset_summary(base_output: Path, set_ids: list[str]) -> Path:
    """Write comprehensive full-run summary answering publication checklist questions."""
    sections: dict[str, str] = {}
    per_set: list[dict] = []

    for set_id in set_ids:
        root = full_work_root(base_output, set_id)
        if not (root / "manifests" / f"stage_full_{set_id}_complete.json").exists():
            continue
        marker = json.loads((root / "manifests" / f"stage_full_{set_id}_complete.json").read_text())
        nm = pd.read_csv(root / "manifests" / "normalization_manifest.csv")
        wm = pd.read_csv(root / "manifests" / "window_manifest.csv")
        gs = pd.read_csv(root / "graph" / f"{set_id}_graph_statistics.csv").iloc[0].to_dict()
        scen = pd.read_csv(root / "results" / "scenario_evaluation" / f"{set_id}_run_level_metrics.csv")
        comm = pd.read_csv(root / "results" / "descriptor_transfer" / "communication_summary.csv")
        subset_metrics = pd.read_csv(root / "results" / "local_detection" / f"{set_id}_by_subset.csv")

        vehicles = sorted(wm["vehicle_id"].unique())
        attacks = sorted(set(wm["attack_type"].unique()) - {"benign"})
        per_set.append(
            {
                "set_id": set_id,
                "files": len(nm),
                "rows": int(nm["row_count"].sum()) if "row_count" in nm.columns else 0,
                "windows": len(wm),
                "descriptors": marker.get("descriptors", 0),
                "vehicles": vehicles,
                "attacks": attacks,
                "cross_vehicle_edge_pct": gs.get("cross_vehicle_edge_pct"),
                "descriptor_rate": float(comm.iloc[0]["candidate_transmission_rate"]),
            }
        )

        sections[f"{set_id} local detection by subset"] = (
            subset_metrics[subset_metrics["attack_type"] == "all"]
            .groupby("subset_name")[["precision", "recall", "f1"]]
            .mean()
            .round(4)
            .to_string()
        )
        sections[f"{set_id} scenario means"] = (
            scen.groupby("scenario")[
                [
                    "local_or_incident_detected",
                    "fleet_campaign_detected",
                    "false_campaign",
                    "incorrect_merge_rate",
                    "campaign_f1",
                ]
            ]
            .mean()
            .round(4)
            .to_string()
        )

    if not per_set:
        sections["Status"] = "No completed full sets found."
    else:
        df = pd.DataFrame(per_set)
        sections["1. Sets processed"] = ", ".join(df["set_id"].tolist())
        sections["2. Files/rows/windows/descriptors per set"] = df[
            ["set_id", "files", "rows", "windows", "descriptors"]
        ].to_string(index=False)
        sections["3. Vehicles per set"] = "\n".join(f"{r['set_id']}: {', '.join(r['vehicles'])}" for r in per_set)
        sections["4. Attacks per set"] = "\n".join(f"{r['set_id']}: {', '.join(r['attacks'])}" for r in per_set)
        sections["8. Cross-vehicle edge % per set"] = "\n".join(
            f"{r['set_id']}: {r['cross_vehicle_edge_pct']}" for r in per_set
        )
        sections["10. Descriptor candidate rate per set"] = "\n".join(
            f"{r['set_id']}: {r['descriptor_rate']:.4f}" for r in per_set
        )

    pooled = pooled_work_root(base_output)
    tables = sorted((pooled / "tables").glob("table_CTT*.csv")) if (pooled / "tables").exists() else []
    figures = sorted((pooled / "figures").glob("figure_CTT*.png")) if (pooled / "figures").exists() else []
    sections["18. Main-paper tables"] = ", ".join(t.name for t in tables)
    sections["18. Main-paper figures"] = ", ".join(f.name for f in figures)
    sections["19. Limitations"] = (
        "Simulated cross-vehicle campaigns on independent CTT splits; "
        "sensitivity analyses use graph-statistics proxies; "
        "caps applied for reproducible publication runs."
    )

    summary_path = base_output / "full" / "CAN_TRAIN_AND_TEST_FULL_CROSS_DATASET_SUMMARY.md"
    write_markdown(summary_path, "CAN Train-and-Test Full Cross-Dataset Summary", sections)
    return summary_path


def generate_pooled_outputs(base_output: Path, set_ids: list[str]) -> Path:
    """Aggregate per-set full outputs into full/pooled publication artifacts."""
    pooled = ensure_dir(pooled_work_root(base_output))
    for subdir in (
        "audit", "manifests", "tables", "figures", "statistics", "logs", "validation", "results",
    ):
        ensure_dir(pooled / subdir)

    inv_path = base_output / "manifests" / "ctt_file_inventory.csv"
    file_inventory = pd.read_csv(inv_path) if inv_path.exists() else pd.DataFrame()

    window_parts, metrics_parts, scenario_parts = [], [], []
    desc_parts, campaign_parts, edge_parts = [], [], []
    graph_rows = []

    for set_id in set_ids:
        root = full_work_root(base_output, set_id)
        if not (root / "manifests" / f"stage_full_{set_id}_complete.json").exists():
            continue
        wm = root / "manifests" / "window_manifest.csv"
        if wm.exists():
            window_parts.append(pd.read_csv(wm).assign(set_id=set_id))
        pred_metrics = root / "results" / "local_detection" / f"{set_id}_by_subset.csv"
        if pred_metrics.exists():
            metrics_parts.append(pd.read_csv(pred_metrics).assign(set_id=set_id))
        scen = root / "results" / "scenario_evaluation" / f"{set_id}_run_level_metrics.csv"
        if scen.exists():
            scenario_parts.append(pd.read_csv(scen).assign(set_id=set_id))
        ds = root / "results" / "descriptor_transfer" / "communication_summary.csv"
        if ds.exists():
            desc_parts.append(pd.read_csv(ds).assign(set_id=set_id))
        cs = root / "results" / "campaign_size" / f"{set_id}_run_level.csv"
        if cs.exists():
            campaign_parts.append(pd.read_csv(cs))
        es = root / "results" / "edge_sensitivity" / f"{set_id}_run_level.csv"
        if es.exists():
            edge_parts.append(pd.read_csv(es))
        gs = root / "graph" / f"{set_id}_graph_statistics.csv"
        if gs.exists():
            graph_rows.append(pd.read_csv(gs).assign(set_id=set_id).iloc[0].to_dict())

    window_manifest = pd.concat(window_parts, ignore_index=True) if window_parts else pd.DataFrame()
    metrics_df = pd.concat(metrics_parts, ignore_index=True) if metrics_parts else pd.DataFrame()
    scenario_results = pd.concat(scenario_parts, ignore_index=True) if scenario_parts else pd.DataFrame()
    desc_summary = pd.concat(desc_parts, ignore_index=True) if desc_parts else pd.DataFrame()
    campaign_size = pd.concat(campaign_parts, ignore_index=True) if campaign_parts else pd.DataFrame()
    edge_sensitivity = pd.concat(edge_parts, ignore_index=True) if edge_parts else pd.DataFrame()
    graph_stats = graph_rows[0] if graph_rows else {}

    if not window_manifest.empty:
        window_manifest.to_csv(pooled / "manifests" / "pooled_window_manifest.csv", index=False)
    if not scenario_results.empty:
        scenario_results.to_csv(pooled / "results" / "pooled_scenario_run_level.csv", index=False)
        aggregate_scenario_results(scenario_results).to_csv(
            pooled / "results" / "pooled_scenario_summary.csv", index=False
        )

    generate_publication_tables(
        file_inventory,
        window_manifest,
        metrics_df,
        desc_summary,
        graph_stats,
        scenario_results,
        campaign_size,
        edge_sensitivity,
        pooled,
    )
    generate_figures(
        window_manifest,
        pd.DataFrame(),
        metrics_df,
        graph_stats,
        scenario_results,
        campaign_size,
        edge_sensitivity,
        desc_summary,
        pooled,
    )

    summary_path = generate_full_cross_dataset_summary(base_output, completed)
    (ensure_dir(base_output / "full" / "manifests") / "stage_full_complete.json").write_text(
        json.dumps({"stage": "full", "status": "complete", "sets": set_ids}, indent=2),
        encoding="utf-8",
    )
    return summary_path


def run_stage_full(cfg: RunConfig, progress: ProgressLogger) -> dict:
    """Run full publication stage for one or all sets, then pooled aggregation."""
    ok, msg = check_full_prerequisites(cfg.output_root)
    if not ok:
        raise RuntimeError(msg)

    set_ids = [cfg.set_id] if cfg.set_id else list(SETS)
    results = {}
    for set_id in set_ids:
        progress.info(f"Full stage processing {set_id}")
        set_cfg = RunConfig(
            stage="full",
            dataset_root=cfg.dataset_root,
            output_root=cfg.output_root,
            set_id=set_id,
            max_files=cfg.max_files,
            max_rows_per_file=cfg.max_rows_per_file,
            max_windows=cfg.max_windows,
            max_graph_nodes=cfg.max_graph_nodes,
            max_descriptors=cfg.max_descriptors,
            resume=cfg.resume,
            skip_existing=cfg.skip_existing,
            confirm_large_run=cfg.confirm_large_run,
        )
        results[set_id] = run_stage_full_set(set_cfg, progress)

    completed = [s for s in set_ids if (full_work_root(cfg.output_root, s) / "manifests" / f"stage_full_{s}_complete.json").exists()]
    summary_path = None
    if not cfg.set_id:
        summary_path = generate_pooled_outputs(cfg.output_root, completed)
    return {"sets": completed, "results": results, "summary_path": summary_path}
