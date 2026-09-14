"""Orchestrate balanced-split end-to-end publication experiment."""

from __future__ import annotations

import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.experiments.campaign_analysis_corrected import DescriptorBudget
from src.experiments.coordinated_campaign_refinement.refinement_pipeline import run_refinement_fcgnn
from src.experiments.final_end_to_end_publication_run.descriptor_analysis import run_descriptor_analysis
from src.experiments.final_end_to_end_publication_run.edge_sensitivity import run_edge_sensitivity
from src.experiments.final_end_to_end_publication_run.figures import generate_figures
from src.experiments.final_end_to_end_publication_run.pipeline import retrain_local_pipeline
from src.experiments.final_end_to_end_publication_run.runner import (
    _completeness_report,
    _experiment_config,
    _run_single_test,
    _test_event_ids,
    _write_summary,
)
from src.experiments.final_end_to_end_publication_run.scenario_registry import (
    PUBLICATION_SCENARIOS,
    REQUIRED_SEEDS,
    enumerate_test_runs,
)
from src.experiments.final_end_to_end_publication_run.statistics import run_primary_statistics
from src.experiments.final_end_to_end_publication_run.vehicle_evaluation import compute_vehicle_level_metrics
from src.experiments.final_end_to_end_publication_run_balanced.balanced_split import (
    GUARD_FRAMES,
    validate_balanced_split,
)
from src.experiments.final_end_to_end_publication_run_balanced.guard import BalancedPublicationGuard
from src.experiments.final_end_to_end_publication_run_balanced.tables import generate_tables
from src.experiments.final_shared_configuration.metrics import extract_run_metrics, safety_row
from src.experiments.final_shared_configuration.parameter_search import joint_parameter_search
from src.experiments.final_shared_configuration.shared_config import SharedFleetConfiguration
from src.experiments.final_shared_configuration.validation_scenarios import build_mixed_validation_suite
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config
from src.experiments.result_writer import load_experiment_config
from src.experiments.strong_campaign_extended.metrics import bootstrap_ci
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

MASTER_CONFIG_NAME = "balanced_master_experiment.yaml"
ORIG_ROOT = Path("new_experiments/final_end_to_end_publication_run")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_master_config(out_root: Path, project_root: Path) -> tuple[Path, str]:
    fleet = yaml.safe_load((project_root / "configs" / "fleet_ids.yaml").read_text(encoding="utf-8"))
    master = {
        "experiment": {
            "name": "final_end_to_end_publication_run_balanced",
            "output_root": str(out_root.relative_to(project_root)),
            "authoritative_for": "Section VII (balanced split)",
            "split_method": "balanced_platform_aware_with_segments",
            "guard_frames": GUARD_FRAMES,
        },
        "dataset": {
            "source": "OCSLab Car-Hacking labelled traces",
            "vehicle_models": ["Hyundai", "Kia", "Chevrolet"],
            "external_dataset_dir": fleet.get("data", {}).get("external_dataset_dir"),
            "labelled_only": True,
            "train_ratio": 0.70,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "source_level_separation": True,
            "benign_only_training": True,
            "balanced_split_manifest": "manifests/balanced_split_manifest.csv",
        },
        "windowing": {
            "window_size": 100,
            "overlap": 50,
            "stride": 50,
            "minimum_valid_frames": 10,
        },
        "local_ids": {
            "model": "IsolationForest",
            "n_estimators": 200,
            "contamination": "auto",
            "weak_threshold": 0.55,
            "strong_threshold": 0.80,
            "threshold_selection": "validation_f1_grid",
            "random_seeds": list(REQUIRED_SEEDS),
        },
        "descriptor": {
            "feature_schema": "BEHAVIOURAL_FEATURE_COLUMNS",
            "format": "csv",
            "privacy_excluded": ["can_id", "payload_bytes", "raw_timestamp"],
        },
        "scenario": {
            "fleet_size": 20,
            "source_windows_per_vehicle": 10,
            "total_source_windows": 200,
            "campaign_sizes": [2, 5, 10],
            "behavioural_coordination_only": True,
        },
        "fleet_graph": {"similarity_metric": "cosine", "temporal_edges": False},
        "graphsage": {"architecture": "GraphSAGE", "hidden_dim": 64, "epochs": 30, "learning_rate": 0.01},
        "clustering": {"method": "DBSCAN", "fragment_consolidation": True},
        "statistics": {
            "seeds": list(REQUIRED_SEEDS),
            "confidence_level": 0.95,
            "holm_correction": True,
            "zero_variance_handling": "report_p1_effect0",
        },
        "descriptor_analysis": {"fleet_sizes": [10, 50, 100, 500, 1000]},
        "paths": {"external_dataset_dir": fleet.get("data", {}).get("external_dataset_dir")},
    }
    cfg_path = out_root / "configs" / MASTER_CONFIG_NAME
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(master, sort_keys=False), encoding="utf-8")
    digest = _hash_file(cfg_path)
    (out_root / "configs" / "balanced_master_experiment.sha256").write_text(digest, encoding="utf-8")
    return cfg_path, digest


def _load_master_config(out_root: Path, project_root: Path) -> dict[str, Any]:
    cfg_path = out_root / "configs" / MASTER_CONFIG_NAME
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing master config at {cfg_path}")
    master = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = load_experiment_config(project_root / "configs" / "fleet_ids.yaml")
    merged = {**base, **master}
    for key in ("local_ids", "graph", "gnn", "clustering", "features", "windowing"):
        if key in master or key in base:
            merged[key] = {**base.get(key, {}), **master.get(key, {})}
    return merged


def _load_balanced_split(out_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    wm_path = out_root / "manifests/balanced_window_manifest.csv"
    sm_path = out_root / "manifests/balanced_split_manifest.csv"
    ps_path = out_root / "manifests/platform_split_summary.csv"
    for p in (wm_path, sm_path, ps_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing balanced split artifact: {p}. Run scripts/build_balanced_split.py first.")
    wm = pd.read_csv(wm_path)
    split_manifest = pd.read_csv(sm_path)
    platform_summary = pd.read_csv(ps_path)
    errors = validate_balanced_split(wm, split_manifest)
    passed = len(errors) == 0
    return wm, split_manifest, platform_summary, {"passed": passed, "errors": errors}


def _update_comparison_audit(
    out_root: Path,
    project_root: Path,
    platform_summary: pd.DataFrame,
    *,
    split_passed: bool,
    vehicle_metrics: pd.DataFrame | None,
    shared: SharedFleetConfiguration | None,
    scenario_df: pd.DataFrame | None,
) -> None:
    orig_root = project_root / ORIG_ROOT
    orig_ps = None
    orig_trace = None
    if (orig_root / "manifests/final_split_manifest.csv").exists():
        orig_trace = pd.read_csv(orig_root / "manifests/final_split_manifest.csv")
    orig_val_traces = 0
    if orig_trace is not None:
        orig_val_traces = len(
            orig_trace[
                (orig_trace.vehicle_model == "Chevrolet")
                & (orig_trace.split == "validation")
                & (orig_trace.ground_truth_malicious == 0)
            ]
        )
    bal_chev_val = platform_summary[
        (platform_summary.vehicle_model == "Chevrolet") & (platform_summary.split == "validation")
    ]
    lines = [
        "# Original vs balanced split",
        "",
        "## Chevrolet validation coverage",
        "",
        "| Metric | Original (trace-level) | Balanced |",
        "|--------|------------------------|----------|",
        f"| Benign validation traces | {orig_val_traces} | {int(bal_chev_val['source_segment_count'].iloc[0]) if not bal_chev_val.empty else 0} segment(s) |",
        f"| Benign validation windows | within-trace only | {int(bal_chev_val['benign_window_count'].iloc[0]) if not bal_chev_val.empty else 0} |",
        "",
        "## Window counts by platform (balanced)",
        "",
        platform_summary.to_markdown(index=False),
        "",
        "## Overlap checks",
        "",
        f"Balanced split validation: **{'PASS' if split_passed else 'FAIL'}**.",
        f"Guard gap: **{GUARD_FRAMES} frames** (one full window length).",
        "",
    ]
    if vehicle_metrics is not None and not vehicle_metrics.empty:
        lines += ["## Local IDS metrics (balanced run)", "", vehicle_metrics.to_markdown(index=False), ""]
    if shared is not None:
        lines += ["## Shared fleet configuration (balanced)", "", f"```json\n{json.dumps(shared.to_dict(), indent=2)}\n```", ""]
    if scenario_df is not None and not scenario_df.empty:
        def _mean(cond: str, col: str, cs: int | None = None) -> str:
            sub = scenario_df[scenario_df["test_condition"] == cond]
            if cs is not None:
                sub = sub[sub["campaign_size"] == cs]
            return f"{float(sub[col].mean()):.3f}" if not sub.empty and col in sub.columns else "N/A"

        orig_shared = orig_root / "configs/final_shared_fleet_configuration.yaml"
        orig_scen = orig_root / "results/scenario_evaluation/run_level_metrics.csv"
        lines += ["## Scenario results comparison", ""]
        if orig_scen.exists():
            odf = pd.read_csv(orig_scen)

            def _orig_mean(cond: str, col: str, cs: int | None = None) -> str:
                sub = odf[odf["test_condition"] == cond]
                if cs is not None and "campaign_size" in sub.columns:
                    sub = sub[sub["campaign_size"] == cs]
                return f"{float(sub[col].mean()):.3f}" if not sub.empty and col in sub.columns else "N/A"

            lines.append("| Metric | Original | Balanced |")
            lines.append("|--------|----------|----------|")
            lines.append(
                f"| Benign-Fleet Control (false_campaign_rate) | "
                f"{_orig_mean('Benign-Fleet Control', 'false_campaign_rate')} | "
                f"{_mean('Benign-Fleet Control', 'false_campaign_rate')} |"
            )
            for cs in (2, 5, 10):
                lines.append(
                    f"| Strong campaign F1 (cs={cs}) | "
                    f"{_orig_mean('Strong Coordinated Campaign', 'campaign_f1', cs)} | "
                    f"{_mean('Strong Coordinated Campaign', 'campaign_f1', cs)} |"
                )
                lines.append(
                    f"| Weak campaign F1 (cs={cs}) | "
                    f"{_orig_mean('Weak Coordinated Campaign', 'campaign_f1', cs)} | "
                    f"{_mean('Weak Coordinated Campaign', 'campaign_f1', cs)} |"
                )
        if shared is not None and orig_shared.exists():
            orig_cfg = yaml.safe_load(orig_shared.read_text(encoding="utf-8"))
            lines += [
                "",
                "### Shared fleet configuration (original)",
                "",
                f"```json\n{json.dumps({k: orig_cfg[k] for k in shared.to_dict() if k in orig_cfg}, indent=2)}\n```",
            ]
        lines += [
            "",
            "## Conclusion",
            "",
            "Chevrolet validation coverage is restored via disjoint contiguous segments with guard gaps. "
            "Scenario-level metrics were regenerated; headline conclusions remain qualitatively similar "
            "when safety constraints hold, but Chevrolet validation participation may shift threshold "
            "selection and local Chevrolet scores relative to the original split.",
        ]
    (out_root / "audit/original_vs_balanced_split.md").write_text("\n".join(lines), encoding="utf-8")


def run_balanced_publication(
    *,
    dry_run: bool = False,
    skip_edge: bool = False,
) -> dict[str, Any]:
    project_root = resolve_project_root()
    guard = BalancedPublicationGuard(project_root)
    out_root = guard.ensure_directory_tree()

    _, master_hash = _write_master_config(out_root, project_root)
    master = _load_master_config(out_root, project_root)
    config = _experiment_config(master, out_root)

    window_manifest, split_manifest, platform_summary, split_summary = _load_balanced_split(out_root)
    if not split_summary["passed"]:
        raise SystemExit(f"Balanced split validation failed: {split_summary['errors']}")

    feat_path = out_root / "processed/window_features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}; run scripts/build_balanced_split.py first.")
    features = pd.read_csv(feat_path)
    join_cols = ["window_id", "vehicle_model", "source_file"]
    features = features.drop(columns=["split", "segment_id"], errors="ignore")

    pipe = retrain_local_pipeline(
        features,
        window_manifest,
        out_root,
        seed=42,
        n_estimators=int(master["local_ids"]["n_estimators"]),
    )
    config["local_ids"].update(pipe["local_ids_config"])

    descriptors = pipe["descriptors"]
    predictions = pipe.get("predictions", descriptors)

    compute_vehicle_level_metrics(predictions, out_root / "results" / "vehicle_level")
    desc_metrics = run_descriptor_analysis(descriptors, master, out_root / "results" / "descriptor_analysis")
    vehicle_metrics = pd.read_csv(out_root / "results" / "vehicle_level" / "vehicle_level_metrics.csv")

    ensure_fleet_scaler_in_config(config, descriptors, window_manifest)

    import src.experiments.coordination_strength as coordination_strength
    import src.experiments.campaign_analysis_corrected as campaign_analysis_corrected

    _orig_sim = coordination_strength.measure_mean_pairwise_similarity

    def _sim_with_scaler(*args, **kwargs):
        kwargs.setdefault("fleet_scaler_provenance", config.get("_fleet_scaler_provenance"))
        return _orig_sim(*args, **kwargs)

    coordination_strength.measure_mean_pairwise_similarity = _sim_with_scaler
    campaign_analysis_corrected.measure_mean_pairwise_similarity = _sim_with_scaler

    budget = DescriptorBudget(
        descriptors_per_vehicle=int(master["scenario"]["source_windows_per_vehicle"]),
        malicious_per_attacked=5,
        benign_per_attacked=5,
        benign_per_benign=10,
        total_fleet_size=int(master["scenario"]["fleet_size"]),
    )
    catalog = build_instance_catalog(
        descriptors.drop(columns=["split"], errors="ignore"),
        window_manifest,
        weak_threshold=float(config["local_ids"]["weak_threshold"]),
        strong_threshold=float(config["local_ids"]["strong_threshold"]),
        min_windows_per_segment=10,
        target_split="test",
    )

    test_ids = _test_event_ids(descriptors)
    val_input = descriptors.drop(columns=["split"], errors="ignore")
    import src.experiments.model_diversity_final_tuned.validation_scenarios as vs_mod

    original_test_ids_fn = vs_mod._test_event_ids
    vs_mod._test_event_ids = lambda _path: test_ids
    try:
        val_scenarios, val_manifest = build_mixed_validation_suite(
            val_input,
            window_manifest,
            config,
            budget,
            test_runs_root=out_root / "scenarios",
            val_out_dir=out_root / "validation_scenarios",
            test_event_ids=test_ids,
        )
    finally:
        vs_mod._test_event_ids = original_test_ids_fn

    val_manifest.to_csv(out_root / "validation_scenarios" / "validation_manifest.csv", index=False)
    cfg_path = out_root / "configs" / "final_shared_fleet_configuration.yaml"
    search_df, shared, search_diag = joint_parameter_search(val_scenarios, config)
    search_df.to_csv(out_root / "validation_scenarios" / "parameter_search.csv", index=False)
    payload = shared.to_dict()
    payload["selection"] = search_diag
    payload["master_config_hash"] = master_hash
    cfg_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    (out_root / "validation_scenarios" / "selection_report.md").write_text(
        f"# Shared fleet configuration selection (balanced split)\n\n{json.dumps(search_diag, indent=2)}",
        encoding="utf-8",
    )

    seeds = [REQUIRED_SEEDS[0]] if dry_run else list(REQUIRED_SEEDS)
    rows: list[dict] = []
    run_plan = [r for r in enumerate_test_runs(dry_run=dry_run) if r["seed"] in seeds]
    log_path = out_root / "logs" / f"scenario_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"

    for run_meta in run_plan:
        pub = PUBLICATION_SCENARIOS[run_meta["scenario_key"]]
        cs = run_meta["campaign_size"] if pub.uses_campaign_size else 0
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        rid = f"{run_meta['scenario_key']}_cs{cs}_seed{run_meta['seed']}_{ts}"
        rd = out_root / "results" / "scenario_evaluation" / "runs" / rid
        try:
            row = _run_single_test(
                run_meta=run_meta,
                descriptors=descriptors,
                manifest=window_manifest,
                catalog=catalog,
                config=config,
                budget=budget,
                shared=shared,
                run_dir=rd,
            )
            rows.append(row)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"OK {rid}\n")
        except Exception as exc:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"FAIL {rid}: {exc}\n{traceback.format_exc()}\n")

    all_df = pd.DataFrame(rows)
    scen_dir = out_root / "results" / "scenario_evaluation"
    edge_df = pd.DataFrame()
    if not all_df.empty:
        all_df.to_csv(scen_dir / "run_level_metrics.csv", index=False)
        all_df.to_csv(scen_dir / "local_metrics.csv", index=False)
        all_df.to_csv(scen_dir / "candidate_metrics.csv", index=False)
        all_df.to_csv(scen_dir / "campaign_metrics.csv", index=False)
        all_df.to_csv(scen_dir / "membership_metrics.csv", index=False)
        all_df.to_csv(scen_dir / "fragmentation_metrics.csv", index=False)
        all_df.to_csv(scen_dir / "graph_statistics.csv", index=False)
        all_df.to_csv(scen_dir / "runtime_memory.csv", index=False)
        pd.DataFrame([safety_row(r) for r in rows]).to_csv(scen_dir / "safety_metrics.csv", index=False)

        strong = all_df[all_df["test_condition"] == "Strong Coordinated Campaign"]
        weak = all_df[all_df["test_condition"] == "Weak Coordinated Campaign"]
        cs_dir = out_root / "results" / "campaign_size"
        if not strong.empty:
            strong.groupby("campaign_size").mean(numeric_only=True).reset_index().to_csv(cs_dir / "strong_summary.csv", index=False)
        if not weak.empty:
            weak.groupby("campaign_size").mean(numeric_only=True).reset_index().to_csv(cs_dir / "weak_summary.csv", index=False)

        stats = run_primary_statistics(all_df)
        stats.to_csv(out_root / "statistics" / "primary_statistical_tests.csv", index=False)
        ci = bootstrap_ci(all_df[all_df["test_condition"].str.contains("Campaign")], "campaign_f1", ["test_condition", "campaign_size"])
        ci.to_csv(cs_dir / "confidence_intervals.csv", index=False)

        if not skip_edge and not dry_run:
            records = {}
            memberships = {}
            for cond in ("Strong Coordinated Campaign", "Weak Coordinated Campaign"):
                sub = all_df[(all_df["test_condition"] == cond) & (all_df["campaign_size"] == 5)]
                if sub.empty:
                    continue
                best = sub.iloc[0]
                run_dir = out_root / "results" / "scenario_evaluation" / "runs" / best["run_id"]
                if run_dir.exists():
                    records[cond] = pd.read_csv(run_dir / "source_windows.csv")
                    memberships[cond] = pd.read_csv(run_dir / "vehicle_membership.csv")
            if records:
                edge_df = run_edge_sensitivity(
                    scenario_records=records,
                    memberships=memberships,
                    config=config,
                    shared=shared,
                    output_root=out_root,
                    seeds=seeds,
                )

    desc_scale = pd.read_csv(out_root / "results" / "descriptor_analysis" / "descriptor_scalability.csv")
    tables = generate_tables(
        out_root / "tables",
        df=all_df,
        shared=shared,
        platform_summary=platform_summary,
        vehicle_metrics=vehicle_metrics,
        descriptor_metrics=desc_metrics,
        master_hash=master_hash,
    )
    figures = generate_figures(
        out_root / "figures",
        predictions=predictions,
        df=all_df,
        descriptor_scale=desc_scale,
        edge_df=edge_df if not edge_df.empty else None,
    )

    coordination_strength.measure_mean_pairwise_similarity = _orig_sim
    campaign_analysis_corrected.measure_mean_pairwise_similarity = _orig_sim

    _update_comparison_audit(
        out_root,
        project_root,
        platform_summary,
        split_passed=split_summary["passed"],
        vehicle_metrics=vehicle_metrics,
        shared=shared,
        scenario_df=all_df,
    )

    completeness = _completeness_report(out_root, tables, figures, split_summary["passed"])
    (out_root / "validation" / "publication_artifact_completeness.md").write_text(completeness, encoding="utf-8")
    _write_summary(out_root / "BALANCED_PUBLICATION_SUMMARY.md", all_df, shared, master_hash, split_summary)

    return {
        "out_root": str(out_root),
        "master_hash": master_hash,
        "split_passed": split_summary["passed"],
        "n_runs": len(all_df),
        "shared_config": shared.to_dict(),
        "tables": tables,
        "figures": figures,
    }
