"""Orchestrate final shared-configuration experiment."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.experiments.campaign_analysis_corrected import DescriptorBudget, REQUIRED_SEEDS
from src.experiments.coordinated_campaign_refinement.refinement_pipeline import run_refinement_fcgnn
from src.experiments.final_shared_configuration.guard import FinalSharedConfigurationGuard
from src.experiments.final_shared_configuration.metrics import extract_run_metrics, safety_row
from src.experiments.final_shared_configuration.parameter_search import joint_parameter_search
from src.experiments.final_shared_configuration.shared_config import SharedFleetConfiguration
from src.experiments.final_shared_configuration.source_windows import build_test_scenario
from src.experiments.final_shared_configuration.statistics import run_campaign_statistics
from src.experiments.final_shared_configuration.validation_scenarios import build_mixed_validation_suite
from src.experiments.fleet_scaler_loader import ensure_fleet_scaler_in_config
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.scenario_registry import SCENARIO_REGISTRY, enumerate_run_plan
from src.experiments.strong_campaign_extended.metrics import bootstrap_ci
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

CONFIG_SRC = "new_experiments/final_validated_runs/configs/phase3_campaign_size_corrected.yaml"
MANIFEST_SRC = "new_experiments/final_validated_runs/model_diversity_final/manifests/final_window_split_manifest.csv"
SCENARIO_KEYS = list(SCENARIO_REGISTRY.keys())
CAMPAIGN_SIZES = [2, 5, 10]


def _load_config(project_root: Path) -> dict[str, Any]:
    with (project_root / CONFIG_SRC).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _hash_file(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _write_frozen_audit(out: Path, project_root: Path, config: dict) -> None:
    lines = [
        "# Frozen pipeline components",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Non-tunable components",
        "",
        "- Isolation Forest: offline per-OEM models in `data/processed/anomaly_descriptors.csv`",
        f"- Weak threshold: {config.get('local_ids', {}).get('weak_threshold', 0.55)}",
        f"- Strong threshold: {config.get('local_ids', {}).get('strong_threshold', 0.80)}",
        "- GraphSAGE architecture: unchanged (fcgnn, 30 epochs)",
        "- No temporal edges; behavioural similarity graph",
        "",
        "## Recorded hashes",
        "",
        f"- Descriptor table: `{_hash_file(project_root / config['paths']['anomaly_descriptors'])}`",
        f"- Split manifest: `{_hash_file(project_root / 'new_experiments/final_validated_runs/manifests/split_manifest.csv')}`",
        f"- Phase3 config: `{_hash_file(project_root / CONFIG_SRC)}`",
        f"- Scenario registry: `{_hash_file(project_root / 'src/experiments/scenario_registry.py')}`",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


def _test_event_ids(descriptors: pd.DataFrame, manifest: pd.DataFrame) -> set[str]:
    join_cols = [c for c in ("window_id", "vehicle_model", "source_file") if c in descriptors.columns and c in manifest.columns]
    merged = descriptors.merge(manifest[join_cols + ["split"]].drop_duplicates(), on=join_cols, how="left")
    return set(merged.loc[merged["split"] == "test", "event_id"].astype(str))


def _run_single(
    *,
    run_id: str,
    scenario_id: str,
    attack_strength: str,
    seed: int,
    campaign_size: int,
    scenario_key: str,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict,
    budget: DescriptorBudget,
    shared: SharedFleetConfiguration,
    run_dir: Path,
) -> dict[str, Any]:
    source_df, membership, cand_stats = build_test_scenario(
        scenario_key, seed=seed, campaign_size=campaign_size,
        descriptors=descriptors, manifest=manifest, catalog=catalog,
        config=config, budget=budget,
    )
    ensure_fleet_scaler_in_config(config, source_df, membership)
    design = shared.to_refinement_design()
    outputs, _, annotated, consol = run_refinement_fcgnn(source_df, membership, config, seed, design)

    run_dir.mkdir(parents=True, exist_ok=True)
    source_df.to_csv(run_dir / "source_windows.csv", index=False)
    annotated.to_csv(run_dir / "annotated_source_records.csv", index=False)
    membership.to_csv(run_dir / "vehicle_membership.csv", index=False)
    outputs.event_predictions.to_csv(run_dir / "event_predictions.csv", index=False)
    outputs.graph_stats.to_csv(run_dir / "graph_statistics.csv", index=False)
    outputs.edge_list.to_csv(run_dir / "edge_list.csv", index=False)
    if not consol.empty:
        consol.to_csv(run_dir / "fragment_consolidation_log.csv", index=False)

    spec = SCENARIO_REGISTRY[scenario_key]
    expect = spec.expect_coordinated_campaign
    row = extract_run_metrics(
        run_id=run_id, scenario_id=scenario_id, attack_strength=attack_strength,
        seed=seed, campaign_size=campaign_size, source_df=source_df, membership=membership,
        outputs=outputs, cand_stats=cand_stats, expect_campaign=expect,
    )
    row["scenario_key"] = scenario_key
    return row


def run_final_shared_configuration(*, dry_run: bool = False) -> dict[str, Any]:
    project_root = resolve_project_root()
    guard = FinalSharedConfigurationGuard(project_root)
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
    manifest_path = project_root / MANIFEST_SRC
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        splits = config.get("splits", {})
        manifest = ensure_split_manifest(
            descriptors, features,
            output_path=project_root / "new_experiments/final_validated_runs/manifests/split_manifest.csv",
            seed=int(splits.get("seed", 42)),
            train_ratio=float(splits.get("train", 0.70)),
            val_ratio=float(splits.get("validation", 0.15)),
            test_ratio=float(splits.get("test", 0.15)),
        )
    local_cfg = config.get("local_ids", {})
    catalog = build_instance_catalog(
        descriptors.drop(columns=["split"], errors="ignore"),
        manifest,
        weak_threshold=float(local_cfg.get("weak_threshold", 0.55)),
        strong_threshold=float(local_cfg.get("strong_threshold", 0.80)),
        min_windows_per_segment=10,
        target_split="test",
    )
    manifest.to_csv(out_root / "manifests" / "split_manifest_reference.csv", index=False)

    _write_frozen_audit(out_root / "audit" / "frozen_pipeline_components.md", project_root, config)

    test_ids = _test_event_ids(descriptors, manifest)
    val_scenarios, val_manifest = build_mixed_validation_suite(
        descriptors, manifest, config, budget,
        test_runs_root=project_root / "new_experiments/final_validated_runs/results",
        val_out_dir=out_root / "validation_scenarios",
        test_event_ids=test_ids,
    )
    val_manifest.to_csv(out_root / "validation_scenarios" / "validation_manifest.csv", index=False)

    cfg_path = out_root / "configs" / "final_shared_fleet_configuration.yaml"
    search_csv = out_root / "parameter_search" / "joint_shared_parameter_search.csv"

    if cfg_path.exists() and cfg_path.stat().st_size > 10:
        frozen = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        shared = SharedFleetConfiguration.from_dict(frozen)
        search_df = pd.read_csv(search_csv) if search_csv.exists() and search_csv.stat().st_size > 10 else pd.DataFrame()
        search_diag = frozen.get("selection", {"candidates_tested": len(search_df), "constraints_met": False})
    else:
        search_df, shared, search_diag = joint_parameter_search(val_scenarios, config)
        search_df.to_csv(search_csv, index=False)
        cfg_payload = shared.to_dict()
        cfg_payload["selection"] = search_diag
        cfg_payload["validation_scenarios"] = ["V0", "V1", "V2", "V3_cs2", "V3_cs5", "V3_cs10", "V4_cs2", "V4_cs5", "V4_cs10"]
        cfg_path.write_text(yaml.safe_dump(cfg_payload, sort_keys=False), encoding="utf-8")
        (out_root / "configs" / "final_shared_fleet_configuration.sha256").write_text(
            hashlib.sha256(cfg_path.read_bytes()).hexdigest(), encoding="utf-8"
        )

    cfg_payload = shared.to_dict()
    if cfg_path.exists():
        cfg_payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    seeds = [REQUIRED_SEEDS[0]] if dry_run else REQUIRED_SEEDS
    runs = enumerate_run_plan(
        scenario_keys=SCENARIO_KEYS,
        methods=["fcgnn"],
        seeds=seeds,
        campaign_sizes=CAMPAIGN_SIZES,
        coordination_strengths=[0.0, 1.0],
    )

    rows: list[dict] = []
    run_root = out_root / "results" / "scenarios" / "runs"
    log = out_root / "logs" / f"test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"

    for entry in runs:
        spec = SCENARIO_REGISTRY[entry["scenario_key"]]
        atk = "strong" if spec.scenario_id == "S3" else ("weak" if spec.scenario_id == "S4" else "n/a")
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        rid = f"shared_{spec.scenario_id}_n{entry['campaign_size']}_seed{entry['seed']}_{ts}"
        rd = run_root / rid
        try:
            row = _run_single(
                run_id=rid, scenario_id=spec.scenario_id, attack_strength=atk,
                seed=entry["seed"], campaign_size=entry["campaign_size"],
                scenario_key=entry["scenario_key"],
                descriptors=descriptors, manifest=manifest, catalog=catalog,
                config=config, budget=budget, shared=shared, run_dir=rd,
            )
            rows.append(row)
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"OK {rid}\n")
        except Exception as exc:
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"FAIL {rid}: {exc}\n")

    all_df = pd.DataFrame(rows)
    scen_dir = out_root / "results" / "scenarios"
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

        s3 = all_df[all_df["scenario_id"] == "S3"]
        s4 = all_df[all_df["scenario_id"] == "S4"]
        cs_dir = out_root / "results" / "campaign_size"
        if not s3.empty:
            s3.groupby("campaign_size").mean(numeric_only=True).reset_index().to_csv(cs_dir / "strong_summary.csv", index=False)
        if not s4.empty:
            s4.groupby("campaign_size").mean(numeric_only=True).reset_index().to_csv(cs_dir / "weak_summary.csv", index=False)
        all_df.groupby(["scenario_id", "campaign_size"]).mean(numeric_only=True).reset_index().to_csv(
            cs_dir / "candidate_transmission_summary.csv", index=False
        )
        stats = run_campaign_statistics(all_df)
        if not stats.empty:
            stats.to_csv(out_root / "statistical_analysis" / "statistical_tests.csv", index=False)
            stats.to_csv(cs_dir / "statistical_tests.csv", index=False)
        ci = bootstrap_ci(all_df[all_df["scenario_id"].isin(["S3", "S4"])], "campaign_f1", ["scenario_id", "campaign_size"])
        ci.to_csv(cs_dir / "confidence_intervals.csv", index=False)

        search_df.head(1).to_csv(out_root / "results" / "graph_connectivity" / "shared_configuration_diagnostics.csv", index=False)
        bw = all_df.groupby("scenario_id")[["source_windows", "candidate_nodes", "transmission_percentage"]].mean()
        bw.to_csv(out_root / "results" / "communication_cost" / "bandwidth_summary.csv")

    from src.experiments.final_shared_configuration.tables import generate_tables
    from src.experiments.final_shared_configuration.figures import generate_figures
    from src.experiments.final_shared_configuration.audit import write_comparison_audits

    tables = generate_tables(out_root / "tables", all_df, shared, cfg_payload)
    figures = generate_figures(out_root / "figures", all_df)
    write_comparison_audits(out_root / "audit", all_df, shared, search_diag)

    _write_summary(out_root / "FINAL_SHARED_CONFIGURATION_SUMMARY.md", all_df, shared, search_diag, val_manifest, tables, figures)
    return {"out_root": str(out_root), "n_runs": len(all_df), "shared_config": shared.to_dict(), "candidates_tested": search_diag.get("candidates_tested", 0)}


def _write_summary(out: Path, df: pd.DataFrame, shared: SharedFleetConfiguration, diag: dict, val_manifest: pd.DataFrame, tables: list, figures: list) -> None:
    def m(sid, cs, col):
        sub = df[(df["scenario_id"] == sid) & (df["campaign_size"] == cs)]
        return float(sub[col].mean()) if col in sub.columns and not sub.empty else None

    lines = [
        "# Final shared configuration summary",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Key answers",
        "",
        "1. **Identical fleet parameters for S3 and S4:** Yes — one frozen `SharedFleetConfiguration`.",
        f"2. **Validation scenarios:** V0–V4 mixed suite ({len(val_manifest)} runs); overlap count max = {int(val_manifest['test_overlap_count'].max()) if not val_manifest.empty else 'N/A'}.",
        f"3. **Selected configuration:** tau={shared.similarity_threshold}, k_same={shared.max_same_vehicle_neighbors}, k_cross={shared.max_cross_vehicle_neighbors}, eps={shared.dbscan_eps}, min_samples={shared.dbscan_min_samples}.",
        f"4. **S0 campaign-free:** false campaign rate mean = {m('S0', 0, 'false_campaign_rate')}.",
        f"5. **S1 isolated:** qualifying clusters mean = {df[df['scenario_id']=='S1']['n_predicted_campaign_clusters'].mean() if 'n_predicted_campaign_clusters' in df.columns else 'N/A'}.",
        f"6. **S2 separation:** incorrect merging mean = {df[df['scenario_id']=='S2']['incorrect_merging'].mean() if not df[df['scenario_id']=='S2'].empty else 'N/A'}.",
        f"7–8. **S3/S4 by size:** see `results/campaign_size/strong_summary.csv` and `weak_summary.csv`.",
        f"9. **Candidate transmission:** mean {df['transmission_percentage'].mean():.1f}% across runs." if "transmission_percentage" in df.columns else "",
        "10–18. See tables FS1–FS8 and figures FS1–FS10.",
        f"16. Tables: {', '.join(tables)}; Figures: {', '.join(figures)}",
        "17. Extended scalability: see `audit/extended_scalability_readiness.md`.",
        "18. Benign weak candidates may still enter graph when score ≥ weak threshold.",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
