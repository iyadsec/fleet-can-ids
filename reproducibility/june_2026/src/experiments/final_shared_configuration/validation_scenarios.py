"""Mixed V0–V4 validation scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.campaign_analysis_corrected import DescriptorBudget
from src.experiments.final_shared_configuration.source_windows import build_validation_campaign_scenario
from src.experiments.model_diversity_final_tuned.validation_scenarios import (
    VALIDATION_SEEDS,
    _scenario_hash,
    build_validation_scenarios,
)
from src.experiments.vehicle_instance_builder import build_instance_catalog

SEARCH_SEEDS = VALIDATION_SEEDS[:3]
V3_SIZES = [2, 5, 10]
V4_SIZES = [2, 5, 10]


def build_mixed_validation_suite(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    budget: DescriptorBudget,
    *,
    test_runs_root: Path,
    val_out_dir: Path,
    test_event_ids: set[str] | None = None,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]], pd.DataFrame]:
    test_event_ids = test_event_ids or set()
    val_catalog = build_instance_catalog(
        descriptors.drop(columns=["split"], errors="ignore"),
        manifest,
        weak_threshold=float(config.get("local_ids", {}).get("weak_threshold", 0.55)),
        strong_threshold=float(config.get("local_ids", {}).get("strong_threshold", 0.80)),
        min_windows_per_segment=10,
        target_split="validation",
    )

    join_cols = [c for c in ("window_id", "vehicle_model", "source_file") if c in descriptors.columns and c in manifest.columns]
    desc = descriptors.merge(manifest[join_cols + ["split"]].drop_duplicates(), on=join_cols, how="left")

    val_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_cache = val_out_dir / "validation_manifest.csv"
    if manifest_cache.exists() and manifest_cache.stat().st_size > 10:
        base_manifest = pd.read_csv(manifest_cache)
    else:
        base_manifest = build_validation_scenarios(
            desc, manifest, val_catalog, config, test_runs_root, val_out_dir,
        )

    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    extra_rows: list[dict[str, Any]] = []

    for cs in V3_SIZES:
        for seed in SEARCH_SEEDS:
            run_id = f"val_V3_cs{cs}_seed{seed}"
            try:
                ann, mem, stats = build_validation_campaign_scenario(
                    attack_strength="strong", campaign_size=cs, seed=seed + cs * 17,
                    descriptors=descriptors, manifest=manifest, config=config, budget=budget,
                )
                eids = set(ann["event_id"].astype(str))
                overlap = len(eids & test_event_ids)
                scenarios[run_id] = (ann, mem, stats)
                extra_rows.append({
                    "validation_run_id": run_id,
                    "validation_seed": seed,
                    "scenario": "V3",
                    "attack_strength": "strong",
                    "campaign_size": cs,
                    "source_trace_ids": "[]",
                    "event_ids": json.dumps(sorted(eids)),
                    "source_window_count": len(ann),
                    "candidate_count": int(stats.get("candidate_nodes", 0)),
                    "scenario_hash": _scenario_hash(ann, "V3", seed),
                    "test_overlap_count": overlap,
                    "validation_passed": overlap == 0,
                })
            except ValueError:
                continue

    for cs in V4_SIZES:
        for seed in SEARCH_SEEDS:
            run_id = f"val_V4_cs{cs}_seed{seed}"
            try:
                ann, mem, stats = build_validation_campaign_scenario(
                    attack_strength="weak", campaign_size=cs, seed=seed + cs * 23,
                    descriptors=descriptors, manifest=manifest, config=config, budget=budget,
                )
                eids = set(ann["event_id"].astype(str))
                overlap = len(eids & test_event_ids)
                scenarios[run_id] = (ann, mem, stats)
                extra_rows.append({
                    "validation_run_id": run_id,
                    "validation_seed": seed,
                    "scenario": "V4",
                    "attack_strength": "weak",
                    "campaign_size": cs,
                    "source_trace_ids": "[]",
                    "event_ids": json.dumps(sorted(eids)),
                    "source_window_count": len(ann),
                    "candidate_count": int(stats.get("candidate_nodes", 0)),
                    "scenario_hash": _scenario_hash(ann, "V4", seed),
                    "test_overlap_count": overlap,
                    "validation_passed": overlap == 0,
                })
            except ValueError:
                continue

    # Load V0–V2 from persisted CSVs
    for _, row in base_manifest.iterrows():
        if row["scenario"] not in ("V0", "V1", "V2"):
            continue
        csv_path = val_out_dir / f"{row['validation_run_id']}_records.csv"
        mem_path = val_out_dir / f"{row['validation_run_id']}_membership.csv"
        if not csv_path.exists():
            continue
        ann = pd.read_csv(csv_path)
        mem = pd.read_csv(mem_path)
        from src.experiments.coordinated_campaign_refinement.candidates import annotate_fleet_candidates, candidate_statistics
        local_cfg = config.get("local_ids", {})
        ann = annotate_fleet_candidates(
            ann,
            weak_threshold=float(local_cfg.get("weak_threshold", 0.55)),
            strong_threshold=float(local_cfg.get("strong_threshold", 0.80)),
        )
        stats = candidate_statistics(ann)
        stats["source_windows"] = len(ann)
        scenarios[str(row["validation_run_id"])] = (ann, mem, stats)

    # Also add V3/V4 cs=5 from base manifest if present
    for _, row in base_manifest.iterrows():
        if row["scenario"] not in ("V3", "V4"):
            continue
        rid = str(row["validation_run_id"])
        if rid in scenarios:
            continue
        csv_path = val_out_dir / f"{rid}_records.csv"
        if not csv_path.exists():
            continue
        ann = pd.read_csv(csv_path)
        mem = pd.read_csv(val_out_dir / f"{rid}_membership.csv")
        from src.experiments.coordinated_campaign_refinement.candidates import annotate_fleet_candidates, candidate_statistics
        ann = annotate_fleet_candidates(ann, weak_threshold=0.55, strong_threshold=0.80)
        stats = candidate_statistics(ann)
        stats["source_windows"] = len(ann)
        scenarios[rid] = (ann, mem, stats)

    manifest_df = pd.concat([base_manifest, pd.DataFrame(extra_rows)], ignore_index=True)
    manifest_df = manifest_df.drop_duplicates(subset=["validation_run_id"], keep="last")
    return scenarios, manifest_df
