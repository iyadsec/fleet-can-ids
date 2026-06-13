#!/usr/bin/env python3
"""Run corrected Phase 3 campaign-size sensitivity (240 runs)."""

from __future__ import annotations

import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.campaign_analysis_corrected import DescriptorBudget, audit_descriptor_budget
from src.experiments.campaign_analysis_runner import run_campaign_analysis_corrected_single
from src.experiments.campaign_analysis_writer import (
    CampaignAnalysisGuard,
    CampaignRunContext,
    load_campaign_analysis_config,
)
from src.experiments.campaign_size_corrected_outputs import (
    collect_corrected_metrics,
    export_corrected_raw_outputs,
    export_corrected_tables,
    generate_corrected_figures,
    write_corrected_summary,
    write_original_vs_corrected,
)
from src.experiments.result_writer import RunAlreadyExistsError, load_experiment_config
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.scenario_registry import resolve_method
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

CONFIG = "new_experiments/final_validated_runs/configs/phase3_campaign_size_corrected.yaml"
EXPERIMENT = "campaign_size_corrected"
PYTHON = sys.executable


def _merge_config(cfg: dict) -> dict:
    base_path = cfg.get("paths", {}).get("base_scenario_config")
    base = load_experiment_config(base_path) if base_path else {}
    merged = {**base, **cfg}
    for key in ("local_ids", "graph", "gnn", "campaign"):
        merged[key] = {**base.get(key, {}), **cfg.get(key, {})}
    return merged


def main() -> int:
    project_root = resolve_project_root()

    audit_code = subprocess.call([PYTHON, "scripts/audit_corrected_phase3_budget.py"], cwd=project_root)
    if audit_code != 0:
        print("Budget audit failed — aborting corrected Phase 3.", file=sys.stderr)
        return audit_code

    cfg = load_campaign_analysis_config(CONFIG)
    merged = _merge_config(cfg)
    output_root = project_root / cfg["general"]["output_root"]
    guard = CampaignAnalysisGuard(project_root, cfg["general"]["output_root"])
    guard.ensure_directory_tree()

    archive_note = output_root / "results" / "campaign_size" / "PRELIMINARY_PHASE3_ARCHIVE.md"
    if not archive_note.exists():
        archive_note.parent.mkdir(parents=True, exist_ok=True)
        archive_note.write_text(
            "# Preliminary Phase 3 Archive\n\n"
            "Original campaign-size results under `results/campaign_size/` are retained as "
            "**preliminary** (variable node counts, uncontrolled platform composition).\n\n"
            "Publication results use `results/campaign_size_corrected/` only.\n",
            encoding="utf-8",
        )

    descriptors, features = load_descriptor_tables(
        project_root / cfg["paths"]["anomaly_descriptors"],
        project_root / cfg["paths"]["window_features"],
    )
    splits = merged.get("splits", {})
    manifest = ensure_split_manifest(
        descriptors,
        features,
        output_path=output_root / "manifests" / "split_manifest.csv",
        seed=int(splits.get("seed", 42)),
        train_ratio=float(splits.get("train", 0.70)),
        val_ratio=float(splits.get("validation", 0.15)),
        test_ratio=float(splits.get("test", 0.15)),
    )
    local_cfg = merged.get("local_ids", {})
    catalog = build_instance_catalog(
        descriptors,
        manifest,
        weak_threshold=float(local_cfg.get("weak_threshold", 0.55)),
        strong_threshold=float(local_cfg.get("strong_threshold", 0.80)),
        min_windows_per_segment=int(merged.get("campaign", {}).get("min_windows_per_segment", 10)),
    )
    budget, audit_summary = audit_descriptor_budget(catalog)
    cc = cfg.get("corrected_campaign_size", {})
    sizes = cc.get("campaign_sizes", [2, 5, 10])
    strengths = cc.get("attack_strengths", ["strong", "weak"])
    seeds = [int(s) for s in cfg["general"]["seeds"]]
    methods = [
        resolve_method(a)
        for a, ok in [
            ("local", cfg.get("methods", {}).get("local_ids", True)),
            ("clustering", cfg.get("methods", {}).get("descriptor_clustering", True)),
            ("standard_gnn", cfg.get("methods", {}).get("standard_gnn", True)),
            ("fcgnn", cfg.get("methods", {}).get("fcgnn", True)),
        ]
        if ok
    ]
    coord = float(cc.get("coordination_strength", 1.0))
    fleet = int(cc.get("total_fleet_size", 20))

    log_dir = output_root / "logs" / "campaign_size_corrected"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
    completed = failed = skipped = 0
    excluded_rows: list[dict] = []

    for strength in strengths:
        for cs in sizes:
            for seed in seeds:
                for method in methods:
                    try:
                        ctx = CampaignRunContext.create(
                            guard=guard,
                            experiment=EXPERIMENT,
                            attack_strength=strength,
                            method=method,
                            seed=seed,
                            campaign_size=int(cs),
                            coordination_strength=coord,
                            overwrite=bool(cfg.get("general", {}).get("overwrite", False)),
                        )
                        run_campaign_analysis_corrected_single(
                            ctx,
                            descriptors=descriptors,
                            manifest=manifest,
                            catalog=catalog,
                            config=merged,
                            budget=budget,
                            total_fleet_size=fleet,
                        )
                        completed += 1
                        msg = f"OK {strength} {method} seed={seed} cs={cs}"
                        print(msg)
                        with log_path.open("a", encoding="utf-8") as fh:
                            fh.write(msg + "\n")
                    except RunAlreadyExistsError:
                        skipped += 1
                    except Exception as exc:
                        failed += 1
                        excluded_rows.append(
                            {
                                "attack_strength": strength,
                                "campaign_size": cs,
                                "seed": seed,
                                "method": method,
                                "reason": str(exc),
                            }
                        )
                        print(f"FAIL {strength} cs={cs} seed={seed} {method}: {exc}", file=sys.stderr)
                        with log_path.open("a", encoding="utf-8") as fh:
                            fh.write(f"FAIL {strength} cs={cs} seed={seed} {method}: {exc}\n")
                            fh.write(traceback.format_exc() + "\n")

    res_dir = output_root / "results" / EXPERIMENT
    res_dir.mkdir(parents=True, exist_ok=True)
    if excluded_rows:
        import pandas as pd

        pd.DataFrame(excluded_rows).to_csv(res_dir / "excluded_runs.csv", index=False)
    else:
        (res_dir / "excluded_runs.csv").write_text(
            "attack_strength,campaign_size,seed,method,reason\n", encoding="utf-8"
        )

    df = collect_corrected_metrics(output_root)
    export_corrected_raw_outputs(df, output_root)
    export_corrected_tables(df, output_root)
    generate_corrected_figures(df, output_root)
    write_original_vs_corrected(output_root, df)
    write_corrected_summary(
        output_root,
        df,
        audit_summary,
        budget={
            **audit_summary.get("chosen_budget", {}),
            "expected_total_nodes": audit_summary.get("expected_total_nodes"),
        },
        validation_passed=False,
        excluded_runs=__import__("pandas").read_csv(res_dir / "excluded_runs.csv")
        if (res_dir / "excluded_runs.csv").stat().st_size > 40
        else None,
    )

    val_code = subprocess.call(
        [PYTHON, "scripts/validate_corrected_phase3_campaign_size.py"],
        cwd=project_root,
    )
    write_corrected_summary(
        output_root,
        df,
        audit_summary,
        budget={
            **audit_summary.get("chosen_budget", {}),
            "expected_total_nodes": audit_summary.get("expected_total_nodes"),
        },
        validation_passed=(val_code == 0),
        excluded_runs=__import__("pandas").read_csv(res_dir / "excluded_runs.csv")
        if (res_dir / "excluded_runs.csv").stat().st_size > 40
        else None,
    )

    print(f"Completed={completed}, skipped={skipped}, failed={failed}")
    return val_code if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
