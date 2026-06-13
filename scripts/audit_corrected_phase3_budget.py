#!/usr/bin/env python3
"""Audit held-out data support for corrected Phase 3 fixed descriptor budget."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.campaign_analysis_corrected import audit_descriptor_budget
from src.experiments.campaign_analysis_writer import load_campaign_analysis_config
from src.experiments.result_writer import load_experiment_config
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

CONFIG = "new_experiments/final_validated_runs/configs/phase3_campaign_size_corrected.yaml"
OUTPUT_ROOT = "new_experiments/final_validated_runs"


def main() -> int:
    project_root = resolve_project_root()
    cfg = load_campaign_analysis_config(CONFIG)
    base = load_experiment_config(cfg["paths"]["base_scenario_config"])
    merged = {**base, **cfg}
    for key in ("local_ids", "graph", "gnn", "campaign"):
        merged[key] = {**base.get(key, {}), **cfg.get(key, {})}

    output_root = project_root / cfg["general"]["output_root"]
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

    try:
        budget, summary = audit_descriptor_budget(catalog)
        passed = True
        err = None
    except ValueError as exc:
        passed = False
        budget = None
        summary = {"failures": [str(exc)]}
        err = str(exc)

    out_dir = project_root / OUTPUT_ROOT / "validation" / "campaign_size_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "corrected_phase3_budget_audit.md"
    lines = [
        "# Corrected Phase 3 Descriptor Budget Audit",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Status:** {'PASS' if passed else 'FAIL'}",
        "",
    ]
    if budget:
        lines.extend(
            [
                "## Chosen budget",
                "",
                f"- Descriptors per vehicle: **{budget.descriptors_per_vehicle}**",
                f"- Malicious per attacked vehicle: **{budget.malicious_per_attacked}**",
                f"- Benign per attacked vehicle: **{budget.benign_per_attacked}**",
                f"- Benign per benign vehicle: **{budget.benign_per_benign}**",
                f"- Expected total nodes: **{budget.expected_total_nodes}**",
                f"- Preferred budget supported: **{summary.get('preferred_budget_supported')}**",
                "",
                f"- Campaign sizes supported: `{summary.get('campaign_sizes_supported')}`",
                "",
            ]
        )
    if err:
        lines.extend(["## Failure", "", f"- {err}", ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Audit → {report} ({'PASS' if passed else 'FAIL'})")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
