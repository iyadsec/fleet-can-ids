#!/usr/bin/env python3
"""Phase 3: campaign-size sensitivity under final_validated_runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.campaign_analysis_audit import run_data_availability_audit
from src.experiments.campaign_analysis_outputs import collect_run_metrics
from src.experiments.campaign_analysis_writer import CampaignAnalysisGuard, load_campaign_analysis_config
from src.experiments.final_validated_phase3_outputs import (
    export_phase3_tables,
    generate_phase3_figures,
    write_phase3_summary,
)
from src.experiments.result_writer import load_experiment_config
from src.utils.paths import resolve_project_root

CONFIG = "new_experiments/final_validated_runs/configs/final_validated_campaign_analysis.yaml"
PYTHON = sys.executable


def main() -> int:
    project_root = resolve_project_root()

    # Phase 2 gate
    code = subprocess.call([PYTHON, "scripts/validate_phase2_completion.py"], cwd=project_root)
    if code != 0:
        print("Phase 2 completion check failed — aborting Phase 3.", file=sys.stderr)
        return code

    campaign_cfg = load_campaign_analysis_config(CONFIG)
    # Phase 3 only — disable model-diversity experiment
    campaign_cfg = {**campaign_cfg, "experiment_b": {**campaign_cfg.get("experiment_b", {}), "enabled": False}}
    import yaml
    phase3_cfg_path = project_root / "new_experiments/final_validated_runs/configs/phase3_campaign_size_only.yaml"
    phase3_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    phase3_cfg_path.write_text(yaml.safe_dump(campaign_cfg, sort_keys=False), encoding="utf-8")

    guard = CampaignAnalysisGuard(project_root, campaign_cfg["general"]["output_root"])
    catalog, audit_summary = run_data_availability_audit(
        {**load_experiment_config(campaign_cfg["paths"]["base_scenario_config"]), **campaign_cfg},
        guard.output_root,
    )
    audit_path = guard.output_root / "validation" / "phase3_data_availability_audit.md"
    (guard.output_root / "validation").mkdir(parents=True, exist_ok=True)
    if (guard.output_root / "validation" / "data_availability_audit.md").exists():
        audit_path.write_text(
            (guard.output_root / "validation" / "data_availability_audit.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    subprocess.call(
        [
            PYTHON,
            "scripts/run_campaign_analysis.py",
            "--config",
            str(phase3_cfg_path.relative_to(project_root)),
            "--experiment",
            "campaign_size",
            "--reuse-existing",
        ],
        cwd=project_root,
    )

    df = collect_run_metrics(guard.output_root, "campaign_size")
    if df.empty:
        print("No campaign_size runs found — aborting.", file=sys.stderr)
        return 1
    export_phase3_tables(df, guard.output_root)
    generate_phase3_figures(df, guard.output_root)

    val_code = subprocess.call([PYTHON, "scripts/validate_phase3_campaign_size.py"], cwd=project_root)
    write_phase3_summary(guard.output_root, df, audit_summary, validation_passed=(val_code == 0))
    return val_code


if __name__ == "__main__":
    raise SystemExit(main())
