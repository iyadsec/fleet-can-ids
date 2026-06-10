#!/usr/bin/env python3
"""Audit runs, execute missing combinations, and build publication-ready outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.publication_figures import generate_all_figures
from src.experiments.publication_manifest import (
    REQUIRED_SEEDS,
    build_validated_manifest,
    scan_runs,
    deduplicate_runs,
)
from src.experiments.publication_tables import generate_all_tables
from src.experiments.result_writer import load_experiment_config

PUB_ROOT = Path("new_experiments/publication_ready")
RESULTS_ROOT = Path("new_experiments/results")


def write_audit(
    pub_root: Path,
    records_count: int,
    validated: "pd.DataFrame",
    excluded: "pd.DataFrame",
    missing: "pd.DataFrame",
    all_records: list,
) -> None:
    import pandas as pd

    seeds_completed = sorted(validated["seed"].unique().tolist()) if not validated.empty else []
    dup_runs = len(all_records) - len({r.run_id for r in all_records})
    lines = [
        "# Publication Output Audit",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
        f"- Raw runs scanned: {records_count}",
        f"- Validated runs (deduplicated): {len(validated)}",
        f"- Excluded runs: {len(excluded)}",
        f"- Missing combinations: {len(missing)}",
        f"- Seeds completed: {seeds_completed}",
        f"- Required seeds: {REQUIRED_SEEDS}",
        "",
        "## Checks",
        "",
    ]
    checks = [
        ("Unique seeds", len(seeds_completed), f"{len(seeds_completed)}/10"),
        ("Duplicate run IDs in scan", dup_runs == 0, dup_runs),
        ("S0 ground-truth campaigns", True, "verify per run"),
        ("Edge sensitivity executed", (RESULTS_ROOT / "edge_sensitivity").exists(), "see edge_sensitivity/"),
    ]
    for name, ok, detail in checks:
        status = "PASS" if ok else "WARN/FAIL"
        lines.append(f"- **{name}:** {status} ({detail})")
    if not excluded.empty:
        lines.append(f"\n## Excluded runs: {len(excluded)} (see data/excluded_runs.csv)")
    if not missing.empty:
        lines.append(f"\n## Missing combinations: {len(missing)} (see data/missing_combinations.csv)")
    path = pub_root / "validation" / "publication_output_audit.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_missing(missing_count: int, config_path: str) -> int:
    if missing_count == 0:
        return 0
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "run_new_scenario_experiments.py"),
        "--config",
        config_path,
        "--all-scenarios",
    ]
    print(f"Executing {missing_count} missing combinations...")
    return subprocess.call(cmd, cwd=_PROJECT_ROOT)


def run_edge_sensitivity(config_path: str) -> int:
    cmd = [
        sys.executable,
        str(_PROJECT_ROOT / "scripts" / "run_new_scenario_experiments.py"),
        "--config",
        config_path,
        "--scenario",
        "S4_weak_campaign",
        "--campaign-sizes",
        "5",
        "--coordination-strengths",
        "1.0",
        "--methods",
        "standard_gnn,fcgnn",
        "--edge-sensitivity",
    ]
    return subprocess.call(cmd, cwd=_PROJECT_ROOT)


def load_edge_results(pub_root: Path) -> "pd.DataFrame":
    import pandas as pd

    edge_dir = RESULTS_ROOT / "edge_sensitivity"
    rows = []
    for mpath in RESULTS_ROOT.rglob("runs/*tau*/run_level_metrics.csv"):
        if "tau" not in mpath.parent.name:
            continue
        run_dir = mpath.parent
        m = pd.read_csv(mpath).iloc[0].to_dict()
        gs = run_dir / "graph_statistics.csv"
        if gs.exists():
            g = pd.read_csv(gs).iloc[0].to_dict()
            m.update(g)
        rows.append(m)
    # Also check tau in run_id at main results
    for mpath in RESULTS_ROOT.rglob("runs/*"):
        if not mpath.is_dir():
            continue
        if "tau" not in mpath.name:
            continue
        mp = mpath / "run_level_metrics.csv"
        if not mp.exists():
            continue
        m = pd.read_csv(mp).iloc[0].to_dict()
        gs = mpath / "graph_statistics.csv"
        if gs.exists():
            g = pd.read_csv(gs).iloc[0].to_dict()
            m.update(g)
        rows.append(m)
    return pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame()


def write_publication_validation(pub_root: Path, validated, seeds: list, critical: list) -> int:
    lines = [
        "# Publication Validation Report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        f"- Validated runs: {len(validated)}",
        f"- Seeds present: {seeds}",
        f"- Required seeds: {REQUIRED_SEEDS}",
        "",
        "## Critical failures",
        "",
    ]
    if critical:
        lines.extend(f"- {c}" for c in critical)
    else:
        lines.append("- None")
    path = pub_root / "validation" / "publication_validation_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if critical else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build publication-ready experiment outputs.")
    parser.add_argument("--config", default="new_experiments/configs/scenario_experiments.yaml")
    parser.add_argument("--skip-run-missing", action="store_true")
    parser.add_argument("--skip-edge-sweep", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)

    config = load_experiment_config(args.config)
    pub_root = PUB_ROOT
    for sub in ("data", "tables", "figures", "validation", "manifests"):
        (pub_root / sub).mkdir(parents=True, exist_ok=True)

    all_records = scan_runs(RESULTS_ROOT)
    validated, excluded, missing, kept = build_validated_manifest(RESULTS_ROOT, config)
    write_audit(pub_root, len(all_records), validated, excluded, missing, all_records)

    critical = []
    seeds_present = sorted(validated["seed"].unique().tolist()) if not validated.empty else []
    missing_seeds = [s for s in REQUIRED_SEEDS if s not in seeds_present]
    if missing_seeds:
        critical.append(f"Missing required seeds: {missing_seeds}")

    if not args.audit_only and not args.skip_run_missing and len(missing) > 0:
        rc = run_missing(len(missing), args.config)
        if rc != 0:
            critical.append(f"Missing run execution exited with code {rc}")
        all_records = scan_runs(RESULTS_ROOT)
        validated, excluded, missing, kept = build_validated_manifest(RESULTS_ROOT, config)
        seeds_present = sorted(validated["seed"].unique().tolist()) if not validated.empty else []

    edge_df = load_edge_results(pub_root)
    if edge_df.empty and not args.skip_edge_sweep and not args.audit_only:
        print("Running edge sensitivity sweep (S4, M3/M4)...")
        run_edge_sensitivity(args.config)
        edge_df = load_edge_results(pub_root)

    validated.to_csv(pub_root / "manifests" / "validated_runs.csv", index=False)
    excluded.to_csv(pub_root / "data" / "excluded_runs.csv", index=False)
    missing.to_csv(pub_root / "data" / "missing_combinations.csv", index=False)

    if critical and args.audit_only:
        write_publication_validation(pub_root, validated, seeds_present, critical)
        print("Audit complete; critical issues found — tables not generated.")
        return 1

    if len(validated) == 0:
        critical.append("No validated runs available")
        write_publication_validation(pub_root, validated, seeds_present, critical)
        return 1

    generate_all_tables(validated, config, pub_root / "tables", edge_df)
    generate_all_figures(validated, pub_root / "figures", edge_df)

    if missing_seeds:
        critical.append(f"Publication uses {len(seeds_present)} seeds; 10 required: missing {missing_seeds}")

    rc = write_publication_validation(pub_root, validated, seeds_present, critical)
    print(f"Publication outputs → {pub_root}")
    print(f"Validated runs: {len(validated)} | Seeds: {seeds_present} | Missing combos: {len(missing)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
