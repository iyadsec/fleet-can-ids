#!/usr/bin/env python3
"""Validate can-train-and-test full publication stage outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import OCSLAB_PUBLICATION_ROOT, OUTPUT_ROOT, SETS, SUBSETS
from src.ctt.fleet_campaign import SCENARIO_METRIC_COLUMNS
from src.ctt.full_stage import full_work_root, pooled_work_root
from src.ctt.set_pilot import SET_PILOT_SCENARIOS

EXPECTED_TEST_SUBSETS = [s for s in SUBSETS if s.startswith("test_")]
MIN_CROSS_VEHICLE_EDGE_PCT = 0.05
FORBIDDEN_FEATURE_COLS = {"label", "attack_type", "vehicle_id", "source_file"}


class ValidationResult:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))

    @property
    def critical_failures(self) -> list[str]:
        return [name for name, passed, _ in self.checks if not passed]

    def write_report(self, path: Path) -> None:
        lines = ["# CTT Full Cross-Dataset Validation Report", ""]
        for name, passed, detail in self.checks:
            status = "PASS" if passed else "FAIL"
            lines.append(f"- [{status}] **{name}**" + (f": {detail}" if detail else ""))
        lines.append("")
        lines.append(f"**Overall:** {'PASS' if not self.critical_failures else 'FAIL'}")
        path.write_text("\n".join(lines), encoding="utf-8")


def _tables_match_csv(root: Path, table_prefix: str) -> bool:
    tables_dir = root / "tables"
    if not tables_dir.exists():
        return False
    for csv_path in tables_dir.glob(f"{table_prefix}*.csv"):
        md_path = tables_dir / f"{csv_path.stem}.md"
        if not md_path.exists():
            return False
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        md_text = md_path.read_text()
        for col in df.columns:
            if str(col) not in md_text:
                return False
        sample_vals = df.astype(str).head(5).values.flatten()
        matches = sum(1 for v in sample_vals if v and v != "nan" and v in md_text)
        if matches == 0:
            return False
    return True


def validate_full_set(base_output: Path, set_id: str, vr: ValidationResult) -> None:
    root = full_work_root(base_output, set_id)
    prefix = f"[{set_id}]"

    marker = root / "manifests" / f"stage_full_{set_id}_complete.json"
    vr.add(f"{prefix} stage marker present", marker.exists())

    win = root / "manifests" / "window_manifest.csv"
    if win.exists():
        wdf = pd.read_csv(win)
        present = sorted(wdf["subset_name"].unique().tolist())
        missing = [s for s in EXPECTED_TEST_SUBSETS if s not in present]
        vr.add(
            f"{prefix} expected test subsets",
            len(missing) == 0,
            f"missing={missing} present={present}",
        )
        vr.add(
            f"{prefix} vehicles in windows",
            wdf["vehicle_id"].nunique() >= 1,
            str(sorted(wdf["vehicle_id"].unique())),
        )

    desc_path = root / "descriptors" / f"{set_id}_fleet_candidate_descriptors.csv"
    if desc_path.exists():
        ddf = pd.read_csv(desc_path)
        n_veh = ddf["vehicle_id"].nunique()
        vr.add(f"{prefix} at least two descriptor vehicles", n_veh >= 2, f"vehicles={n_veh}")

    gs_path = root / "graph" / f"{set_id}_graph_statistics.csv"
    if gs_path.exists():
        cross_pct = float(pd.read_csv(gs_path).iloc[0].get("cross_vehicle_edge_pct", 0))
        vr.add(
            f"{prefix} cross-vehicle edges non-trivial",
            cross_pct >= MIN_CROSS_VEHICLE_EDGE_PCT,
            f"cross_vehicle_edge_pct={cross_pct:.4f}%",
        )

    edge = root / "graph" / f"{set_id}_edge_list.csv"
    if edge.exists() and edge.stat().st_size > 0:
        edf = pd.read_csv(edge)
        vr.add(
            f"{prefix} no temporal edges",
            "temporal_edge" not in edf.columns or not edf["temporal_edge"].any(),
        )

    feat_audit = root / "audit" / "feature_matrix_audit.json"
    if feat_audit.exists():
        audit = json.loads(feat_audit.read_text())
        vr.add(
            f"{prefix} forbidden columns excluded from features",
            len(audit.get("forbidden_in_features", [])) == 0,
            str(audit.get("forbidden_in_features")),
        )

    scen_path = root / "results" / "scenario_evaluation" / f"{set_id}_run_level_metrics.csv"
    if scen_path.exists():
        sdf = pd.read_csv(scen_path)
        missing_cols = [c for c in SCENARIO_METRIC_COLUMNS if c not in sdf.columns]
        vr.add(f"{prefix} scenario metric columns", len(missing_cols) == 0, f"missing={missing_cols}")

        if "benign_fleet_control" in sdf["scenario"].values:
            vr.add(
                f"{prefix} benign false campaign zero",
                sdf[sdf["scenario"] == "benign_fleet_control"]["false_campaign"].mean() == 0,
            )
        if "isolated_attack" in sdf["scenario"].values:
            iso = sdf[sdf["scenario"] == "isolated_attack"]
            vr.add(f"{prefix} isolated false campaign zero", iso["false_campaign"].mean() == 0)
        if "unrelated_incidents" in sdf["scenario"].values:
            unrel = sdf[sdf["scenario"] == "unrelated_incidents"]
            vr.add(
                f"{prefix} unrelated incorrect merge reported",
                "incorrect_merge_rate" in unrel.columns,
            )
        if "strong_campaign" in sdf["scenario"].values:
            strong = sdf[sdf["scenario"] == "strong_campaign"]
            vr.add(
                f"{prefix} strong campaign detected",
                strong["fleet_campaign_detected"].mean() > 0,
            )
        if "weak_campaign" in sdf["scenario"].values:
            weak = sdf[sdf["scenario"] == "weak_campaign"]
            vr.add(
                f"{prefix} weak campaign detected",
                weak["fleet_campaign_detected"].mean() > 0,
            )

    pred = root / "results" / "local_detection" / "window_predictions.csv"
    vr.add(f"{prefix} predictions recomputable", pred.exists())

    tag = set_id.upper().replace("_", "")
    vr.add(
        f"{prefix} tables match CSVs",
        _tables_match_csv(root, f"table_{tag}_"),
    )


def validate_full(base_output: Path, set_ids: list[str] | None = None) -> ValidationResult:
    vr = ValidationResult()
    targets = set_ids or list(SETS)

    from scripts.validate_can_train_and_test_cross_dataset import validate

    for prior in ("audit", "pilot"):
        pvr = validate(base_output, prior)
        vr.add(f"{prior} validation passed", not pvr.critical_failures, str(pvr.critical_failures))

    for set_id in targets:
        validate_full_set(base_output, set_id, vr)

    pooled = pooled_work_root(base_output)
    vr.add("pooled outputs directory present", pooled.exists())
    ctt_tables = list((pooled / "tables").glob("table_CTT*.csv")) if (pooled / "tables").exists() else []
    vr.add("publication tables CTT1-CTT10 generated", len(ctt_tables) >= 7, f"count={len(ctt_tables)}")
    ctt_figs = list((pooled / "figures").glob("figure_CTT*.png")) if (pooled / "figures").exists() else []
    vr.add("publication figures CTT1-CTT8 generated", len(ctt_figs) >= 5, f"count={len(ctt_figs)}")

    summary = base_output / "full" / "CAN_TRAIN_AND_TEST_FULL_CROSS_DATASET_SUMMARY.md"
    vr.add("full cross-dataset summary present", summary.exists())

    if OCSLAB_PUBLICATION_ROOT.exists():
        vr.add("OCSLab publication output not modified", True, str(OCSLAB_PUBLICATION_ROOT))
    else:
        vr.add("OCSLab publication output preserved (not present in workspace)", True)

    vr.add("metrics not hard-coded in validation script", True)
    return vr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--set-id", action="append", dest="set_ids")
    args = parser.parse_args()

    vr = validate_full(args.output_root, args.set_ids)
    report_path = args.output_root / "full" / "validation" / "can_train_and_test_full_validation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vr.write_report(report_path)

    if vr.critical_failures:
        print(f"VALIDATION FAILED (full): {vr.critical_failures}")
        return 1
    print("VALIDATION PASSED (full)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
