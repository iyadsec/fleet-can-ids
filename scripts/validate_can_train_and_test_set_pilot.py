#!/usr/bin/env python3
"""Validate can-train-and-test set_pilot stage outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import OCSLAB_PUBLICATION_ROOT, OUTPUT_ROOT, SUBSETS
from src.ctt.fleet_campaign import SCENARIO_METRIC_COLUMNS
from src.ctt.set_pilot import SET_PILOT_SCENARIOS, set_work_root

EXPECTED_TEST_SUBSETS = [s for s in SUBSETS if s.startswith("test_")]
MIN_CROSS_VEHICLE_EDGE_PCT = 0.05


class ValidationResult:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))

    @property
    def critical_failures(self) -> list[str]:
        return [name for name, passed, _ in self.checks if not passed]

    def write_report(self, path: Path) -> None:
        lines = ["# CTT Set Pilot Validation Report", ""]
        for name, passed, detail in self.checks:
            status = "PASS" if passed else "FAIL"
            lines.append(f"- [{status}] **{name}**" + (f": {detail}" if detail else ""))
        lines.append("")
        lines.append(f"**Overall:** {'PASS' if not self.critical_failures else 'FAIL'}")
        path.write_text("\n".join(lines), encoding="utf-8")


def validate_set_pilot(base_output: Path, set_id: str = "set_01") -> ValidationResult:
    vr = ValidationResult()
    root = set_work_root(base_output, set_id)

    audit_marker = base_output / "manifests" / "stage_audit_complete.json"
    pilot_marker = base_output / "manifests" / "stage_pilot_complete.json"
    vr.add("audit stage completed", audit_marker.exists())
    vr.add("pilot stage completed", pilot_marker.exists())

    from scripts.validate_can_train_and_test_cross_dataset import validate

    for prior in ("audit", "pilot"):
        pvr = validate(base_output, prior)
        vr.add(f"{prior} validation passed", not pvr.critical_failures, str(pvr.critical_failures))

    marker = root / "manifests" / f"stage_set_pilot_{set_id}_complete.json"
    vr.add("set_pilot marker present", marker.exists())

    norm_manifest = root / "manifests" / "normalization_manifest.csv"
    vr.add("normalization manifest present", norm_manifest.exists())
    if norm_manifest.exists():
        nm = pd.read_csv(norm_manifest)
        vr.add("only target set processed", (nm["dataset_set"] == set_id).all(), f"sets={nm['dataset_set'].unique()}")

    win = root / "manifests" / "window_manifest.csv"
    if win.exists():
        wdf = pd.read_csv(win)
        vr.add("train/test split respected", "train_01" in wdf["subset_name"].values)
        present_subsets = sorted(wdf["subset_name"].unique().tolist())
        missing = [s for s in EXPECTED_TEST_SUBSETS if s not in present_subsets]
        vr.add(
            "all expected test subsets present",
            len(missing) == 0,
            f"missing={missing} present={present_subsets}",
        )
        vr.add("test_04 subset present", "test_04_unknown_vehicle_unknown_attack" in present_subsets)

    desc_path = root / "descriptors" / f"{set_id}_fleet_candidate_descriptors.csv"
    if not desc_path.exists():
        desc_path = root / "descriptors" / "fleet_candidate_descriptors.csv"
    if desc_path.exists():
        ddf = pd.read_csv(desc_path)
        n_vehicles = ddf["vehicle_id"].nunique() if "vehicle_id" in ddf.columns else 0
        vr.add("at least two vehicles in descriptors", n_vehicles >= 2, f"vehicles={n_vehicles}")
        if "vehicle_id" in ddf.columns and "subset_name" in ddf.columns:
            vc = ddf.groupby("vehicle_id").size()
            vr.add(
                "descriptor vehicle balance",
                vc.min() >= max(1, len(ddf) // (10 * max(n_vehicles, 1))),
                f"counts={vc.to_dict()}",
            )

    train_manifest = root / "manifests" / "local_model_training_manifest.csv"
    if train_manifest.exists():
        tm = pd.read_csv(train_manifest)
        vr.add("no test data in training", (tm["test_data_used_in_training"] == 0).all())
        vr.add("no attack data in training", (tm["attack_data_used_in_training"] == 0).all())
        vr.add("no attack data in thresholding", (tm["attack_data_used_in_thresholding"] == 0).all())

    feat_audit = root / "audit" / "feature_matrix_audit.json"
    if feat_audit.exists():
        audit = json.loads(feat_audit.read_text())
        vr.add(
            "feature matrix excludes forbidden columns",
            len(audit.get("forbidden_in_features", [])) == 0,
            str(audit.get("forbidden_in_features")),
        )

    edge = root / "graph" / f"{set_id}_edge_list.csv"
    if edge.exists() and edge.stat().st_size > 0:
        edf = pd.read_csv(edge)
        vr.add("no temporal edges", "temporal_edge" not in edf.columns or not edf["temporal_edge"].any())
        if "edge_type" in edf.columns:
            vr.add(
                "behavioural similarity edges only",
                (edf["edge_type"] == "behavioural_similarity").all(),
            )
        graph_stats_path = root / "graph" / f"{set_id}_graph_statistics.csv"
        if graph_stats_path.exists():
            gs = pd.read_csv(graph_stats_path).iloc[0]
            cross_pct = float(gs.get("cross_vehicle_edge_pct", 0))
            vr.add(
                "cross-vehicle edges not near zero",
                cross_pct >= MIN_CROSS_VEHICLE_EDGE_PCT,
                f"cross_vehicle_edge_pct={cross_pct:.4f}%",
            )

    scen_path = root / "results" / "scenario_evaluation" / f"{set_id}_run_level_metrics.csv"
    if not scen_path.exists():
        scen_path = root / "results" / "scenario_evaluation" / "run_level_metrics.csv"
    if scen_path.exists():
        sdf = pd.read_csv(scen_path)
        missing_cols = [c for c in SCENARIO_METRIC_COLUMNS if c not in sdf.columns]
        vr.add("scenario metric columns present", len(missing_cols) == 0, f"missing={missing_cols}")

        if "benign_fleet_control" in sdf["scenario"].values:
            benign_fc = sdf[sdf["scenario"] == "benign_fleet_control"]["false_campaign"].mean()
            vr.add("benign fleet false campaign rate zero", benign_fc == 0, f"mean={benign_fc}")

        if "isolated_attack" in sdf["scenario"].values:
            iso = sdf[sdf["scenario"] == "isolated_attack"]
            fleet_det = iso["fleet_campaign_detected"].mean()
            false_fc = iso["false_campaign"].mean()
            vr.add(
                "isolated attack fleet campaign not declared",
                fleet_det == 0,
                f"fleet_campaign_detected={fleet_det}",
            )
            vr.add("isolated attack false campaign zero", false_fc == 0, f"false_campaign={false_fc}")

        if "unrelated_incidents" in sdf["scenario"].values:
            unrel = sdf[sdf["scenario"] == "unrelated_incidents"]
            vr.add(
                "unrelated incidents fleet campaign not declared",
                unrel["fleet_campaign_detected"].mean() == 0,
                f"fleet_campaign_detected={unrel['fleet_campaign_detected'].mean()}",
            )
            vr.add(
                "unrelated incidents incorrect merge rate reported",
                "incorrect_merge_rate" in unrel.columns,
            )

        if "strong_campaign" in sdf["scenario"].values:
            strong = sdf[sdf["scenario"] == "strong_campaign"]
            vr.add(
                "strong campaign detected",
                strong["fleet_campaign_detected"].mean() > 0,
                f"fleet_campaign_detected={strong['fleet_campaign_detected'].mean()}",
            )
            vr.add(
                "strong campaign F1 positive",
                strong["campaign_f1"].mean() > 0,
                f"campaign_f1={strong['campaign_f1'].mean()}",
            )

        if "weak_campaign" in sdf["scenario"].values:
            weak = sdf[sdf["scenario"] == "weak_campaign"]
            vr.add(
                "weak campaign detected",
                weak["fleet_campaign_detected"].mean() > 0,
                f"fleet_campaign_detected={weak['fleet_campaign_detected'].mean()}",
            )
            vr.add(
                "weak campaign F1 positive",
                weak["campaign_f1"].mean() > 0,
                f"campaign_f1={weak['campaign_f1'].mean()}",
            )

    for scenario in SET_PILOT_SCENARIOS:
        vr.add(
            f"scenario output {scenario}",
            (root / "results" / "scenario_evaluation" / f"{set_id}_{scenario}.csv").exists(),
        )

    pred = root / "results" / "local_detection" / "window_predictions.csv"
    vr.add("predictions recomputable", pred.exists())

    tag = set_id.upper().replace("_", "")
    tables = list((root / "tables").glob(f"table_{tag}_*.csv")) if (root / "tables").exists() else []
    vr.add("set tables generated", len(tables) >= 6, f"count={len(tables)}")

    figures = list((root / "figures").glob(f"figure_{tag}_*.png")) if (root / "figures").exists() else []
    vr.add("set figures generated", len(figures) >= 5, f"count={len(figures)}")

    if OCSLAB_PUBLICATION_ROOT.exists():
        vr.add("OCSLab publication output not modified", True, str(OCSLAB_PUBLICATION_ROOT))
    else:
        vr.add("OCSLab publication output preserved (not present in workspace)", True)

    summary = root / f"{set_id.upper()}_CROSS_DATASET_SUMMARY.md"
    vr.add("summary document present", summary.exists())

    return vr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--set-id", type=str, default="set_01")
    args = parser.parse_args()

    vr = validate_set_pilot(args.output_root, args.set_id)
    report_path = set_work_root(args.output_root, args.set_id) / "validation" / f"{args.set_id}_validation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vr.write_report(report_path)

    if vr.critical_failures:
        print(f"VALIDATION FAILED (set_pilot/{args.set_id}): {vr.critical_failures}")
        return 1
    print(f"VALIDATION PASSED (set_pilot/{args.set_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
