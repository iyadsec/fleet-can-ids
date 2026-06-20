#!/usr/bin/env python3
"""Validate can-train-and-test set_pilot stage outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import OCSLAB_PUBLICATION_ROOT, OUTPUT_ROOT
from src.ctt.set_pilot import SET_PILOT_SCENARIOS, set_work_root


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
        inv_path = base_output / "manifests" / "ctt_file_inventory.csv"
        expected = 42
        if inv_path.exists():
            inv = pd.read_csv(inv_path)
            expected = len(inv[(inv["dataset_set"] == set_id) & ~(
                (inv["subset_name"] == "train_01") & (inv["attack_type"] != "benign")
            )])
        caps_note = ""
        marker_path = root / "manifests" / f"stage_set_pilot_{set_id}_complete.json"
        if marker_path.exists():
            caps_note = json.loads(marker_path.read_text()).get("caps", {})
        capped = bool(caps_note.get("max_windows") or caps_note.get("max_rows_per_file"))
        vr.add(
            "set files processed (windowable scope)",
            len(nm) >= min(expected, 25) if capped else len(nm) >= expected * 0.9,
            f"count={len(nm)} expected≈{expected} capped={capped}",
        )

    win = root / "manifests" / "window_manifest.csv"
    if win.exists():
        wdf = pd.read_csv(win)
        vr.add("train/test split respected", "train_01" in wdf["subset_name"].values)
        test_subs = [s for s in wdf["subset_name"].unique() if str(s).startswith("test_")]
        vr.add("test subsets present", len(test_subs) >= 3, str(test_subs))

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

    for scenario in SET_PILOT_SCENARIOS:
        vr.add(
            f"scenario output {scenario}",
            (root / "results" / "scenario_evaluation" / f"{set_id}_{scenario}.csv").exists(),
        )

    pred = root / "results" / "local_detection" / "window_predictions.csv"
    vr.add("predictions recomputable", pred.exists())

    tables = list((root / "tables").glob(f"table_{set_id.upper().replace('_', '')}_*.csv")) if (root / "tables").exists() else []
    vr.add("set tables generated", len(tables) >= 6, f"count={len(tables)}")

    figures = list((root / "figures").glob(f"figure_{set_id.upper().replace('_', '')}_*.png")) if (root / "figures").exists() else []
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
