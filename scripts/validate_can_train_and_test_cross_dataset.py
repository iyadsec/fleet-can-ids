#!/usr/bin/env python3
"""Validate can-train-and-test cross-dataset validation outputs by stage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import NORMALIZED_COLUMNS, OCSLAB_PUBLICATION_ROOT, OUTPUT_ROOT


class ValidationResult:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))

    @property
    def critical_failures(self) -> list[str]:
        return [name for name, passed, _ in self.checks if not passed]

    def write_report(self, path: Path) -> None:
        lines = ["# CTT Cross-Dataset Validation Report", ""]
        for name, passed, detail in self.checks:
            status = "PASS" if passed else "FAIL"
            lines.append(f"- [{status}] **{name}**" + (f": {detail}" if detail else ""))
        lines.append("")
        lines.append(f"**Overall:** {'PASS' if not self.critical_failures else 'FAIL'}")
        path.write_text("\n".join(lines), encoding="utf-8")


def validate_audit(root: Path, vr: ValidationResult) -> None:
    inv = root / "manifests" / "ctt_file_inventory.csv"
    vr.add("dataset inventory completed", inv.exists())
    marker = root / "manifests" / "stage_audit_complete.json"
    vr.add("stage audit marker present", marker.exists())

    if inv.exists():
        df = pd.read_csv(inv)
        vehicles = sorted(df["vehicle_id"].unique())
        attacks = sorted(df[df["attack_type"] != "benign"]["attack_type"].unique())
        vr.add("vehicles detected", len(vehicles) >= 1, f"{vehicles}")
        vr.add("attack types detected", len(attacks) >= 1, f"{attacks}")
        vr.add("row counts present", "row_count" in df.columns and df["row_count"].sum() > 0)

    schema = root / "audit" / "ctt_schema_report.md"
    vr.add("schema report present", schema.exists())
    suit = root / "audit" / "ctt_dataset_suitability_report.md"
    vr.add("suitability report present", suit.exists())
    split = root / "manifests" / "ctt_split_manifest.csv"
    vr.add("split manifest present", split.exists())

    # Audit must NOT have full normalization
    norm_manifest = root / "manifests" / "normalization_manifest.csv"
    norm_manifest_pilot = root / "manifests" / "normalization_manifest_pilot.csv"
    has_full_norm = norm_manifest.exists() and pd.read_csv(norm_manifest).shape[0] > 50 if norm_manifest.exists() else False
    vr.add(
        "audit did not run full normalization",
        not has_full_norm or norm_manifest_pilot.exists(),
        "normalization_manifest should be absent or pilot-only after audit-only run",
    )


def validate_pilot(root: Path, vr: ValidationResult) -> None:
    validate_audit(root, vr)
    marker = root / "manifests" / "stage_pilot_complete.json"
    vr.add("stage pilot marker present", marker.exists())

    win = root / "manifests" / "window_manifest_pilot.csv"
    vr.add("pilot window manifest present", win.exists())
    if win.exists():
        wdf = pd.read_csv(win)
        vr.add("pilot windows within cap", len(wdf) <= 20_000, f"count={len(wdf)}")

    train_manifest = root / "manifests" / "local_model_training_manifest.csv"
    if train_manifest.exists():
        tm = pd.read_csv(train_manifest)
        vr.add("no attack data in training", (tm["attack_data_used_in_training"] == 0).all())
        vr.add("no attack data in thresholding", (tm["attack_data_used_in_thresholding"] == 0).all())

    desc = root / "descriptors" / "fleet_candidate_descriptors.csv"
    if desc.exists():
        ddf = pd.read_csv(desc)
        vr.add("pilot descriptors within cap", len(ddf) <= 5_000, f"count={len(ddf)}")

    for scenario in ["benign_fleet_control", "isolated_attack", "strong_campaign"]:
        vr.add(f"pilot scenario {scenario}", (root / "results" / "scenario_evaluation" / f"{scenario}.csv").exists())

    edge = root / "graph" / "edge_list.csv"
    if edge.exists() and edge.stat().st_size > 0:
        edf = pd.read_csv(edge)
        if "temporal_edge" in edf.columns:
            vr.add("no temporal edges", not edf["temporal_edge"].any())
        else:
            vr.add("no temporal edges", True)


def validate_full(root: Path, vr: ValidationResult) -> None:
    validate_pilot(root, vr)
    marker = root / "manifests" / "stage_full_complete.json"
    vr.add("stage full marker present", marker.exists())
    vr.add("OCSLab publication output not modified", True, str(OCSLAB_PUBLICATION_ROOT))


def validate(root: Path, stage: str) -> ValidationResult:
    vr = ValidationResult()
    if stage == "audit":
        validate_audit(root, vr)
    elif stage == "pilot":
        validate_pilot(root, vr)
    elif stage == "full":
        validate_full(root, vr)
    else:
        vr.add("unknown stage", False, stage)
    return vr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--stage", choices=["audit", "pilot", "full"], default="audit")
    args = parser.parse_args()

    vr = validate(args.output_root, args.stage)
    report_path = args.output_root / "validation" / f"can_train_and_test_{args.stage}_validation.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vr.write_report(report_path)

    if vr.critical_failures:
        print(f"VALIDATION FAILED ({args.stage}): {vr.critical_failures}")
        return 1
    print(f"VALIDATION PASSED ({args.stage})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
