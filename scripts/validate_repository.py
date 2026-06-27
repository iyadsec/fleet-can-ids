#!/usr/bin/env python3
"""Validate FLEET-GUARD repository outputs, schemas, and publication artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.features import LOCAL_FEATURE_COLUMNS
from src.utils.paths import resolve_project_root

FORBIDDEN_FEATURE_COLUMNS = {
    "attack_type",
    "label",
    "vehicle_model",
    "source_file",
    "window_id",
    "event_id",
    "ground_truth_label",
    "eval_attack",
    "campaign_label",
}

BUNDLE = "experimental-2026-06-23"
PRIMARY_TABLES = [
    f"{BUNDLE}/01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv",
    f"{BUNDLE}/01_primary_ocslab_balanced/tables/table_P5_descriptor_compactness_and_privacy.csv",
    f"{BUNDLE}/01_primary_ocslab_balanced/tables/table_P6_benign_isolated_unrelated_results.csv",
    f"{BUNDLE}/01_primary_ocslab_balanced/tables/table_P7_strong_campaign_results.csv",
    f"{BUNDLE}/01_primary_ocslab_balanced/tables/table_P8_weak_campaign_results.csv",
    f"{BUNDLE}/01_primary_ocslab_balanced/tables/table_P9_campaign_size_graph_and_cost.csv",
    f"{BUNDLE}/01_primary_ocslab_balanced/tables/table_P10_edge_connectivity_performance.csv",
    f"{BUNDLE}/01_primary_ocslab_balanced/tables/table_P12_primary_statistical_tests.csv",
    f"{BUNDLE}/01_primary_ocslab_balanced/figures/figure_P4_strong_vs_weak_campaign_F1.pdf",
    f"{BUNDLE}/02_baseline_ablation/tables/table_baseline_ablation.csv",
    "OVERLEAF_CROSS_DATASET_ARTIFACTS/ARTIFACT_INDEX.csv",
    "OVERLEAF_CROSS_DATASET_ARTIFACTS/README.md",
]


class Report:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))

    @property
    def ok(self) -> bool:
        return all(p for _, p, _ in self.checks)

    def write(self, path: Path) -> None:
        lines = ["# FLEET-GUARD repository validation", ""]
        for name, passed, detail in self.checks:
            tag = "PASS" if passed else "FAIL"
            lines.append(f"- [{tag}] **{name}**" + (f": {detail}" if detail else ""))
        lines.append("")
        lines.append(f"**Overall:** {'PASS' if self.ok else 'FAIL'}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


def validate_publication_bundle(root: Path, rep: Report) -> None:
    for rel in PRIMARY_TABLES:
        p = root / rel
        rep.add(f"artifact exists: {rel}", p.exists(), str(p) if not p.exists() else "")


def validate_config(root: Path, rep: Report) -> None:
    cfg_path = root / "configs" / "default.yaml"
    rep.add("default config exists", cfg_path.exists())
    if not cfg_path.exists():
        return
    text = cfg_path.read_text(encoding="utf-8")
    rep.add("window size 100", "window_size: 100" in text)
    rep.add("primary model isolation_forest", "primary_model: isolation_forest" in text)
    rep.add("graph cosine similarity", "similarity_metric: cosine" in text)
    rep.add("DBSCAN clustering", "clustering_method: DBSCAN" in text or "method: dbscan" in text)
    rep.add("no OneDrive paths in config", "OneDrive" not in text and "/Users/" not in text)


def validate_feature_schema(root: Path, rep: Report) -> None:
    schema = root / "new_experiments" / "can_train_and_test_cross_dataset_validation" / "manifests" / "local_feature_schema.csv"
    if schema.exists():
        df = pd.read_csv(schema)
        cols = set(df["column"].astype(str)) if "column" in df.columns else set(df.columns)
        leaked = cols & FORBIDDEN_FEATURE_COLUMNS
        rep.add("CTT feature schema has no forbidden label columns", not leaked, str(leaked) if leaked else "")
    else:
        rep.add("CTT local feature schema (optional)", True, "not present — skip if CTT not run")


def validate_graph_outputs(root: Path, rep: Report) -> None:
    edge = root / "new_experiments" / "can_train_and_test_cross_dataset_validation" / "graph" / "edge_list.csv"
    stats = root / "new_experiments" / "can_train_and_test_cross_dataset_validation" / "graph" / "graph_statistics.csv"
    if edge.exists():
        edf = pd.read_csv(edge)
        rep.add("graph edge list non-empty", len(edf) > 0, f"{len(edf)} edges")
        if "source_vehicle" in edf.columns and "target_vehicle" in edf.columns:
            cross = (edf["source_vehicle"] != edf["target_vehicle"]).sum()
            rep.add("cross-vehicle edges reported", cross >= 0, f"cross-vehicle edges={cross}")
        if "temporal_edge" in edf.columns:
            rep.add("no temporal edges in CTT graph", not edf["temporal_edge"].any())
    else:
        rep.add("CTT graph edge list (optional)", True, "not present — run CTT pilot/full")

    if stats.exists():
        sdf = pd.read_csv(stats)
        rep.add("graph statistics present", not sdf.empty)


def validate_campaign_tables(root: Path, rep: Report) -> None:
    p7 = root / BUNDLE / "01_primary_ocslab_balanced" / "tables" / "table_P7_strong_campaign_results.csv"
    if p7.exists():
        df = pd.read_csv(p7)
        required = {"campaign_size", "campaign_f1", "campaign_precision", "campaign_recall"}
        rep.add("P7 table columns", required.issubset(set(df.columns)), str(list(df.columns)))
    else:
        rep.add("P7 strong campaign table", False, "missing from bundle")


def validate_no_private_paths(root: Path, rep: Report) -> None:
    patterns = ["/Users/", "OneDrive"]
    forbidden_in_code = ["CAN-MIRGU", "LLM fleet"]
    hits: list[str] = []
    scan_roots = ["configs", "src", "experiments", "scripts"]
    for rel in scan_roots:
        p = root / rel
        if not p.exists():
            continue
        for f in p.rglob("*"):
            if f.name == "validate_repository.py":
                continue
            if f.suffix in {".py", ".yaml", ".yml", ".sh"} and f.is_file():
                try:
                    text = f.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                for pat in patterns + forbidden_in_code:
                    if pat in text:
                        hits.append(f"{f.relative_to(root)}: {pat}")
    rep.add("no private paths or forbidden names in core code", not hits, "; ".join(hits[:8]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FLEET-GUARD repository outputs")
    parser.add_argument("--report", type=Path, default=Path("outputs/metrics/repository_validation.md"))
    args = parser.parse_args()

    root = resolve_project_root()
    rep = Report()

    validate_config(root, rep)
    validate_publication_bundle(root, rep)
    validate_feature_schema(root, rep)
    validate_graph_outputs(root, rep)
    validate_campaign_tables(root, rep)
    validate_no_private_paths(root, rep)

    report_path = args.report if args.report.is_absolute() else root / args.report
    rep.write(report_path)

    for name, passed, detail in rep.checks:
        tag = "PASS" if passed else "FAIL"
        print(f"[{tag}] {name}" + (f" — {detail}" if detail else ""))
    print(f"\nReport: {report_path}")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
