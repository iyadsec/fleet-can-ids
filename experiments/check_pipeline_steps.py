#!/usr/bin/env python3
"""Verify each Fleet CAN-IDS pipeline step: script, inputs, and outputs."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config
from src.utils.paths import ProjectPaths

STEP_ORDER = (
    "load_dataset",
    "generate_windows",
    "extract_features",
    "train_vehicle_ids",
    "classify_evidence",
    "generate_descriptors",
    "compare_descriptor_size",
    "build_fleet_graph",
    "train_gnn",
    "cluster_campaigns",
    "final_decision",
    "summarize_research_evidence",
    "generate_research_figures",
)

STEP_SCRIPTS: dict[str, str] = {
    "load_dataset": "experiments/01_load_dataset.py",
    "generate_windows": "experiments/02_generate_windows.py",
    "extract_features": "experiments/03_extract_features.py",
    "train_vehicle_ids": "experiments/04_train_vehicle_ids.py",
    "classify_evidence": "experiments/run_full_pipeline.py",
    "generate_descriptors": "experiments/05_generate_descriptors.py",
    "compare_descriptor_size": "experiments/run_full_pipeline.py",
    "build_fleet_graph": "experiments/06_build_graph.py",
    "train_gnn": "experiments/07_train_gnn.py",
    "cluster_campaigns": "experiments/08_cluster_campaigns.py",
    "final_decision": "experiments/09_final_decision.py",
    "summarize_research_evidence": "experiments/run_full_pipeline.py",
    "generate_research_figures": "experiments/run_full_pipeline.py",
}

CSV_COLUMN_CHECKS: dict[str, list[str]] = {
    "clean_can_data": [
        "timestamp",
        "can_id",
        "dlc",
        "byte0",
        "label",
        "attack_type",
        "vehicle_model",
        "source_file",
    ],
    "window_metadata": ["window_id", "vehicle_model", "attack_type", "label"],
    "window_features": ["window_id", "vehicle_model", "attack_type", "label"],
    "vehicle_results": ["vehicle_model", "model", "accuracy", "precision", "recall", "f1_score", "roc_auc"],
    "vehicle_anomaly_predictions": ["window_id", "vehicle_model", "true_label", "anomaly_score", "local_alert", "weak_signal", "evidence_level"],
    "anomaly_descriptors": ["event_id", "window_id", "vehicle_model", "attack_type", "evidence_level", "local_alert", "weak_signal"],
    "fleet_edges": ["source_event_id", "target_event_id", "source_vehicle", "target_vehicle", "similarity_score", "is_cross_vehicle_edge"],
    "fleet_cluster_results": ["event_id", "window_id", "vehicle_model", "cluster_id", "num_unique_vehicles", "mean_cluster_similarity"],
    "final_detection_outcomes": ["event_id", "final_outcome", "was_upgraded_by_fleet"],
    "raw_vs_descriptor_size": ["total_raw_windows", "total_anomaly_descriptors", "estimated_raw_bytes", "estimated_descriptor_bytes", "compression_ratio", "percentage_reduction"],
    "graph_statistics": ["num_nodes", "num_edges", "similarity_threshold", "max_neighbours_per_node", "num_cross_vehicle_edges", "graph_density", "average_degree", "connected_components"],
    "fleet_value_summary": ["total_events", "total_strong_local_anomalies", "total_weak_suspicious_signals", "total_fleet_level_patterns", "percentage_weak_signals_upgraded"],
    "weak_signal_upgrade_summary": ["total_weak_suspicious_signals", "weak_signals_not_alerted_locally", "weak_signals_upgraded_by_fleet", "upgrade_percentage"],
    "cross_vehicle_cluster_summary": ["cluster_id", "cluster_size", "num_unique_vehicles", "vehicles_in_cluster", "dominant_attack_type", "mean_cluster_similarity"],
}


@dataclass
class CheckResult:
    step: str
    status: str  # PASS | FAIL | WARN
    messages: list[str] = field(default_factory=list)

    def add(self, msg: str, *, fail: bool = False, warn: bool = False) -> None:
        self.messages.append(msg)
        if fail:
            self.status = "FAIL"
        elif warn and self.status == "PASS":
            self.status = "WARN"


def _artifact(root: Path, artifacts: dict[str, str], key: str, default: str) -> Path:
    rel = artifacts.get(key, default)
    p = Path(rel)
    return p if p.is_absolute() else root / p


def _check_file(path: Path, *, min_bytes: int = 1) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing: {path}"
    size = path.stat().st_size
    if size < min_bytes:
        return False, f"empty or too small ({size} B): {path}"
    return True, f"ok ({size:,} B): {path}"


def _check_csv_columns(path: Path, required: list[str]) -> tuple[bool, str]:
    try:
        header = pd.read_csv(path, nrows=0).columns.tolist()
    except Exception as exc:
        return False, f"cannot read CSV header: {exc}"
    missing = [c for c in required if c not in header]
    if missing:
        return False, f"missing columns {missing}"
    return True, f"columns ok ({len(header)} total)"


def _check_script(root: Path, rel_script: str) -> tuple[bool, str]:
    script = root / rel_script
    if not script.exists():
        return False, f"script missing: {script}"
    try:
        subprocess.run(
            [sys.executable, "-m", "py_compile", str(script)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return False, f"syntax error: {exc.stderr.strip() or exc}"
    return True, f"script ok: {rel_script}"


def _check_import(module_path: Path) -> tuple[bool, str]:
    name = module_path.stem.replace(".", "_")
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        return False, "import spec failed"
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        return False, f"import failed: {exc}"
    return True, "imports ok"


def check_prerequisites(root: Path, config: dict) -> CheckResult:
    result = CheckResult(step="prerequisites", status="PASS")

    cfg_path = root / "configs/default.yaml"
    ok, msg = _check_file(cfg_path)
    result.add(msg, fail=not ok)

    data_cfg = config.get("data", {})
    external = Path(data_cfg.get("external_dataset_dir", ""))
    if external.exists():
        result.add(f"external dataset: {external}")
        challenge = external / "In-Vehicle Network Intrusion Detection Challenge"
        if challenge.exists():
            train_dirs = list(challenge.glob("*_train"))
            release_dirs = list(challenge.glob("*_release"))
            result.add(f"  train folders: {len(train_dirs)}, release folders: {len(release_dirs)}")
        else:
            result.add(f"challenge folder missing under {external}", warn=True)
    else:
        result.add(f"external dataset missing: {external}", fail=True)

    raw_dir = root / config.get("paths", {}).get("raw_dir", "data/raw")
    if raw_dir.exists():
        result.add(f"local raw dir: {raw_dir}")
    else:
        result.add(f"local raw dir missing: {raw_dir}", warn=True)

    return result


def check_step(step: str, root: Path, artifacts: dict[str, str]) -> CheckResult:
    result = CheckResult(step=step, status="PASS")
    script_rel = STEP_SCRIPTS.get(step, "")
    if not script_rel:
        result.add("no script mapped", fail=True)
        return result

    ok, msg = _check_script(root, script_rel)
    result.add(msg, fail=not ok)

    if step == "load_dataset":
        out = _artifact(root, artifacts, "clean_can_data", "data/processed/clean_can_data.csv")
        ok, msg = _check_file(out, min_bytes=1000)
        result.add(msg, fail=not ok)
        if ok:
            ok2, msg2 = _check_csv_columns(out, CSV_COLUMN_CHECKS["clean_can_data"])
            result.add(msg2, fail=not ok2)

    elif step == "generate_windows":
        inp = _artifact(root, artifacts, "clean_can_data", "data/processed/clean_can_data.csv")
        out = _artifact(root, artifacts, "window_metadata", "data/processed/window_metadata.csv")
        ok, msg = _check_file(inp)
        result.add(f"input {msg}", fail=not ok)
        ok, msg = _check_file(out, min_bytes=1000)
        result.add(f"output {msg}", fail=not ok)
        if out.exists():
            ok2, msg2 = _check_csv_columns(out, CSV_COLUMN_CHECKS["window_metadata"])
            result.add(msg2, fail=not ok2)

    elif step == "extract_features":
        inp = _artifact(root, artifacts, "window_metadata", "data/processed/window_metadata.csv")
        out = _artifact(root, artifacts, "window_features", "data/processed/window_features.csv")
        ok, msg = _check_file(inp)
        result.add(f"input {msg}", fail=not ok)
        ok, msg = _check_file(out, min_bytes=1000)
        result.add(f"output {msg}", fail=not ok)
        if out.exists():
            ok2, msg2 = _check_csv_columns(out, CSV_COLUMN_CHECKS["window_features"])
            result.add(msg2, fail=not ok2)

    elif step == "train_vehicle_ids":
        inp = _artifact(root, artifacts, "window_features", "data/processed/window_features.csv")
        out = _artifact(
            root,
            artifacts,
            "vehicle_results",
            "outputs/metrics/vehicle_level_self_supervised_results.csv",
        )
        ok, msg = _check_file(inp)
        result.add(f"input {msg}", fail=not ok)
        ok, msg = _check_file(out)
        result.add(f"output {msg}", fail=not ok)
        if out.exists():
            ok2, msg2 = _check_csv_columns(out, CSV_COLUMN_CHECKS["vehicle_results"])
            result.add(msg2, fail=not ok2)
        pred = _artifact(
            root, artifacts, "vehicle_anomaly_predictions", "data/processed/vehicle_anomaly_predictions.csv"
        )
        ok, msg = _check_file(pred)
        result.add(f"prediction output {msg}", fail=not ok)
        model = _artifact(
            root, artifacts, "vehicle_ids_model", "outputs/models/vehicle_isolation_forest.joblib"
        )
        ok, msg = _check_file(model, min_bytes=100)
        result.add(f"model output {msg}", fail=not ok)
        if pred.exists():
            ok2, msg2 = _check_csv_columns(pred, CSV_COLUMN_CHECKS["vehicle_anomaly_predictions"])
            result.add(msg2, fail=not ok2)

    elif step == "classify_evidence":
        pred = _artifact(
            root, artifacts, "vehicle_anomaly_predictions", "data/processed/vehicle_anomaly_predictions.csv"
        )
        ok, msg = _check_file(pred, min_bytes=1000)
        result.add(f"prediction output {msg}", fail=not ok)
        if pred.exists():
            ok2, msg2 = _check_csv_columns(pred, CSV_COLUMN_CHECKS["vehicle_anomaly_predictions"])
            result.add(msg2, fail=not ok2)

    elif step == "generate_descriptors":
        feat = _artifact(root, artifacts, "window_features", "data/processed/window_features.csv")
        pred = _artifact(
            root, artifacts, "vehicle_anomaly_predictions", "data/processed/vehicle_anomaly_predictions.csv"
        )
        out = _artifact(
            root, artifacts, "anomaly_descriptors", "data/processed/anomaly_descriptors.csv"
        )
        for label, path in [("features", feat), ("predictions", pred), ("descriptors", out)]:
            ok, msg = _check_file(path, min_bytes=100 if label != "predictions" else 1000)
            result.add(f"{label} {msg}", fail=not ok)
        if out.exists():
            ok2, msg2 = _check_csv_columns(out, CSV_COLUMN_CHECKS["anomaly_descriptors"])
            result.add(msg2, fail=not ok2)

    elif step == "compare_descriptor_size":
        out = _artifact(root, artifacts, "raw_vs_descriptor_size", "outputs/metrics/raw_vs_descriptor_size.csv")
        fig = _artifact(root, artifacts, "raw_vs_descriptor_size_figure", "outputs/figures/raw_vs_descriptor_size.png")
        for label, path in [("metrics", out), ("figure", fig)]:
            ok, msg = _check_file(path, min_bytes=100)
            result.add(f"output {label}: {msg}", fail=not ok)
        if out.exists():
            ok2, msg2 = _check_csv_columns(out, CSV_COLUMN_CHECKS["raw_vs_descriptor_size"])
            result.add(msg2, fail=not ok2)

    elif step == "build_fleet_graph":
        inp = _artifact(
            root, artifacts, "anomaly_descriptors", "data/processed/anomaly_descriptors.csv"
        )
        pt = _artifact(root, artifacts, "fleet_graph", "data/processed/fleet_graph.pt")
        nodes = _artifact(root, artifacts, "fleet_nodes", "data/processed/fleet_nodes.csv")
        edges = _artifact(root, artifacts, "fleet_edges", "data/processed/fleet_edges.csv")
        graphml = _artifact(root, artifacts, "fleet_graph_graphml", "outputs/fleet_graph.graphml")
        stats = _artifact(root, artifacts, "graph_statistics", "outputs/metrics/graph_statistics.csv")
        ok, msg = _check_file(inp)
        result.add(f"input {msg}", fail=not ok)
        for label, path in [("nodes", nodes), ("edges", edges), ("graph.pt", pt), ("graphml", graphml), ("stats", stats)]:
            ok, msg = _check_file(path, min_bytes=100)
            result.add(f"output {label}: {msg}", fail=not ok)
        if edges.exists():
            ok2, msg2 = _check_csv_columns(edges, CSV_COLUMN_CHECKS["fleet_edges"])
            result.add(msg2, fail=not ok2)
        if stats.exists():
            ok2, msg2 = _check_csv_columns(stats, CSV_COLUMN_CHECKS["graph_statistics"])
            result.add(msg2, fail=not ok2)

    elif step == "train_gnn":
        graph = _artifact(root, artifacts, "fleet_graph", "data/processed/fleet_graph.pt")
        emb = _artifact(root, artifacts, "node_embeddings", "data/processed/node_embeddings.csv")
        metrics = _artifact(root, artifacts, "gnn_metrics", "outputs/metrics/gnn_training_metrics.csv")
        ok, msg = _check_file(graph)
        result.add(f"input {msg}", fail=not ok)
        ok, msg = _check_file(emb, min_bytes=100)
        result.add(f"output embeddings: {msg}", fail=not ok)
        ok, msg = _check_file(metrics, min_bytes=10)
        if not ok:
            result.add(f"output metrics: {msg}", warn=True)

    elif step == "cluster_campaigns":
        desc = _artifact(
            root, artifacts, "anomaly_descriptors", "data/processed/anomaly_descriptors.csv"
        )
        emb = _artifact(root, artifacts, "node_embeddings", "data/processed/node_embeddings.csv")
        out = _artifact(root, artifacts, "fleet_cluster_results", "data/processed/fleet_cluster_results.csv")
        ok, msg = _check_file(desc)
        result.add(f"input descriptors: {msg}", fail=not ok)
        if emb.exists():
            result.add(f"input embeddings: ok ({emb.stat().st_size:,} B)")
        else:
            result.add("input embeddings: missing (clustering may use behavioural fallback)", warn=True)
        ok, msg = _check_file(out, min_bytes=1000)
        result.add(f"output {msg}", fail=not ok)
        if out.exists():
            ok2, msg2 = _check_csv_columns(out, CSV_COLUMN_CHECKS["fleet_cluster_results"])
            result.add(msg2, fail=not ok2)

    elif step == "final_decision":
        inp = _artifact(root, artifacts, "fleet_cluster_results", "data/processed/fleet_cluster_results.csv")
        out = _artifact(root, artifacts, "final_detection_outcomes", "outputs/metrics/final_detection_outcomes.csv")
        summary = _artifact(root, artifacts, "final_outcome_summary", "outputs/metrics/final_outcome_summary.csv")
        ok, msg = _check_file(inp)
        result.add(f"input {msg}", fail=not ok)
        ok, msg = _check_file(out, min_bytes=100)
        result.add(f"output outcomes: {msg}", fail=not ok)
        ok, msg = _check_file(summary, min_bytes=50)
        result.add(f"output summary: {msg}", fail=not ok)
        if out.exists():
            ok2, msg2 = _check_csv_columns(out, CSV_COLUMN_CHECKS["final_detection_outcomes"])
            result.add(msg2, fail=not ok2)

    elif step == "summarize_research_evidence":
        for key, default, columns in [
            ("fleet_value_summary", "outputs/metrics/fleet_value_summary.csv", "fleet_value_summary"),
            ("weak_signal_upgrade_summary", "outputs/metrics/weak_signal_upgrade_summary.csv", "weak_signal_upgrade_summary"),
            ("cross_vehicle_cluster_summary", "outputs/metrics/cross_vehicle_cluster_summary.csv", "cross_vehicle_cluster_summary"),
        ]:
            path = _artifact(root, artifacts, key, default)
            ok, msg = _check_file(path, min_bytes=50)
            result.add(f"output {key}: {msg}", fail=not ok)
            if path.exists():
                ok2, msg2 = _check_csv_columns(path, CSV_COLUMN_CHECKS[columns])
                result.add(msg2, fail=not ok2)

    elif step == "generate_research_figures":
        figures = [
            root / "outputs/figures/local_vs_fleet_outcomes.png",
            root / "outputs/figures/weak_signal_upgrade_chart.png",
            root / "outputs/figures/cross_vehicle_clusters_by_attack_type.png",
            root / "outputs/figures/fleet_cluster_vehicle_distribution.png",
        ]
        for fig in figures:
            ok, msg = _check_file(fig, min_bytes=100)
            result.add(f"output figure: {msg}", fail=not ok)

    return result


def print_results(results: list[CheckResult]) -> int:
    fails = 0
    warns = 0
    print("\n" + "=" * 60)
    print("Fleet CAN-IDS — Pipeline Step Check")
    print("=" * 60)
    for r in results:
        icon = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}.get(r.status, "?")
        print(f"\n[{icon} {r.status}] {r.step}")
        for line in r.messages:
            print(f"    {line}")
        if r.status == "FAIL":
            fails += 1
        elif r.status == "WARN":
            warns += 1

    print("\n" + "-" * 60)
    print(f"Summary: {len(results)} checks, {fails} failed, {warns} warnings")
    print("-" * 60 + "\n")
    return 1 if fails else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check pipeline steps one by one")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--step",
        choices=["prerequisites", *STEP_ORDER],
        help="Check only this step (default: all)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = _ROOT
    config = load_config(root / args.config)
    artifacts = config.get("pipeline", {}).get("artifacts", {}) or {}

    if args.step:
        if args.step == "prerequisites":
            results = [check_prerequisites(root, config)]
        else:
            results = [check_step(args.step, root, artifacts)]
    else:
        results = [check_prerequisites(root, config)]
        for step in STEP_ORDER:
            results.append(check_step(step, root, artifacts))

    return print_results(results)


if __name__ == "__main__":
    raise SystemExit(main())
