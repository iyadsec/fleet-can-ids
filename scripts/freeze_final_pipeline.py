#!/usr/bin/env python3
"""Freeze corrected pipeline provenance for final validated runs."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.final_gnn_fleet_decision_experiment import GNN_FEATURE_COLUMNS
from src.experiments.local_descriptor_normalisation import load_scaler_provenance
from src.experiments.result_writer import load_experiment_config
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.graph.fleet_similarity_features import BEHAVIOR_GRAPH_CANDIDATE_COLUMNS
from src.utils.paths import resolve_project_root

OUTPUT_ROOT = "new_experiments/final_validated_runs"
CONFIG_SRC = f"{OUTPUT_ROOT}/configs/final_validated_runs.yaml"
CAMPAIGN_CONFIG_SRC = f"{OUTPUT_ROOT}/configs/final_validated_campaign_analysis.yaml"


def _git_info(project_root: Path) -> dict[str, str]:
    info: dict[str, str] = {}
    for key, cmd in [
        ("commit", ["git", "rev-parse", "HEAD"]),
        ("branch", ["git", "branch", "--show-current"]),
        ("status_porcelain", ["git", "status", "--porcelain"]),
    ]:
        try:
            info[key] = subprocess.check_output(cmd, cwd=project_root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            info[key] = "unknown"
    return info


def _package_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = [
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "torch-geometric",
        "networkx",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "joblib",
    ]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = md.version(name)
        except Exception:
            versions[name] = "not installed"
    return versions


def freeze_pipeline(config_path: str = CONFIG_SRC) -> Path:
    project_root = resolve_project_root()
    out_root = project_root / OUTPUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "provenance").mkdir(parents=True, exist_ok=True)
    (out_root / "configs").mkdir(parents=True, exist_ok=True)

    config = load_experiment_config(config_path)
    git = _git_info(project_root)
    pkgs = _package_versions()

    scaler_path = project_root / config.get("fleet_normalisation", {}).get(
        "scaler_cache", "new_experiments/metadata_correction/manifests/fleet_benign_scaler.json"
    )
    scaler_note = "scaler not found"
    if scaler_path.exists():
        prov = load_scaler_provenance(scaler_path)
        dest = out_root / "manifests" / "fleet_benign_scaler.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(scaler_path, dest)
        scaler_note = (
            f"scaler_id={prov.scaler_id}, split={prov.training_split}, "
            f"fit_rows={prov.fit_row_count}, attack_labels_used={prov.attack_labels_used}"
        )

    for src_name in (Path(CONFIG_SRC).name, Path(CAMPAIGN_CONFIG_SRC).name, "final_validated_quick_test.yaml"):
        src = project_root / OUTPUT_ROOT / "configs" / src_name
        dst = out_root / "configs" / src_name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)

    uncommitted = git.get("status_porcelain", "")
    methodological_changes = [
        line for line in uncommitted.splitlines()
        if line.strip() and any(
            p in line for p in ("src/", "scripts/", "tests/", "new_experiments/final_validated_runs/configs/")
        )
    ]

    lines = [
        "# Final Validated Pipeline Snapshot",
        "",
        f"**Frozen at:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Repository",
        "",
        f"- **Git commit:** `{git.get('commit', 'unknown')}`",
        f"- **Branch:** `{git.get('branch', 'unknown')}`",
        f"- **Uncommitted methodological files:** {len(methodological_changes)}",
        "",
    ]
    if methodological_changes:
        lines.append("### Documented uncommitted changes")
        lines.append("")
        lines.append("The following methodological files differ from HEAD (correction + final pipeline wiring):")
        lines.append("")
        for line in methodological_changes[:40]:
            lines.append(f"- `{line.strip()}`")
        if len(methodological_changes) > 40:
            lines.append(f"- … and {len(methodological_changes) - 40} more")
        lines.append("")

    lines.extend(
        [
            "## Environment",
            "",
            f"- **Python:** `{sys.version.split()[0]}` ({sys.executable})",
            f"- **OS:** `{platform.platform()}`",
            f"- **Processor:** `{platform.processor() or platform.machine()}`",
            "",
            "### Package versions",
            "",
        ]
    )
    for pkg, ver in sorted(pkgs.items()):
        lines.append(f"- `{pkg}`: {ver}")

    graph = config.get("graph", {})
    local = config.get("local_ids", {})
    campaign = config.get("campaign", {})
    gnn = config.get("gnn", {})

    lines.extend(
        [
            "",
            "## Configuration files",
            "",
            f"- `{CONFIG_SRC}`",
            f"- `{CAMPAIGN_CONFIG_SRC}`",
            f"- Fleet scaler: `{scaler_path.relative_to(project_root) if scaler_path.exists() else scaler_path}`",
            "",
            "## Descriptor features",
            "",
            f"- **Local IDS (25 cols):** `{BEHAVIOURAL_FEATURE_COLUMNS}`",
            f"- **Fleet similarity behaviour cols:** `{list(BEHAVIOR_GRAPH_CANDIDATE_COLUMNS)}`",
            f"- **GNN node cols:** `{list(GNN_FEATURE_COLUMNS)}`",
            "",
            "## Normalisation",
            "",
            "- **Method:** Global benign-training z-score (`local_descriptor_normalisation.fit_benign_fleet_scaler`)",
            "- **Grouping:** None by vehicle_model at fleet layer",
            f"- **Benign training only:** yes ({scaler_note})",
            "- **View alias:** `behavior_only_vehicle_normalized` → locally normalised behavioural view",
            "",
            "## Vehicle identity",
            "",
            "- **Production token:** `vehicle_token` (`V_0001`, …)",
            "- **Offline instance ID:** `scenario_vehicle_id` (opaque, same allocator)",
            "- **Evaluation only:** `vehicle_model`, `source_trace`, `source_file`",
            "- **Graph kNN / PyG vehicle_id:** opaque instance tokens only",
            "",
            "## Definitions",
            "",
            "- **Campaign size:** distinct attacked opaque vehicle instances",
            "- **Fleet size:** attacked + benign instances in scenario",
            "- **Vehicle-model diversity:** distinct OEM platforms among attacked members (evaluation metadata)",
            "- **Coordination:** shared behavioural characteristics (not timestamp alignment)",
            "",
            "## Graph construction",
            "",
            f"- similarity: `{graph.get('similarity_metric')}`",
            f"- view: `{graph.get('similarity_feature_view')}`",
            f"- top_k_same_vehicle: `{graph.get('top_k_same_vehicle')}`",
            f"- top_k_cross_vehicle: `{graph.get('top_k_cross_vehicle')}`",
            f"- default threshold: `{graph.get('default_similarity_threshold')}`",
            f"- temporal edges: `{graph.get('use_temporal_edges')}` (must be false)",
            "",
            "## Campaign decision / clustering",
            "",
            f"- DBSCAN eps: `{campaign.get('dbscan_eps')}`",
            f"- DBSCAN min_samples: `{campaign.get('dbscan_min_samples')}`",
            f"- PCA components: `{campaign.get('dbscan_pca_components')}`",
            f"- min cluster size: `{campaign.get('min_cluster_size')}`",
            f"- min behavioural cohesion: `{campaign.get('min_behavioral_cohesion')}`",
            "",
            "## Local IDS thresholds",
            "",
            f"- weak: `{local.get('weak_threshold')}`",
            f"- strong: `{local.get('strong_threshold')}`",
            "",
            "## GNN / FCGNN hyperparameters",
            "",
            f"- epochs: `{gnn.get('epochs')}`",
            f"- hidden_channels: `{gnn.get('hidden_channels')}`",
            f"- embedding_dim: `{gnn.get('embedding_dim')}`",
            f"- learning_rate: `{gnn.get('learning_rate')}`",
            f"- M3 architecture: `{gnn.get('standard_architecture')}`",
            f"- M4 architecture: `{gnn.get('proposed_architecture')}`",
            f"- supervision: `{gnn.get('supervision')}`",
            "",
            "## Seeds",
            "",
            f"`{config.get('general', {}).get('seeds')}`",
            "",
        ]
    )

    snapshot_path = out_root / "provenance" / "pipeline_snapshot.md"
    snapshot_path.write_text("\n".join(lines), encoding="utf-8")

    meta = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "python": sys.version,
        "packages": pkgs,
        "config": config_path,
        "scaler": scaler_note,
        "uncommitted_methodological_count": len(methodological_changes),
    }
    (out_root / "provenance" / "pipeline_snapshot.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return snapshot_path


def main() -> int:
    path = freeze_pipeline()
    print(f"Pipeline frozen → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
