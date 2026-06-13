"""Freeze and hash pipeline inputs from model_diversity_final."""

from __future__ import annotations

import hashlib
from pathlib import Path


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_frozen_pipeline_audit(source_root: Path, output_path: Path) -> dict[str, str]:
    artifacts = {
        "final_split_manifest": source_root / "manifests/final_split_manifest.csv",
        "final_window_split_manifest": source_root / "manifests/final_window_split_manifest.csv",
        "local_model_training_manifest": source_root / "manifests/local_model_training_manifest.csv",
        "scaler_manifest": source_root / "scalers/scaler_manifest.csv",
        "all_descriptors": source_root / "descriptors/all_descriptors.csv",
        "validation_descriptors": source_root / "descriptors/validation_descriptors.csv",
        "provisional_campaign_gate": source_root / "configs/final_campaign_gate.yaml",
        "phase4_config": source_root / "configs/phase4_model_diversity_final.yaml",
    }
    local_models = sorted((source_root / "local_models").glob("*.joblib")) if (source_root / "local_models").exists() else []
    scalers = sorted((source_root / "scalers").glob("*.json")) if (source_root / "scalers").exists() else []

    hashes: dict[str, str] = {}
    lines = [
        "# Frozen pipeline inputs",
        "",
        "Artifacts reused from `model_diversity_final/` without retraining or descriptor regeneration.",
        "",
        "## File hashes",
        "",
        "| Artifact | Path | SHA256 (16) |",
        "|----------|------|-------------|",
    ]
    for name, path in artifacts.items():
        digest = _file_hash(path)
        hashes[name] = digest
        lines.append(f"| {name} | `{path.relative_to(source_root.parent.parent)}` | `{digest}` |")

    lines += ["", "## Local Isolation Forest models", ""]
    for p in local_models:
        digest = _file_hash(p)
        hashes[p.name] = digest
        lines.append(f"- `{p.name}`: `{digest}`")

    lines += ["", "## Fleet scalers", ""]
    for p in scalers:
        digest = _file_hash(p)
        hashes[p.name] = digest
        lines.append(f"- `{p.name}`: `{digest}`")

    lines += [
        "",
        "## Frozen components",
        "",
        "- Balanced source-level split manifest",
        "- Per-platform Isolation Forest models (train benign only)",
        "- Local thresholds from validation split",
        "- Train-only scaler fitting",
        "- Regenerated descriptors and anomaly scores",
        "- Graph construction methodology and DBSCAN parameters",
        "- GraphSAGE checkpoints / architecture",
        "- Random seeds: production `[11, 23, 37, 41, 53, 67, 71, 83, 97, 101]`",
        "- Validation seeds: separate from production (see validation_scenario_manifest.csv)",
        "",
        "## Integrity policy",
        "",
        "No local IDS retraining, descriptor regeneration, or graph-model changes in this tuned run.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return hashes
