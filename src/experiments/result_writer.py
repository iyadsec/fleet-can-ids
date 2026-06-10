"""Safe output handling for new_experiments/ — never overwrite protected legacy results."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import resolve_project_root

# Top-level directories that MUST NOT be written by new experiments.
PROTECTED_RELATIVE_DIRS: tuple[str, ...] = (
    "results",
    "figures",
    "tables",
    "paper",
    "models",
)

SCENARIO_SUBDIRS: tuple[str, ...] = (
    "S0_benign_control",
    "S1_isolated",
    "S2_non_coordinated",
    "S3_strong_campaign",
    "S4_weak_campaign",
    "method_ablation",
    "coordination_sensitivity",
    "edge_sensitivity",
)


class ProtectedOutputError(RuntimeError):
    """Raised when a write would touch protected legacy output directories."""


class RunAlreadyExistsError(FileExistsError):
    """Raised when overwrite is disabled and the run directory already exists."""


class ProtectedOutputGuard:
    """Validates that paths stay under new_experiments/."""

    def __init__(self, project_root: Path, output_root: str = "new_experiments") -> None:
        self.project_root = project_root.resolve()
        self.output_root = (self.project_root / output_root).resolve()
        self.protected_roots = tuple(
            (self.project_root / name).resolve() for name in PROTECTED_RELATIVE_DIRS
        )

    def is_under_output_root(self, path: Path | str) -> bool:
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.output_root)
            return True
        except ValueError:
            return False

    def is_protected_legacy_path(self, path: Path | str) -> bool:
        resolved = Path(path).resolve()
        for protected in self.protected_roots:
            if resolved == protected:
                return True
            try:
                resolved.relative_to(protected)
                return True
            except ValueError:
                continue
        return False

    def validate_write_path(self, path: Path | str) -> Path:
        resolved = Path(path).resolve()
        if self.is_protected_legacy_path(resolved) and not self.is_under_output_root(resolved):
            raise ProtectedOutputError(
                f"Refusing to write to protected legacy path: {resolved}. "
                f"All new experiment outputs must be under {self.output_root}"
            )
        if not self.is_under_output_root(resolved):
            raise ProtectedOutputError(
                f"Refusing to write outside new_experiments/: {resolved}"
            )
        return resolved

    def ensure_directory_tree(self) -> None:
        """Create the standard new_experiments/ subtree (idempotent)."""
        for sub in (
            "configs",
            "scenarios",
            "logs",
            "models",
            "embeddings",
            "manifests",
            "validation",
        ):
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)
        for category in ("results", "figures", "tables"):
            for scenario in SCENARIO_SUBDIRS:
                (self.output_root / category / scenario).mkdir(parents=True, exist_ok=True)
            for emb_scenario in SCENARIO_SUBDIRS:
                (self.output_root / "embeddings" / emb_scenario).mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentRunContext:
    """Unique run context with provenance snapshot."""

    run_id: str
    scenario_key: str
    method: str
    seed: int
    campaign_size: int | None
    coordination_strength: float | None
    created_at: str
    output_root: Path
    run_dir: Path
    overwrite: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        guard: ProtectedOutputGuard,
        scenario_key: str,
        method: str,
        seed: int,
        campaign_size: int | None = None,
        coordination_strength: float | None = None,
        overwrite: bool = False,
        suffix: str | None = None,
    ) -> ExperimentRunContext:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        uid = uuid.uuid4().hex[:8]
        parts = [scenario_key, method, f"seed{seed}", timestamp, uid]
        if campaign_size is not None:
            parts.insert(2, f"n{campaign_size}")
        if coordination_strength is not None:
            parts.insert(3, f"cs{coordination_strength:.2f}".replace(".", "p"))
        if suffix:
            parts.append(suffix)
        run_id = "_".join(parts)
        run_dir = guard.output_root / "results" / scenario_key / "runs" / run_id
        guard.validate_write_path(run_dir)
        if run_dir.exists() and not overwrite:
            raise RunAlreadyExistsError(f"Run directory already exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id,
            scenario_key=scenario_key,
            method=method,
            seed=seed,
            campaign_size=campaign_size,
            coordination_strength=coordination_strength,
            created_at=datetime.now(timezone.utc).isoformat(),
            output_root=guard.output_root,
            run_dir=run_dir,
            overwrite=overwrite,
        )

    def write_config_snapshot(self, config: dict[str, Any]) -> Path:
        path = self.run_dir / "config_snapshot.yaml"
        ProtectedOutputGuard(self.output_root.parent, "new_experiments").validate_write_path(path)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        return path

    def write_json(self, name: str, payload: dict[str, Any] | list[Any]) -> Path:
        path = self.run_dir / name
        guard = ProtectedOutputGuard(self.output_root.parent, "new_experiments")
        guard.validate_write_path(path)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def scenario_results_dir(self) -> Path:
        path = self.output_root / "results" / self.scenario_key
        ProtectedOutputGuard(self.output_root.parent, "new_experiments").validate_write_path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_experiment_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = resolve_project_root() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {config_path}")
    return data


def fingerprint_protected_directories(project_root: Path) -> dict[str, str]:
    """Hash file counts and total bytes for protected dirs (validation baseline)."""
    root = project_root.resolve()
    fingerprints: dict[str, str] = {}
    for name in PROTECTED_RELATIVE_DIRS:
        directory = root / name
        if not directory.is_dir():
            fingerprints[name] = "missing"
            continue
        file_count = 0
        total_bytes = 0
        for path in directory.rglob("*"):
            if path.is_file():
                file_count += 1
                total_bytes += path.stat().st_size
        payload = f"{file_count}:{total_bytes}"
        fingerprints[name] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return fingerprints


def save_protection_baseline(guard: ProtectedOutputGuard) -> Path:
    baseline = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project_root": str(guard.project_root),
        "protected_dirs": fingerprint_protected_directories(guard.project_root),
    }
    path = guard.output_root / "validation" / "protected_dirs_baseline.json"
    guard.validate_write_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    return path


def verify_protected_directories_unchanged(
    guard: ProtectedOutputGuard,
    baseline_path: Path | None = None,
) -> tuple[bool, list[str]]:
    baseline_path = baseline_path or (guard.output_root / "validation" / "protected_dirs_baseline.json")
    if not baseline_path.exists():
        return False, [f"Baseline not found: {baseline_path}"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = fingerprint_protected_directories(guard.project_root)
    errors: list[str] = []
    for name, expected in baseline.get("protected_dirs", {}).items():
        if current.get(name) != expected:
            errors.append(
                f"Protected directory '{name}' fingerprint changed: "
                f"baseline={expected}, current={current.get(name)}"
            )
    return len(errors) == 0, errors
