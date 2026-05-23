"""Project root and standard directory paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def resolve_project_root(start: Path | None = None) -> Path:
    """Walk upward from *start* until a directory containing ``configs/`` is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").is_dir():
            return candidate
    return current


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical paths relative to the project root."""

    root: Path
    raw_dir: Path
    processed_dir: Path
    configs_dir: Path
    outputs_dir: Path
    metrics_dir: Path
    figures_dir: Path
    experiments_dir: Path
    notebooks_dir: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> ProjectPaths:
        base = resolve_project_root(root)
        return cls(
            root=base,
            raw_dir=base / "data" / "raw",
            processed_dir=base / "data" / "processed",
            configs_dir=base / "configs",
            outputs_dir=base / "outputs",
            metrics_dir=base / "outputs" / "metrics",
            figures_dir=base / "outputs" / "figures",
            experiments_dir=base / "experiments",
            notebooks_dir=base / "notebooks",
        )
