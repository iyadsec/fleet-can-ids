"""Output guard for Phase 4 model diversity experiments."""

from __future__ import annotations

from pathlib import Path

from src.experiments.result_writer import ProtectedOutputGuard


class ModelDiversityGuard(ProtectedOutputGuard):
    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root, "new_experiments/final_validated_runs/model_diversity")

    def ensure_directory_tree(self) -> None:
        for sub in (
            "audit",
            "configs",
            "results/strong/runs",
            "results/weak/runs",
            "tables",
            "figures",
            "logs",
            "manifests",
            "supplementary",
            "validation",
        ):
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)
