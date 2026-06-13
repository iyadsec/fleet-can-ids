"""Output guard for corrected Phase 4."""

from __future__ import annotations

from pathlib import Path

from src.experiments.result_writer import ProtectedOutputGuard


class ModelDiversityCorrectedGuard(ProtectedOutputGuard):
    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root, "new_experiments/final_validated_runs/model_diversity_corrected")

    def ensure_directory_tree(self) -> None:
        for sub in (
            "audit", "configs", "manifests", "source_pools",
            "results/strong/runs", "results/weak/runs",
            "tables", "figures", "logs", "validation", "comparison",
        ):
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)
