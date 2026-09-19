"""Output guard for balanced E2E publication run."""

from __future__ import annotations

from pathlib import Path

from src.experiments.result_writer import ProtectedOutputGuard, ProtectedOutputError
from src.utils.paths import resolve_project_root

OUTPUT_REL = Path("new_experiments/final_end_to_end_publication_run_balanced")

FORBIDDEN_READ_PREFIXES = (
    "new_experiments/final_validated_runs/",
    "new_experiments/final_publication_scenarios/",
    "new_experiments/final_end_to_end_publication_run/descriptors/",
    "new_experiments/final_end_to_end_publication_run/models/",
    "new_experiments/final_end_to_end_publication_run/results/",
    "new_experiments/results/",
    "archive/",
    "data/processed/anomaly_descriptors.csv",
)


class BalancedPublicationGuard(ProtectedOutputGuard):
    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or resolve_project_root()
        super().__init__(root, str(OUTPUT_REL))

    def validate_write_path(self, path: Path | str) -> Path:
        resolved = super().validate_write_path(path)
        try:
            resolved.relative_to(self.output_root)
        except ValueError as exc:
            raise ProtectedOutputError(f"Write blocked outside balanced root: {resolved}") from exc
        return resolved

    def ensure_directory_tree(self) -> Path:
        subs = [
            "audit", "configs", "manifests", "models", "scalers", "descriptors", "processed",
            "validation_scenarios", "results/vehicle_level", "results/descriptor_analysis",
            "results/scenario_evaluation", "results/campaign_size", "results/edge_sensitivity",
            "results/cost_analysis", "tables", "figures", "statistics", "logs", "validation",
        ]
        for sub in subs:
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)
        return self.output_root
