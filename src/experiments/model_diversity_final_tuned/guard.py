"""Output guard — never overwrite prior experiment roots."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import resolve_project_root

OUTPUT_REL = Path("new_experiments/final_validated_runs/model_diversity_final_tuned")
SOURCE_REL = Path("new_experiments/final_validated_runs/model_diversity_final")
FORBIDDEN_PREFIXES = (
    "new_experiments/final_validated_runs/model_diversity/",
    "new_experiments/final_validated_runs/model_diversity_corrected/",
    "new_experiments/final_validated_runs/model_diversity_final/",
)


class ModelDiversityFinalTunedGuard:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or resolve_project_root()
        self.output_root = self.project_root / OUTPUT_REL
        self.source_root = self.project_root / SOURCE_REL

    def ensure_directory_tree(self) -> None:
        for sub in (
            "audit",
            "configs",
            "validation_scenarios",
            "gate_search",
            "results/strong",
            "results/weak",
            "tables",
            "figures",
            "logs",
            "comparison",
            "validation",
        ):
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)

    def validate_write_path(self, path: Path) -> None:
        rel = str(path.resolve().relative_to(self.project_root.resolve()))
        for forbidden in FORBIDDEN_PREFIXES:
            if rel.startswith(forbidden):
                raise PermissionError(f"Refusing to write into protected root: {rel}")
        if not rel.startswith(str(OUTPUT_REL)):
            raise PermissionError(f"Tuned Phase 4 writes must stay under {OUTPUT_REL}")
