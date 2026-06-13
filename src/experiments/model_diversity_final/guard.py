"""Output guard for final Phase 4 — never overwrite prior experiment roots."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import resolve_project_root

OUTPUT_REL = Path("new_experiments/final_validated_runs/model_diversity_final")
FORBIDDEN_PREFIXES = (
    "new_experiments/final_validated_runs/model_diversity/",
    "new_experiments/final_validated_runs/model_diversity_corrected/",
)


class ModelDiversityFinalGuard:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or resolve_project_root()
        self.output_root = self.project_root / OUTPUT_REL

    def ensure_directory_tree(self) -> None:
        for sub in (
            "audit",
            "configs",
            "manifests",
            "local_models",
            "scalers",
            "descriptors",
            "source_pools",
            "validation_tuning",
            "results/strong",
            "results/weak",
            "results/controlled_same_attack",
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
                raise PermissionError(f"Refusing to write into archived root: {rel}")
        if not rel.startswith(str(OUTPUT_REL)):
            raise PermissionError(f"Final Phase 4 writes must stay under {OUTPUT_REL}")
