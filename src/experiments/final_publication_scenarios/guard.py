"""Output guard — writes only under final_publication_scenarios/."""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import resolve_project_root

OUTPUT_REL = Path("new_experiments/final_publication_scenarios")
FORBIDDEN_PREFIXES = (
    "new_experiments/final_validated_runs/",
    "new_experiments/results/",
    "new_experiments/publication_ready/",
)


class PublicationScenariosGuard:
    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or resolve_project_root()
        self.output_root = self.project_root / OUTPUT_REL

    def ensure_directory_tree(self) -> None:
        for sub in (
            "audit",
            "configs",
            "manifests",
            "results/scenarios",
            "results/campaign_size",
            "results/edge_sensitivity",
            "tables",
            "figures",
            "logs",
            "statistical_analysis",
            "validation",
        ):
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)

    def validate_write_path(self, path: Path) -> None:
        rel = str(path.resolve().relative_to(self.project_root.resolve()))
        for forbidden in FORBIDDEN_PREFIXES:
            if rel.startswith(forbidden):
                raise PermissionError(f"Refusing to write into protected root: {rel}")
        if not rel.startswith(str(OUTPUT_REL)):
            raise PermissionError(f"Publication writes must stay under {OUTPUT_REL}")
