"""Write guard for coordinated_campaign_refinement output root."""

from __future__ import annotations

from pathlib import Path

from src.experiments.result_writer import ProtectedOutputGuard, ProtectedOutputError
from src.utils.paths import resolve_project_root

OUTPUT_REL = Path("new_experiments/final_publication_scenarios/coordinated_campaign_refinement")


class CoordinatedCampaignRefinementGuard(ProtectedOutputGuard):
    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or resolve_project_root()
        super().__init__(root, str(OUTPUT_REL))

    def validate_write_path(self, path: Path | str) -> Path:
        resolved = super().validate_write_path(path)
        try:
            resolved.relative_to(self.output_root)
        except ValueError as exc:
            raise ProtectedOutputError(f"Write blocked outside refinement root: {resolved}") from exc
        return resolved

    def ensure_directory_tree(self) -> Path:
        subs = [
            "audit",
            "configs",
            "validation_scenarios",
            "results/current_all_node_graph",
            "results/candidate_only_graph",
            "results/candidate_only_controlled_connectivity",
            "results/extended_scalability",
            "tables",
            "figures",
            "logs",
            "statistical_analysis",
            "validation",
        ]
        for sub in subs:
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)
        return self.output_root
