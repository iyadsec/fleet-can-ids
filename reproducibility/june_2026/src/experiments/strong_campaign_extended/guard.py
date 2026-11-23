"""Output guard — writes only under strong_campaign_extended/."""

from __future__ import annotations

from pathlib import Path

from src.experiments.result_writer import ProtectedOutputGuard, ProtectedOutputError
from src.utils.paths import resolve_project_root

OUTPUT_REL = Path("new_experiments/final_publication_scenarios/strong_campaign_extended")


class StrongCampaignExtendedGuard(ProtectedOutputGuard):
    def __init__(self, project_root: Path | None = None) -> None:
        root = project_root or resolve_project_root()
        super().__init__(root, str(OUTPUT_REL))

    def validate_write_path(self, path: Path | str) -> Path:
        resolved = super().validate_write_path(path)
        try:
            resolved.relative_to(self.output_root)
        except ValueError as exc:
            raise ProtectedOutputError(
                f"Write outside strong_campaign_extended blocked: {resolved}"
            ) from exc
        return resolved

    def ensure_directory_tree(self) -> Path:
        subs = [
            "audit",
            "configs",
            "manifests",
            "results/existing_sizes",
            "results/extended_sizes/runs",
            "tables",
            "figures",
            "logs",
            "statistical_analysis",
            "validation",
        ]
        for sub in subs:
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)
        return self.output_root
