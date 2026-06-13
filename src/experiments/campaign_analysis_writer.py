"""Output guard and run context for campaign_analysis experiments."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.experiments.result_writer import ProtectedOutputError, ProtectedOutputGuard
from src.utils.paths import resolve_project_root


class CampaignAnalysisGuard(ProtectedOutputGuard):
    """Restrict writes to campaign-analysis output root (configurable)."""

    def __init__(self, project_root: Path, output_root: str = "new_experiments/campaign_analysis") -> None:
        super().__init__(project_root, output_root)

    def ensure_directory_tree(self) -> None:
        subs = [
            "configs",
            "results/campaign_size",
            "results/model_diversity",
            "tables/campaign_size",
            "tables/model_diversity",
            "figures/campaign_size",
            "figures/model_diversity",
            "logs",
            "manifests",
            "validation",
        ]
        if "final_validated_runs" in str(self.output_root):
            subs.extend(
                [
                    "provenance",
                    "embeddings/campaign_size",
                    "embeddings/model_diversity",
                    "results/campaign_size_corrected",
                    "tables/campaign_size_corrected",
                    "figures/campaign_size_corrected",
                    "logs/campaign_size_corrected",
                    "validation/campaign_size_corrected",
                ]
            )
        for sub in subs:
            (self.output_root / sub).mkdir(parents=True, exist_ok=True)


@dataclass
class CampaignRunContext:
    run_id: str
    experiment: str
    attack_strength: str
    method: str
    seed: int
    campaign_size: int
    model_diversity: int | None
    coordination_strength: float
    created_at: str
    output_root: Path
    run_dir: Path
    overwrite: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        guard: CampaignAnalysisGuard,
        experiment: str,
        attack_strength: str,
        method: str,
        seed: int,
        campaign_size: int,
        coordination_strength: float,
        model_diversity: int | None = None,
        overwrite: bool = False,
    ) -> CampaignRunContext:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        uid = uuid.uuid4().hex[:8]
        parts = [
            experiment,
            attack_strength,
            f"n{campaign_size}",
            method,
            f"seed{seed}",
            timestamp,
            uid,
        ]
        if model_diversity is not None:
            parts.insert(3, f"d{model_diversity}")
        run_id = "_".join(parts)
        run_dir = guard.output_root / "results" / experiment / "runs" / run_id
        guard.validate_write_path(run_dir)
        if run_dir.exists() and not overwrite:
            from src.experiments.result_writer import RunAlreadyExistsError

            raise RunAlreadyExistsError(f"Run exists: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            run_id=run_id,
            experiment=experiment,
            attack_strength=attack_strength,
            method=method,
            seed=seed,
            campaign_size=campaign_size,
            model_diversity=model_diversity,
            coordination_strength=coordination_strength,
            created_at=datetime.now(timezone.utc).isoformat(),
            output_root=guard.output_root,
            run_dir=run_dir,
            overwrite=overwrite,
        )

    def write_config_snapshot(self, config: dict[str, Any]) -> Path:
        path = self.run_dir / "config_snapshot.yaml"
        CampaignAnalysisGuard(self.output_root.parent.parent).validate_write_path(path)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=False)
        return path


def load_campaign_analysis_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = resolve_project_root() / config_path
    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {config_path}")
    return data
