"""Shared fleet configuration dataclass."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SharedFleetConfiguration:
    """One frozen fleet-layer configuration for S3 and S4."""

    similarity_threshold: float = 0.90
    max_same_vehicle_neighbors: int = 2
    max_cross_vehicle_neighbors: int = 5
    dbscan_eps: float = 0.8
    dbscan_min_samples: int = 3
    fragment_consolidation_enabled: bool = True
    fragment_centroid_threshold: float = 0.85
    minimum_distinct_vehicles: int = 2
    minimum_cross_vehicle_support: int = 1
    minimum_campaign_cohesion: float = 0.5

    def to_refinement_design(self):
        from src.experiments.coordinated_campaign_refinement.refinement_pipeline import RefinementDesign

        return RefinementDesign(
            design_id="shared",
            candidate_only=True,
            similarity_threshold=self.similarity_threshold,
            top_k_same_vehicle=self.max_same_vehicle_neighbors,
            top_k_cross_vehicle=self.max_cross_vehicle_neighbors,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            consolidate_fragments=self.fragment_consolidation_enabled,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SharedFleetConfiguration:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
