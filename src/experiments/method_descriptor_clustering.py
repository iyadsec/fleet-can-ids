"""M2 — Descriptor clustering without GNN message passing."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.experiments.experiment_pipeline import run_graph_method
from src.experiments.result_writer import ExperimentRunContext


def run_descriptor_clustering_method(
    ctx: ExperimentRunContext,
    scenario_records: pd.DataFrame,
    membership: pd.DataFrame,
    config: dict[str, Any],
    *,
    seed: int,
    similarity_threshold: float | None = None,
    max_neighbors: int | None = None,
) -> "MethodOutputs":
    from src.experiments.experiment_pipeline import MethodOutputs

    return run_graph_method(
        scenario_records,
        membership,
        config,
        seed,
        "descriptor_clustering",
        similarity_threshold=similarity_threshold,
        max_neighbors=max_neighbors,
    )
