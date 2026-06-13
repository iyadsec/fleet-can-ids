"""Privacy checks: vehicle model and GT labels must not enter model inputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

FORBIDDEN_INPUT_FIELDS = frozenset(
    {
        "vehicle_model",
        "manufacturer",
        "source_file",
        "source_trace",
        "attack_type",
        "ground_truth_campaign_id",
        "ground_truth_malicious",
        "scenario_role",
        "model_diversity",
        "diversity_level",
        "composition_label",
        "ground_truth_label",
    }
)


def check_privacy(
    scenario_df: pd.DataFrame,
    *,
    similarity_columns: list[str] | None = None,
    graph_x_columns: list[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    sim_cols = similarity_columns or list(scenario_df.columns)
    for field in FORBIDDEN_INPUT_FIELDS:
        if field in sim_cols:
            errors.append(f"Forbidden field in similarity input: {field}")
    if graph_x_columns:
        for field in FORBIDDEN_INPUT_FIELDS:
            if field in graph_x_columns:
                errors.append(f"Forbidden field in GraphSAGE data.x: {field}")
    return errors


def privacy_check_row(run_id: str, errors: list[str]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "privacy_passed": len(errors) == 0,
        "n_violations": len(errors),
        "violations": "; ".join(errors[:5]),
    }
