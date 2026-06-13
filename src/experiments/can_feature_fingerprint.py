"""Vehicle-model fingerprinting check for aggregate CAN-distribution features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.data_splits import is_benign_attack_type
from src.graph.fleet_similarity_features import (
    BEHAVIOR_GRAPH_CANDIDATE_COLUMNS,
    build_behavior_view_descriptors,
)

CAN_AGGREGATE_FEATURES: tuple[str, ...] = (
    "unique_can_id_count",
    "can_id_entropy",
    "most_common_can_id_ratio",
)

BEHAVIOUR_WITHOUT_CAN_AGG: tuple[str, ...] = tuple(
    c for c in BEHAVIOR_GRAPH_CANDIDATE_COLUMNS if c not in CAN_AGGREGATE_FEATURES
)


def _evaluation_rows(descriptors: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Benign rows with split attached (evaluation-only fingerprinting, not fleet inputs)."""
    join_cols = [c for c in ("window_id", "vehicle_model", "source_file") if c in descriptors.columns]
    desc = descriptors.merge(
        manifest[join_cols + ["split"]].drop_duplicates(),
        on=join_cols,
        how="left",
    )
    return desc.copy()


def run_can_feature_fingerprint_ablation(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Compare vehicle_model classification accuracy with vs without CAN aggregate stats.

    Uses benign rows across splits; vehicle_model is evaluation metadata, not a fleet input.
    """
    test = _evaluation_rows(descriptors, manifest)
    test = test[test["attack_type"].map(is_benign_attack_type)]
    test = build_behavior_view_descriptors(test)
    if test.empty or test["vehicle_model"].nunique() < 2:
        return {
            "with_can_agg_accuracy": float("nan"),
            "without_can_agg_accuracy": float("nan"),
            "delta_accuracy": float("nan"),
            "n_samples": int(len(test)),
            "note": "Insufficient benign test rows for fingerprinting check",
        }

    def _score(feature_cols: tuple[str, ...]) -> float:
        cols = [c for c in feature_cols if c in test.columns]
        if not cols:
            return float("nan")
        X = test[cols].astype(np.float64).fillna(0.0).to_numpy()
        y = test["vehicle_model"].astype(str).to_numpy()
        if len(np.unique(y)) < 2:
            return float("nan")
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, random_state=random_state)),
            ]
        )
        cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=random_state)
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        return float(np.mean(scores))

    with_acc = _score(BEHAVIOR_GRAPH_CANDIDATE_COLUMNS)
    without_acc = _score(BEHAVIOUR_WITHOUT_CAN_AGG)
    return {
        "with_can_agg_accuracy": with_acc,
        "without_can_agg_accuracy": without_acc,
        "delta_accuracy": with_acc - without_acc if not (np.isnan(with_acc) or np.isnan(without_acc)) else float("nan"),
        "n_samples": int(len(test)),
        "features_with": list(BEHAVIOR_GRAPH_CANDIDATE_COLUMNS),
        "features_without": list(BEHAVIOUR_WITHOUT_CAN_AGG),
    }


def write_can_fingerprint_report(
    results: dict[str, Any],
    output_path: Path,
) -> Path:
    lines = [
        "# CAN Aggregate Feature Fingerprint Audit",
        "",
        "## Fields reviewed",
        "",
        "- `unique_can_id_count`: count of distinct CAN IDs in the window (aggregate, not raw IDs)",
        "- `can_id_entropy`: Shannon entropy over CAN-ID frequency distribution",
        "- `most_common_can_id_ratio`: fraction of frames using the dominant CAN ID",
        "",
        "## Exposure assessment",
        "",
        "- These fields do **not** expose individual raw CAN identifiers in the fleet feature matrix.",
        "- They summarise traffic-shape statistics that may correlate with OEM platform behaviour.",
        "",
        "## Vehicle-model classification (benign windows, evaluation-only)",
        "",
        f"- Samples: {results.get('n_samples', 0)}",
        f"- Accuracy **with** CAN aggregates: {results.get('with_can_agg_accuracy', float('nan')):.4f}",
        f"- Accuracy **without** CAN aggregates: {results.get('without_can_agg_accuracy', float('nan')):.4f}",
        f"- Delta (with − without): {results.get('delta_accuracy', float('nan')):.4f}",
        "",
        "## Interpretation",
        "",
    ]
    delta = results.get("delta_accuracy", 0.0)
    if isinstance(delta, float) and not np.isnan(delta):
        if delta > 0.05:
            lines.append(
                "- CAN aggregate features materially improve offline vehicle-model fingerprinting; "
                "retain only if behavioural utility outweighs this privacy limitation."
            )
        else:
            lines.append(
                "- CAN aggregate features add limited vehicle-model fingerprinting signal in this check."
            )
    else:
        lines.append(f"- {results.get('note', 'Ablation could not be computed.')}")
    lines.extend(
        [
            "",
            "## Detection utility",
            "",
            "- Retained in behavioural views pending full experiment rerun; ablation of detection impact "
            "is covered by the result regeneration plan.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
