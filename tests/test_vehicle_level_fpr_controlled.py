"""Smoke test for FPR-controlled vehicle-level metric computation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.vehicle_level_evaluation import EvaluationOutputs, run_vehicle_level_evaluation
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS


def _synthetic_features(n_per_vehicle: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    wid = 0
    for vehicle in ("Chevrolet", "Hyundai", "Kia"):
        for i in range(n_per_vehicle):
            label = 1 if i % 5 == 0 else 0
            attack_type = "flooding" if label else "attack_free"
            base = rng.normal(0.0, 1.0, len(BEHAVIOURAL_FEATURE_COLUMNS))
            if label:
                base += 2.5
            row = {col: float(base[j]) for j, col in enumerate(BEHAVIOURAL_FEATURE_COLUMNS)}
            row.update(
                {
                    "window_id": wid,
                    "vehicle_model": vehicle,
                    "attack_type": attack_type,
                    "label": label,
                    "source_file": f"{vehicle.lower()}_{i}.csv",
                    "start_frame_idx": i * 50,
                    "frame_count": 100,
                    "mean_inter_arrival_time": 0.001,
                }
            )
            rows.append(row)
            wid += 1
    return pd.DataFrame(rows)


def test_fpr_controlled_vehicle_level_outputs(tmp_path: Path):
    features = _synthetic_features()
    outputs = EvaluationOutputs(
        results_dir=tmp_path / "results",
        tables_dir=tmp_path / "tables",
        figures_dir=tmp_path / "figures",
    )
    written = run_vehicle_level_evaluation(
        features,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
        n_estimators=50,
        outputs=outputs,
    )
    metrics = pd.read_csv(written["vehicle_level_metrics_fpr_controlled"])
    assert set(metrics["vehicle_model"]) >= {"Chevrolet", "Hyundai", "Kia", "pooled"}
    pooled = metrics[metrics["vehicle_model"] == "pooled"].iloc[0]
    assert pooled["false_positive_rate"] <= 0.10
    assert (tmp_path / "results" / "vehicle_level_metrics_per_vehicle_fpr_controlled.csv").exists()
    assert (tmp_path / "tables" / "table_vehicle_level_per_vehicle_fpr_controlled.tex").exists()
    assert (tmp_path / "results" / "per_vehicle_validation_report.md").exists()
