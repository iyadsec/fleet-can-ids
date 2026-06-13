"""Statistical reporting must not print p=0."""

import numpy as np
import pandas as pd

from src.experiments.evaluation_correction.statistics import format_p_value, run_corrected_statistical_tests


def test_format_p_value_never_zero_string():
    assert format_p_value(0.0) == "p < 0.001"
    assert format_p_value(1e-12) == "p < 0.001"
    assert format_p_value(0.05) == "0.050"


def test_paired_seed_level_tests():
    rng = np.random.default_rng(0)
    rows = []
    for seed in range(10):
        for method in ("fcgnn", "descriptor_clustering", "standard_gnn"):
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "attack_strength": "weak",
                    "campaign_size": 5,
                    "campaign_f1": float(rng.random()),
                    "campaign_detection_rate": float(rng.random()),
                    "fpr": float(rng.random() * 0.5),
                    "recall": float(rng.random()),
                    "f1": float(rng.random()),
                    "vehicle_precision": float(rng.random()),
                    "campaign_precision": float(rng.random()),
                }
            )
    out = run_corrected_statistical_tests(pd.DataFrame(rows))
    assert not out.empty
    assert (out["raw_p_value"] == 0).sum() == 0
