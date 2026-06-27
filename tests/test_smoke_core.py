"""Tests that do not require PyTorch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.descriptor_generator import behavioural_vector_to_json
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.features.window_generator import generate_windows


def _synthetic_frames(n: int = 250) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        rows.append(
            {
                "timestamp": float(i) * 0.01,
                "can_id": int(rng.integers(0x100, 0x200)),
                "dlc": 8,
                **{f"byte{j}": int(rng.integers(0, 256)) for j in range(8)},
                "label": 0,
                "attack_type": "benign",
                "vehicle_model": "Hyundai",
                "source_file": "synthetic.csv",
            }
        )
    return pd.DataFrame(rows)


def test_window_generation_default_size():
    frames = _synthetic_frames(250)
    windows = generate_windows(frames, window_size=100, stride=50)
    assert not windows.empty
    assert (windows["n_frames"] <= 100).all()


def test_feature_columns_present():
    assert len(BEHAVIOURAL_FEATURE_COLUMNS) >= 10


def test_strong_weak_threshold_logic():
    scores = np.array([0.9, 0.7, 0.3])
    strong_th, weak_th = 0.80, 0.55
    local_alert = (scores >= strong_th).astype(int)
    weak_signal = ((scores >= weak_th) & (scores < strong_th)).astype(int)
    assert local_alert[0] == 1
    assert weak_signal[1] == 1


def test_cosine_similarity():
    x = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sim = cosine_similarity(x)
    assert sim[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_descriptor_vector_json_roundtrip():
    row = pd.Series({c: float(i) for i, c in enumerate(BEHAVIOURAL_FEATURE_COLUMNS)})
    arr = json.loads(behavioural_vector_to_json(row))
    assert len(arr) == len(BEHAVIOURAL_FEATURE_COLUMNS)
