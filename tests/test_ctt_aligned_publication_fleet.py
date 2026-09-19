"""Unit/smoke tests for aligned CTT publication fleet path."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.publication_fleet_core import (
    GNN_FEATURE_COLUMNS,
    PublicationFleetConfig,
    compute_cluster_behavioral_cohesion,
    compute_payload_entropy,
    prepare_gnn_fleet_node_matrix,
    run_publication_fleet_pipeline,
)
from src.experiments.local_descriptor_normalisation import fit_benign_fleet_scaler_from_rows
from src.graph.fleet_similarity_features import build_behavior_view_descriptors


def _toy_descriptors(n: int = 40, n_vehicles: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        vid = f"V{i % n_vehicles}"
        fam = i % 2
        rows.append(
            {
                "event_id": f"E{i}",
                "vehicle_id": vid,
                "vehicle_token": vid,
                "vehicle_model": vid,
                "attack_type": "dos" if fam == 0 else "benign",
                "label": 1 if fam == 0 else 0,
                "anomaly_score": 0.8 if fam == 0 else 0.1,
                "frame_count": 100.0,
                "mean_inter_arrival_time": 0.001 + 0.0002 * fam,
                "std_inter_arrival_time": 0.0004 + 0.0001 * fam,
                "can_id_entropy": 2.0 + 0.2 * fam,
                "most_common_can_id_ratio": 0.5,
                **{f"byte_mean_{b}": float(10 + fam * b + rng.normal(0, 0.01)) for b in range(8)},
                "weak_prediction": 1,
                "strong_prediction": int(fam == 0),
                "subset_name": "train_01" if i < n // 4 else "test_01",
            }
        )
    return pd.DataFrame(rows)


def test_gnn_feature_columns_order():
    assert GNN_FEATURE_COLUMNS == (
        "anomaly_score",
        "message_rate",
        "frame_count",
        "burstiness",
        "mean_inter_arrival_time",
        "std_inter_arrival_time",
        "can_id_entropy",
        "most_common_can_id_ratio",
        "payload_entropy",
    )


def test_payload_entropy_and_prepare_matrix():
    df = _toy_descriptors()
    pe = compute_payload_entropy(df)
    assert len(pe) == len(df)
    assert np.all(np.isfinite(pe))
    view = build_behavior_view_descriptors(df)
    assert "burstiness" in view.columns
    assert np.allclose(view["message_rate"], view["frame_count"])
    benign = df[df["label"] == 0]
    scaler = fit_benign_fleet_scaler_from_rows(benign)
    X, _, cols = prepare_gnn_fleet_node_matrix(df, fleet_scaler_provenance=scaler)
    assert X.shape == (len(df), 9)
    assert cols == list(GNN_FEATURE_COLUMNS)


def test_centroid_cohesion_formula():
    X = np.array([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float64)
    mask = np.array([True, True, True])
    c = compute_cluster_behavioral_cohesion(X, mask)
    assert 0.0 <= c <= 1.0 + 1e-6


def test_publication_pipeline_smoke_no_crash():
    df = _toy_descriptors(n=48, n_vehicles=4, seed=3)
    scaler = fit_benign_fleet_scaler_from_rows(df[df["label"] == 0])
    # Short epochs for unit-test speed only; real CTT runs use freeze epochs=30.
    cfg = PublicationFleetConfig(seed=11, gnn_epochs=2)
    art = run_publication_fleet_pipeline(df, fleet_scaler_provenance=scaler, cfg=cfg)
    assert art.embeddings.shape[0] == art.graph_stats["num_nodes"]
    assert art.graph_stats["similarity_threshold"] == 0.95
    assert art.graph_stats["k_same"] == 2
    assert art.graph_stats["k_cross"] == 5
    assert art.graph_stats["edge_weights_in_sageconv"] is False
    if not art.cluster_summary.empty:
        assert "is_qualifying_campaign_cluster" in art.cluster_summary.columns


def test_freeze_defaults():
    cfg = PublicationFleetConfig()
    assert cfg.similarity_threshold == 0.95
    assert cfg.max_same_vehicle_neighbors == 2
    assert cfg.max_cross_vehicle_neighbors == 5
    assert cfg.dbscan_eps == 0.5
    assert cfg.dbscan_min_samples == 2
    assert cfg.minimum_campaign_cohesion == 0.5
    assert cfg.gnn_learning_rate == 0.01
    assert cfg.gnn_weight_decay == 5e-4
    assert cfg.campaign_loss_weight == 0.25
    assert cfg.gnn_epochs == 30
