"""Unit/smoke tests for aligned CTT publication fleet path."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src.evaluation.publication_fleet_core import (
    GNN_FEATURE_COLUMNS,
    PublicationFleetConfig,
    _cluster_qualifies_campaign,
    cluster_and_gate_campaigns,
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


def test_eta_required_and_independent_of_dbscan_defaults():
    """η defaults to None; DBSCAN min_samples stays 2; no silent coupling."""
    cfg = PublicationFleetConfig()
    assert cfg.dbscan_eps == 0.5
    assert cfg.dbscan_min_samples == 2
    assert cfg.min_campaign_cluster_size is None
    with pytest.raises(ValueError, match="min_campaign_cluster_size"):
        cfg.require_min_campaign_cluster_size()

    # Changing DBSCAN min_samples must not invent η.
    cfg_db = PublicationFleetConfig(dbscan_min_samples=7)
    assert cfg_db.dbscan_min_samples == 7
    assert cfg_db.min_campaign_cluster_size is None

    # Changing η must not change DBSCAN min_samples.
    cfg_eta = PublicationFleetConfig(min_campaign_cluster_size=11)
    assert cfg_eta.min_campaign_cluster_size == 11
    assert cfg_eta.dbscan_min_samples == 2
    assert cfg_eta.require_min_campaign_cluster_size() == 11


def test_publication_pipeline_requires_explicit_eta():
    df = _toy_descriptors(n=24, n_vehicles=3, seed=1)
    scaler = fit_benign_fleet_scaler_from_rows(df[df["label"] == 0])
    with pytest.raises(ValueError, match="min_campaign_cluster_size"):
        run_publication_fleet_pipeline(
            df,
            fleet_scaler_provenance=scaler,
            cfg=PublicationFleetConfig(seed=11, gnn_epochs=1),
        )


def test_publication_pipeline_smoke_no_crash():
    df = _toy_descriptors(n=48, n_vehicles=4, seed=3)
    scaler = fit_benign_fleet_scaler_from_rows(df[df["label"] == 0])
    # Explicit η for unit-test only — NOT a recovered historical P7/P8 value.
    cfg = PublicationFleetConfig(seed=11, gnn_epochs=2, min_campaign_cluster_size=2)
    art = run_publication_fleet_pipeline(df, fleet_scaler_provenance=scaler, cfg=cfg)
    assert art.embeddings.shape[0] == art.graph_stats["num_nodes"]
    assert art.graph_stats["similarity_threshold"] == 0.95
    assert art.graph_stats["k_same"] == 2
    assert art.graph_stats["k_cross"] == 5
    assert art.graph_stats["edge_weights_in_sageconv"] is False
    assert art.config.dbscan_min_samples == 2
    assert art.config.min_campaign_cluster_size == 2
    if not art.cluster_summary.empty:
        assert "is_qualifying_campaign_cluster" in art.cluster_summary.columns
        assert "gate_eta_min_campaign_cluster_size" in art.cluster_summary.columns
        assert "gate_dbscan_min_samples" in art.cluster_summary.columns
        assert set(art.cluster_summary["gate_eta_min_campaign_cluster_size"]) == {2}
        assert set(art.cluster_summary["gate_dbscan_min_samples"]) == {2}


def test_freeze_defaults():
    cfg = PublicationFleetConfig()
    assert cfg.similarity_threshold == 0.95
    assert cfg.max_same_vehicle_neighbors == 2
    assert cfg.max_cross_vehicle_neighbors == 5
    assert cfg.dbscan_eps == 0.5
    assert cfg.dbscan_min_samples == 2
    assert cfg.min_campaign_cluster_size is None
    assert cfg.minimum_campaign_cohesion == 0.5
    assert cfg.minimum_distinct_vehicles == 2
    assert cfg.minimum_cross_vehicle_support == 1
    assert cfg.fragment_centroid_threshold == 0.85
    assert cfg.gnn_learning_rate == 0.01
    assert cfg.gnn_weight_decay == 5e-4
    assert cfg.campaign_loss_weight == 0.25
    assert cfg.gnn_epochs == 30


def test_gate_uses_eta_not_dbscan_min_samples():
    """Post-clustering gate: size>=η; DBSCAN min_samples does not enter the predicate."""
    cfg = PublicationFleetConfig(dbscan_min_samples=2, min_campaign_cluster_size=5)
    eta = cfg.require_min_campaign_cluster_size()
    assert eta == 5
    # Cluster of size 3 fails η=5 even though size >= dbscan_min_samples.
    assert not _cluster_qualifies_campaign(
        size=3, n_vehicles=2, cohesion=0.9, cfg=cfg, eta=eta
    )
    assert _cluster_qualifies_campaign(
        size=5, n_vehicles=2, cohesion=0.9, cfg=cfg, eta=eta
    )
    # Raising dbscan_min_samples alone does not change the gate when η is fixed.
    cfg2 = PublicationFleetConfig(dbscan_min_samples=99, min_campaign_cluster_size=5)
    assert _cluster_qualifies_campaign(
        size=5, n_vehicles=2, cohesion=0.9, cfg=cfg2, eta=cfg2.require_min_campaign_cluster_size()
    )


def test_dbscan_then_eta_gate_stages_and_no_label_leakage(monkeypatch):
    """DBSCAN runs first; η only in the post-clustering gate; labels unused."""
    n = 12
    # Two tight groups of 6 points each → DBSCAN with min_samples=2 forms clusters.
    embeddings = np.vstack(
        [
            np.zeros((6, 4), dtype=np.float64) + np.array([0.0, 0.0, 0.0, 0.0]),
            np.ones((6, 4), dtype=np.float64) * 10.0,
        ]
    )
    behavior = embeddings.copy()
    meta = pd.DataFrame(
        {
            "vehicle_model": [f"V{i % 3}" for i in range(n)],
            "attack_type": ["dos"] * 6 + ["fuzzing"] * 6,
            "label": [1] * n,
            "gt_campaign_id": ["A"] * 6 + ["B"] * 6,
        }
    )
    scores = np.ones(n, dtype=np.float64)

    call_order: list[str] = []
    import src.evaluation.publication_fleet_core as core

    real_run_dbscan = core.run_dbscan
    real_extend = core.extend_dbscan_labels
    real_gate = core._cluster_qualifies_campaign

    def tracked_dbscan(*args, **kwargs):
        call_order.append("dbscan")
        assert kwargs.get("min_samples") == 2
        return real_run_dbscan(*args, **kwargs)

    def tracked_extend(*args, **kwargs):
        call_order.append("extend")
        return real_extend(*args, **kwargs)

    def tracked_gate(**kwargs):
        call_order.append("gate")
        # Gate must see explicit η, never silently use dbscan_min_samples.
        assert kwargs["eta"] == 4
        assert "attack_type" not in kwargs
        assert "label" not in kwargs
        assert "gt_campaign_id" not in kwargs
        return real_gate(**kwargs)

    monkeypatch.setattr(core, "run_dbscan", tracked_dbscan)
    monkeypatch.setattr(core, "extend_dbscan_labels", tracked_extend)
    monkeypatch.setattr(core, "_cluster_qualifies_campaign", tracked_gate)

    cfg = PublicationFleetConfig(
        dbscan_min_samples=2,
        min_campaign_cluster_size=4,
        fragment_consolidation_enabled=False,
        seed=0,
    )
    labels, summary = cluster_and_gate_campaigns(embeddings, meta, scores, behavior, cfg)

    assert call_order[0] == "dbscan"
    assert "extend" in call_order
    assert "gate" in call_order
    assert call_order.index("dbscan") < call_order.index("gate")
    assert set(summary["gate_eta_min_campaign_cluster_size"]) == {4}
    assert set(summary["gate_dbscan_min_samples"]) == {2}
    assert len(labels) == n

    # Mutating evaluation-only columns must not change qualification under fixed geometry.
    meta2 = meta.copy()
    meta2["attack_type"] = "totally_different"
    meta2["label"] = 0
    meta2["gt_campaign_id"] = "Z"
    _, summary2 = cluster_and_gate_campaigns(embeddings, meta2, scores, behavior, cfg)
    assert list(summary["is_qualifying_campaign_cluster"]) == list(
        summary2["is_qualifying_campaign_cluster"]
    )


def test_changing_eta_does_not_change_dbscan_min_samples_in_gate_rows(monkeypatch):
    # Identical rows → cohesion ≈ 1 so the gate exercises η / vehicles, not cohesion.
    embeddings = np.ones((20, 4), dtype=np.float64)
    behavior = embeddings.copy()
    meta = pd.DataFrame(
        {
            "vehicle_model": [f"V{i % 4}" for i in range(20)],
            "attack_type": ["x"] * 20,
        }
    )
    scores = np.zeros(20)

    import src.evaluation.publication_fleet_core as core

    captured: dict[str, int] = {}

    def fake_dbscan(X, *, eps, min_samples, pca_components, random_state):
        captured["dbscan_min_samples"] = int(min_samples)
        # One cluster of all points.
        return np.zeros(len(X), dtype=int), None

    def fake_extend(embeddings, fit_labels, fit_emb, projector, *, eps):
        return np.zeros(len(embeddings), dtype=int)

    monkeypatch.setattr(core, "run_dbscan", fake_dbscan)
    monkeypatch.setattr(core, "extend_dbscan_labels", fake_extend)

    cfg_a = PublicationFleetConfig(
        dbscan_min_samples=2,
        min_campaign_cluster_size=3,
        fragment_consolidation_enabled=False,
    )
    _, sum_a = cluster_and_gate_campaigns(embeddings, meta, scores, behavior, cfg_a)
    assert captured["dbscan_min_samples"] == 2
    assert set(sum_a["gate_eta_min_campaign_cluster_size"]) == {3}

    cfg_b = PublicationFleetConfig(
        dbscan_min_samples=2,
        min_campaign_cluster_size=15,
        fragment_consolidation_enabled=False,
    )
    _, sum_b = cluster_and_gate_campaigns(embeddings, meta, scores, behavior, cfg_b)
    assert captured["dbscan_min_samples"] == 2
    assert set(sum_b["gate_eta_min_campaign_cluster_size"]) == {15}
    # Same DBSCAN param; different η → different qualification for size=20.
    assert bool(sum_a["is_qualifying_campaign_cluster"].iloc[0]) is True
    assert bool(sum_b["is_qualifying_campaign_cluster"].iloc[0]) is True  # 20 >= 15

    cfg_c = PublicationFleetConfig(
        dbscan_min_samples=2,
        min_campaign_cluster_size=21,
        fragment_consolidation_enabled=False,
    )
    _, sum_c = cluster_and_gate_campaigns(embeddings, meta, scores, behavior, cfg_c)
    assert captured["dbscan_min_samples"] == 2
    assert bool(sum_c["is_qualifying_campaign_cluster"].iloc[0]) is False


def test_cluster_and_gate_source_documents_stage_order():
    src = inspect.getsource(cluster_and_gate_campaigns)
    assert "require_min_campaign_cluster_size" in src
    assert "run_dbscan" in src
    assert src.index("run_dbscan") < src.index("_cluster_qualifies_campaign")
    assert "dbscan_min_samples" not in inspect.getsource(_cluster_qualifies_campaign)
