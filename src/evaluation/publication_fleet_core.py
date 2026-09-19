"""Shared FLEET-GUARD publication fleet core (OCSLab freeze + recovered peer path).

This module is the single fleet-correlation implementation for both OCSLab-style
callers and CTT cross-dataset validation. Parameters default to the balanced
publication freeze:

  recovered_publication_pipeline/.../final_shared_fleet_configuration.yaml
  master_config_hash: 72dbfc1760126e799a8a1670f1aa81391bd69a65edeab7beecb61846aa946d9e

Peer code sources (not invented):
  - GNN_FEATURE_COLUMNS / prepare / cohesion: recovered final_gnn_fleet_decision_experiment
  - GraphSAGE train: src.models.gnn_models.train_graphsage_fleet_correlation
  - Graph edges: src.graph.fleet_graph_builder.build_cross_vehicle_constrained_knn_edges
  - DBSCAN: src.evaluation.campaign_clustering.run_dbscan (euclidean PCA path)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.evaluation.campaign_clustering import extend_dbscan_labels, run_dbscan, subsample_indices
from src.experiments.local_descriptor_normalisation import (
    FleetScalerProvenance,
    apply_fleet_scaler,
)
from src.experiments.vehicle_identity import resolve_graph_vehicle_column
from src.graph.fleet_graph_builder import build_cross_vehicle_constrained_knn_edges
from src.graph.fleet_similarity_features import build_behavior_view_descriptors
from src.models.gnn_models import train_graphsage_fleet_correlation
from src.utils.logging import get_logger

logger = get_logger(__name__)

PUBLICATION_MASTER_CONFIG_HASH = (
    "72dbfc1760126e799a8a1670f1aa81391bd69a65edeab7beecb61846aa946d9e"
)

# Exact 9-D GraphSAGE input order from recovered final_gnn_fleet_decision_experiment.
GNN_FEATURE_COLUMNS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class PublicationFleetConfig:
    """Frozen balanced-publication fleet configuration (+ verified peer training defaults)."""

    # Graph construction (FREEZE)
    similarity_threshold: float = 0.95
    max_same_vehicle_neighbors: int = 2
    max_cross_vehicle_neighbors: int = 5

    # GraphSAGE (FREEZE + peer CODE)
    gnn_hidden_channels: int = 64
    gnn_embedding_dim: int = 32
    gnn_epochs: int = 30
    gnn_learning_rate: float = 0.01
    gnn_weight_decay: float = 5e-4  # peer CODE; not in freeze YAML
    campaign_loss_weight: float = 0.25  # peer CODE λ; not in freeze YAML
    gnn_supervision: Literal["structure"] = "structure"
    gnn_train_ratio: float = 0.7
    gnn_val_ratio: float = 0.15

    # Clustering (FREEZE eps/min_samples; peer CODE for PCA/metric)
    dbscan_eps: float = 0.5
    dbscan_min_samples: int = 2
    dbscan_pca_components: int = 8  # peer CODE; not in freeze YAML
    max_clustering_samples: int = 20000

    # Campaign gate (FREEZE). η for |C_k|: peer campaign_detection maps
    # min_cluster_size to dbscan_min_samples when wiring HDBSCAN; freeze has no
    # separate cluster-size key, so η := dbscan_min_samples (=2).
    minimum_distinct_vehicles: int = 2  # γ (named freeze key)
    minimum_cluster_size: int = 2  # η := freeze dbscan_min_samples
    minimum_cross_vehicle_support: int = 1  # freeze key; enforced as r_k>=2 already covers
    minimum_campaign_cohesion: float = 0.5  # β
    fragment_consolidation_enabled: bool = True
    fragment_centroid_threshold: float = 0.85

    seed: int = 42
    master_config_hash: str = PUBLICATION_MASTER_CONFIG_HASH

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class FleetRunArtifacts:
    embeddings: np.ndarray
    campaign_scores: np.ndarray
    cluster_labels: np.ndarray
    cluster_summary: pd.DataFrame
    node_decisions: pd.DataFrame
    graph_stats: dict[str, Any]
    train_metrics: dict[str, Any]
    behavior_features: np.ndarray
    gnn_feature_names: list[str]
    config: PublicationFleetConfig
    scaler_id: str


def compute_payload_entropy(descriptors: pd.DataFrame) -> np.ndarray:
    """Entropy of abs(byte_mean_*) row-normalised — recovered final_gnn path."""
    mean_cols = [c for c in descriptors.columns if c.startswith("byte_mean_")]
    if not mean_cols:
        return np.zeros(len(descriptors), dtype=np.float64)
    vals = np.abs(descriptors[mean_cols].to_numpy(dtype=np.float64)) + 1e-9
    probs = vals / vals.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def prepare_gnn_fleet_node_matrix(
    descriptors: pd.DataFrame,
    *,
    fleet_scaler_provenance: FleetScalerProvenance,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Behaviour-only 9-D GNN node features with benign-training local normalisation."""
    df = build_behavior_view_descriptors(descriptors.copy())
    df["payload_entropy"] = compute_payload_entropy(df)
    missing = [c for c in GNN_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing GNN feature columns: {missing}")
    df = apply_fleet_scaler(df, fleet_scaler_provenance, feature_names=GNN_FEATURE_COLUMNS)
    X = np.nan_to_num(df[list(GNN_FEATURE_COLUMNS)].to_numpy(dtype=np.float32), nan=0.0)
    return X, df, list(GNN_FEATURE_COLUMNS)


def compute_cluster_behavioral_cohesion(
    behavior_features: np.ndarray,
    mask: np.ndarray,
    *,
    max_samples: int = 500,
    seed: int = 42,
) -> float:
    """
    Centroid cohesion: c_k = (1/|C_k|) sum_i d_hat_i^T mu_hat_k

    L2-normalised behaviour features; mean cosine to cluster centroid.
    """
    Xi = np.asarray(behavior_features[mask], dtype=np.float64)
    if Xi.shape[0] < 2:
        return 1.0
    norms = np.linalg.norm(Xi, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    Xn = Xi / norms
    if Xi.shape[0] > max_samples:
        rng = np.random.default_rng(seed)
        pick = rng.choice(Xi.shape[0], size=max_samples, replace=False)
        Xn = Xn[pick]
    centroid = Xn.mean(axis=0)
    cn = float(np.linalg.norm(centroid))
    if cn < 1e-9:
        return 0.0
    centroid /= cn
    return float(np.mean(Xn @ centroid))


def _ensure_operational_columns(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Attach columns required by the shared graph/PyG builders without using GT as features."""
    df = descriptors.copy()
    if "vehicle_token" not in df.columns:
        veh = resolve_graph_vehicle_column(df)
        df["vehicle_token"] = df[veh].astype(str)
    if "vehicle_model" not in df.columns:
        df["vehicle_model"] = df["vehicle_token"]
    if "local_alert" not in df.columns:
        if "strong_prediction" in df.columns:
            df["local_alert"] = df["strong_prediction"].fillna(0).astype(int)
        else:
            df["local_alert"] = 0
    if "weak_signal" not in df.columns:
        if "weak_prediction" in df.columns:
            df["weak_signal"] = df["weak_prediction"].fillna(1).astype(int)
        else:
            df["weak_signal"] = 1
    if "event_id" not in df.columns:
        df["event_id"] = [f"evt_{i}" for i in range(len(df))]
    if "attack_type" not in df.columns:
        df["attack_type"] = "unknown"
    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = 0.0
    return df


def build_publication_fleet_graph(
    descriptors: pd.DataFrame,
    cfg: PublicationFleetConfig,
    *,
    fleet_scaler_provenance: FleetScalerProvenance,
) -> tuple[Data, pd.DataFrame, np.ndarray, list[str], dict[str, Any]]:
    """Build constrained-kNN graph + PyG Data from 9-D normalised GNN features."""
    df = _ensure_operational_columns(descriptors)
    X, feat_df, cols = prepare_gnn_fleet_node_matrix(
        df, fleet_scaler_provenance=fleet_scaler_provenance
    )
    vehicles = feat_df["vehicle_token"].to_numpy()
    _, _, edge_index, edge_weights, sub_idx = build_cross_vehicle_constrained_knn_edges(
        X,
        vehicles,
        top_k_same_vehicle=cfg.max_same_vehicle_neighbors,
        top_k_cross_vehicle=cfg.max_cross_vehicle_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        metric="cosine",
        seed=cfg.seed,
    )
    df_sub = feat_df.iloc[sub_idx].reset_index(drop=True)
    X_sub = X[sub_idx]
    vehicles_sub = vehicles[sub_idx]

    # Bidirectional edge_index for SAGEConv (weights stored but NOT passed to SAGEConv).
    if edge_index.size == 0:
        ei_t = torch.zeros((2, 0), dtype=torch.long)
        ew_t = torch.zeros((0,), dtype=torch.float32)
        same_vehicle_edges = 0
        cross_vehicle_edges = 0
    else:
        src = np.concatenate([edge_index[0], edge_index[1]])
        dst = np.concatenate([edge_index[1], edge_index[0]])
        ei_t = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
        ew = np.concatenate([edge_weights, edge_weights])
        ew_t = torch.tensor(ew, dtype=torch.float32)
        same_vehicle_edges = int(
            sum(
                1
                for u, v in zip(edge_index[0], edge_index[1])
                if vehicles_sub[int(u)] == vehicles_sub[int(v)]
            )
        )
        cross_vehicle_edges = int(edge_index.shape[1] - same_vehicle_edges)

    # Structure training: dummy y (unused); prefer_ground_truth=False path equivalent.
    y = torch.zeros(len(df_sub), dtype=torch.long)
    data = Data(x=torch.tensor(X_sub, dtype=torch.float32), edge_index=ei_t, edge_attr=ew_t, y=y)
    data.num_nodes = int(len(df_sub))
    data.event_ids = df_sub["event_id"].astype(str).tolist()
    data.vehicle_ids = df_sub["vehicle_token"].astype(str).tolist()

    stats = {
        "num_nodes": int(len(df_sub)),
        "num_edges": int(edge_index.shape[1]),
        "same_vehicle_edges": same_vehicle_edges,
        "cross_vehicle_edges": cross_vehicle_edges,
        "similarity_threshold": cfg.similarity_threshold,
        "k_same": cfg.max_same_vehicle_neighbors,
        "k_cross": cfg.max_cross_vehicle_neighbors,
        "edge_weights_in_sageconv": False,
        "gnn_input_dim": int(X_sub.shape[1]),
        "gnn_feature_names": cols,
    }
    return data, df_sub, X_sub, cols, stats


def _merge_fragment_campaigns(
    cluster_summary: pd.DataFrame,
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    threshold: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Merge qualifying campaigns whose embedding centroids have cosine ≥ threshold."""
    if cluster_summary.empty or not bool(cluster_summary["is_qualifying_campaign_cluster"].any()):
        return cluster_summary, cluster_labels

    qual = cluster_summary[cluster_summary["is_qualifying_campaign_cluster"]].copy()
    ids = qual["cluster_id"].astype(int).tolist()
    if len(ids) < 2:
        return cluster_summary, cluster_labels

    cents = []
    for cid in ids:
        mask = cluster_labels == cid
        vec = embeddings[mask].mean(axis=0)
        n = np.linalg.norm(vec)
        cents.append(vec / n if n > 1e-9 else vec)
    C = np.asarray(cents, dtype=np.float64)
    sim = C @ C.T

    parent = {cid: cid for cid in ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, ci in enumerate(ids):
        for j in range(i + 1, len(ids)):
            cj = ids[j]
            if sim[i, j] >= threshold:
                pi, pj = find(ci), find(cj)
                if pi != pj:
                    parent[pj] = pi

    remap = {cid: find(cid) for cid in ids}
    new_labels = cluster_labels.copy()
    for old, new in remap.items():
        new_labels[cluster_labels == old] = new

    # Rebuild summary flags after merge (caller may recompute; keep cluster_id remap).
    out = cluster_summary.copy()
    out["cluster_id"] = out["cluster_id"].map(lambda c: remap.get(int(c), int(c)))
    return out, new_labels


def cluster_and_gate_campaigns(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    campaign_scores: np.ndarray,
    behavior_features: np.ndarray,
    cfg: PublicationFleetConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    """DBSCAN (StandardScaler→PCA→euclidean) + centroid cohesion campaign gate."""
    meta = meta.reset_index(drop=True)
    if "vehicle_model" not in meta.columns:
        meta = meta.copy()
        meta["vehicle_model"] = meta.get("vehicle_token", meta.get("vehicle_id", "V"))

    fit_idx = subsample_indices(meta, cfg.max_clustering_samples, seed=cfg.seed)
    fit_labels, projector = run_dbscan(
        embeddings[fit_idx],
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        pca_components=cfg.dbscan_pca_components,
        random_state=cfg.seed,
    )
    labels = extend_dbscan_labels(
        embeddings, fit_labels, embeddings[fit_idx], projector, eps=cfg.dbscan_eps
    )

    rows: list[dict[str, Any]] = []
    for cid in sorted({int(c) for c in np.unique(labels)}):
        if cid == -1:
            continue
        mask = labels == cid
        size = int(mask.sum())
        n_veh = int(meta.loc[mask, "vehicle_model"].nunique())
        cohesion = compute_cluster_behavioral_cohesion(
            behavior_features, mask, seed=cfg.seed + int(cid)
        )
        # Evaluation-only metadata — NOT used in the gate.
        if "attack_type" in meta.columns and mask.any():
            dom = meta.loc[mask, "attack_type"].mode().iloc[0]
            dom_ratio = float((meta.loc[mask, "attack_type"] == dom).mean())
        else:
            dom, dom_ratio = "unknown", 0.0
        qualifies = bool(
            size >= cfg.minimum_cluster_size
            and n_veh >= cfg.minimum_distinct_vehicles
            and cohesion >= cfg.minimum_campaign_cohesion
        )
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_size": size,
                "vehicles_in_cluster": n_veh,
                "behavioral_cohesion": round(cohesion, 4),
                "mean_campaign_score": round(float(campaign_scores[mask].mean()), 4),
                "eval_dominant_attack_type": dom,
                "eval_attack_type_purity": round(dom_ratio, 4),
                "is_qualifying_campaign_cluster": qualifies,
            }
        )
    summary = pd.DataFrame(rows)
    if cfg.fragment_consolidation_enabled and not summary.empty:
        summary, labels = _merge_fragment_campaigns(
            summary,
            embeddings,
            labels,
            threshold=cfg.fragment_centroid_threshold,
        )
        # Re-aggregate after fragment merge
        if not summary.empty:
            rebuilt: list[dict[str, Any]] = []
            for cid in sorted({int(c) for c in np.unique(labels) if int(c) != -1}):
                mask = labels == cid
                size = int(mask.sum())
                n_veh = int(meta.loc[mask, "vehicle_model"].nunique())
                cohesion = compute_cluster_behavioral_cohesion(
                    behavior_features, mask, seed=cfg.seed + int(cid)
                )
                qualifies = bool(
                    size >= cfg.minimum_cluster_size
                    and n_veh >= cfg.minimum_distinct_vehicles
                    and cohesion >= cfg.minimum_campaign_cohesion
                )
                rebuilt.append(
                    {
                        "cluster_id": int(cid),
                        "cluster_size": size,
                        "vehicles_in_cluster": n_veh,
                        "behavioral_cohesion": round(cohesion, 4),
                        "mean_campaign_score": round(float(campaign_scores[mask].mean()), 4),
                        "is_qualifying_campaign_cluster": qualifies,
                    }
                )
            summary = pd.DataFrame(rebuilt)
    return labels, summary


def assign_node_decisions(
    meta: pd.DataFrame,
    cluster_labels: np.ndarray,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Mark nodes belonging to qualifying campaign clusters (no attack-type rule)."""
    qualifying = set()
    if not cluster_summary.empty:
        qualifying = {
            int(r["cluster_id"])
            for _, r in cluster_summary.iterrows()
            if bool(r["is_qualifying_campaign_cluster"])
        }
    out = meta.reset_index(drop=True).copy()
    out["cluster_id"] = cluster_labels.astype(int)
    out["in_qualifying_campaign"] = out["cluster_id"].map(lambda c: int(c) in qualifying)
    out["is_noise"] = out["cluster_id"] < 0
    return out


def run_publication_fleet_pipeline(
    descriptors: pd.DataFrame,
    *,
    fleet_scaler_provenance: FleetScalerProvenance,
    cfg: PublicationFleetConfig | None = None,
) -> FleetRunArtifacts:
    """End-to-end shared fleet path: graph → GraphSAGE structure train → DBSCAN → gate."""
    cfg = cfg or PublicationFleetConfig()
    data, meta, behavior_X, cols, graph_stats = build_publication_fleet_graph(
        descriptors, cfg, fleet_scaler_provenance=fleet_scaler_provenance
    )
    if data.num_nodes == 0:
        empty = FleetRunArtifacts(
            embeddings=np.zeros((0, cfg.gnn_embedding_dim), dtype=np.float32),
            campaign_scores=np.zeros(0, dtype=np.float32),
            cluster_labels=np.zeros(0, dtype=int),
            cluster_summary=pd.DataFrame(),
            node_decisions=meta,
            graph_stats=graph_stats,
            train_metrics={},
            behavior_features=behavior_X,
            gnn_feature_names=cols,
            config=cfg,
            scaler_id=fleet_scaler_provenance.scaler_id,
        )
        return empty

    _, train_metrics, embeddings, campaign_scores = train_graphsage_fleet_correlation(
        data,
        hidden_channels=cfg.gnn_hidden_channels,
        embedding_dim=cfg.gnn_embedding_dim,
        epochs=cfg.gnn_epochs,
        learning_rate=cfg.gnn_learning_rate,
        weight_decay=cfg.gnn_weight_decay,
        train_ratio=cfg.gnn_train_ratio,
        val_ratio=cfg.gnn_val_ratio,
        seed=cfg.seed,
        campaign_loss_weight=cfg.campaign_loss_weight,
        supervision=cfg.gnn_supervision,
        anomaly_feature_index=0,  # anomaly_score is first GNN column
    )
    train_metrics = {
        **train_metrics,
        "learning_rate": cfg.gnn_learning_rate,
        "weight_decay": cfg.gnn_weight_decay,
        "campaign_loss_weight": cfg.campaign_loss_weight,
        "supervision": cfg.gnn_supervision,
        "edge_attr_used_in_sageconv": False,
        "aggregation": "mean",
        "activation": "relu_after_first_sageconv",
        "architecture": f"{len(cols)}->{cfg.gnn_hidden_channels}->{cfg.gnn_embedding_dim}",
    }

    labels, summary = cluster_and_gate_campaigns(
        embeddings, meta, campaign_scores, behavior_X, cfg
    )
    decisions = assign_node_decisions(meta, labels, summary)
    return FleetRunArtifacts(
        embeddings=embeddings,
        campaign_scores=campaign_scores,
        cluster_labels=labels,
        cluster_summary=summary,
        node_decisions=decisions,
        graph_stats=graph_stats,
        train_metrics=train_metrics,
        behavior_features=behavior_X,
        gnn_feature_names=cols,
        config=cfg,
        scaler_id=fleet_scaler_provenance.scaler_id,
    )


def match_predicted_to_gt_campaigns_jaccard(
    predicted_vehicle_sets: list[set[str]],
    gt_vehicle_sets: list[set[str]],
    *,
    jaccard_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """
    Greedy Jaccard≥τ matching (ablation peer; P7/P8 publication matcher UNRECOVERED).

    Returns list of (pred_idx, gt_idx, jaccard).
    """
    pairs: list[tuple[float, int, int]] = []
    for i, pred in enumerate(predicted_vehicle_sets):
        for j, gt in enumerate(gt_vehicle_sets):
            if not pred and not gt:
                continue
            inter = len(pred & gt)
            union = len(pred | gt)
            jac = inter / union if union else 0.0
            if jac >= jaccard_threshold:
                pairs.append((jac, i, j))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for jac, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, jac))
    return matches
