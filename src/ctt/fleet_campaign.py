"""Fleet campaign decision via GraphSAGE embeddings and DBSCAN."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import DBSCAN
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from src.ctt.constants import OUTPUT_ROOT
from src.ctt.descriptors import load_descriptor_vectors
from src.ctt.utils import ensure_dir, safe_div, write_markdown

SCENARIO_METRIC_COLUMNS = [
    "local_or_incident_detected",
    "fleet_campaign_detected",
    "false_campaign",
    "incorrect_merge_rate",
    "campaign_precision",
    "campaign_recall",
    "campaign_f1",
    "membership_precision",
    "membership_recall",
    "membership_f1",
    "fragmentation_rate",
    "benign_contamination_rate",
]


class FleetGraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


def build_pyg_data(desc_df: pd.DataFrame, edge_df: pd.DataFrame) -> Data:
    X, event_ids = load_descriptor_vectors(desc_df)
    X = np.nan_to_num(X, nan=0.0)
    id_to_idx = {eid: i for i, eid in enumerate(event_ids)}

    src, dst = [], []
    for _, e in edge_df.iterrows():
        if e["source"] in id_to_idx and e["target"] in id_to_idx:
            src.append(id_to_idx[e["source"]])
            dst.append(id_to_idx[e["target"]])
            src.append(id_to_idx[e["target"]])
            dst.append(id_to_idx[e["source"]])

    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=torch.tensor(X, dtype=torch.float32), edge_index=edge_index)
    data.event_ids = event_ids
    data.vehicle_ids = desc_df["vehicle_id"].tolist()
    data.labels = desc_df["label"].tolist()
    data.attack_types = desc_df["attack_type"].tolist()
    return data


def train_graphsage(data: Data, epochs: int = 50, lr: float = 1e-3) -> FleetGraphSAGE:
    """Self-supervised: reconstruct node features from embeddings."""
    model = FleetGraphSAGE(data.x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    decoder = nn.Linear(32, data.x.size(1))

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        recon = decoder(z)
        loss = F.mse_loss(recon, data.x)
        loss.backward()
        optimizer.step()
    return model


def get_embeddings(model: FleetGraphSAGE, data: Data) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)
    return z.numpy()


def dbscan_campaign_decision(
    embeddings: np.ndarray,
    event_ids: list[str],
    vehicle_ids: list[str],
    attack_types: list[str],
    labels: list[int],
    eps: float = 0.8,
    min_samples: int = 2,
) -> pd.DataFrame:
    """Cluster embeddings and decide campaigns."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    cluster_labels = clustering.fit_predict(embeddings)

    rows = []
    for i, eid in enumerate(event_ids):
        rows.append(
            {
                "event_id": eid,
                "vehicle_id": vehicle_ids[i],
                "attack_type": attack_types[i],
                "label": labels[i],
                "cluster_id": int(cluster_labels[i]),
                "is_noise": cluster_labels[i] == -1,
            }
        )
    return pd.DataFrame(rows)


def _cluster_has_attack(group: pd.DataFrame, scenario_type: str) -> bool:
    attack_labels = int((group["label"] == 1).sum())
    attack_families = set(group["attack_type"].unique()) - {"benign"}
    if scenario_type == "benign_fleet_control":
        return attack_labels > 0
    return attack_labels > 0 or bool(attack_families)


def _compute_incorrect_merge(cluster_df: pd.DataFrame, best_cluster: int) -> float:
    """Rate of incorrect multi-vehicle merges for unrelated incidents."""
    if best_cluster < 0:
        return 0.0
    group = cluster_df[cluster_df["cluster_id"] == best_cluster]
    if group["vehicle_id"].nunique() < 2:
        return 0.0
    families = set(group["attack_type"].unique()) - {"benign"}
    if len(families) > 1:
        return 1.0
    return 1.0 if group["vehicle_id"].nunique() >= 2 else 0.0


def _compute_membership_metrics(
    cluster_df: pd.DataFrame,
    best_cluster: int,
    ground_truth_campaign_vehicles: set[str] | None,
) -> tuple[float, float, float, float]:
    """Return membership P/R/F1 and benign contamination for the best cluster."""
    if best_cluster < 0:
        return 0.0, 0.0, 0.0, 0.0

    group = cluster_df[cluster_df["cluster_id"] == best_cluster]
    attack_mask = (group["label"] == 1) | (group["attack_type"] != "benign")
    attack_events = int(attack_mask.sum())
    benign_events = int((~attack_mask).sum())
    cluster_size = len(group)

    membership_precision = safe_div(attack_events, cluster_size)
    total_attack = int(((cluster_df["label"] == 1) | (cluster_df["attack_type"] != "benign")).sum())
    membership_recall = safe_div(attack_events, total_attack)
    membership_f1 = safe_div(2 * membership_precision * membership_recall, membership_precision + membership_recall)
    benign_contamination_rate = safe_div(benign_events, cluster_size)

    if ground_truth_campaign_vehicles:
        detected = set(group["vehicle_id"].unique())
        tp = len(detected & ground_truth_campaign_vehicles)
        membership_precision = safe_div(tp, len(detected))
        membership_recall = safe_div(tp, len(ground_truth_campaign_vehicles))
        membership_f1 = safe_div(2 * membership_precision * membership_recall, membership_precision + membership_recall)

    return membership_precision, membership_recall, membership_f1, benign_contamination_rate


def evaluate_campaign(
    cluster_df: pd.DataFrame,
    scenario_type: str,
    ground_truth_campaign_vehicles: set[str] | None = None,
    ground_truth_attack_family: str | None = None,
) -> dict:
    """Evaluate campaign detection for a scenario."""
    clusters = cluster_df[cluster_df["cluster_id"] >= 0].groupby("cluster_id")
    best_cluster = -1
    best_score = 0.0
    best_n_vehicles = 0
    best_has_attack = False

    for cid, group in clusters:
        has_attack = _cluster_has_attack(group, scenario_type)
        if scenario_type == "benign_fleet_control":
            if not has_attack:
                continue
        else:
            if not has_attack:
                continue

        vehicles = set(group["vehicle_id"].unique())
        n_vehicles = len(vehicles)
        attack_families = set(group["attack_type"].unique()) - {"benign"}

        if n_vehicles < 2 and scenario_type in ("strong_campaign", "weak_campaign", "unrelated_incidents"):
            continue
        if n_vehicles < 2 and scenario_type not in ("isolated_attack",):
            if scenario_type == "isolated_attack":
                pass
            elif scenario_type not in ("benign_fleet_control",):
                continue

        cohesion = 1.0 / len(attack_families) if attack_families else 0.0
        score = n_vehicles * cohesion
        attack_labels = int((group["label"] == 1).sum())
        if attack_labels > 0 or attack_families:
            score += (attack_labels + len(attack_families)) * 0.1
        if score > best_score:
            best_score = score
            best_cluster = cid
            best_n_vehicles = n_vehicles
            best_has_attack = has_attack

    fleet_campaign_detected = int(best_cluster >= 0 and best_n_vehicles >= 2 and best_has_attack)

    if scenario_type == "isolated_attack":
        local_or_incident_detected = int(best_cluster >= 0 and best_n_vehicles == 1 and best_has_attack)
    elif scenario_type == "benign_fleet_control":
        local_or_incident_detected = 0
    else:
        local_or_incident_detected = int(best_cluster >= 0 and best_has_attack)

    false_campaign = 0
    if scenario_type == "benign_fleet_control" and fleet_campaign_detected:
        false_campaign = 1
    elif scenario_type == "isolated_attack" and fleet_campaign_detected:
        false_campaign = 1

    incorrect_merge_rate = 0.0
    if scenario_type == "unrelated_incidents":
        incorrect_merge_rate = _compute_incorrect_merge(cluster_df, best_cluster)

    precision = recall = f1 = 0.0
    if ground_truth_campaign_vehicles and fleet_campaign_detected and best_cluster >= 0:
        detected = set(cluster_df[cluster_df["cluster_id"] == best_cluster]["vehicle_id"])
        tp = len(detected & ground_truth_campaign_vehicles)
        precision = safe_div(tp, len(detected))
        recall = safe_div(tp, len(ground_truth_campaign_vehicles))
        f1 = safe_div(2 * precision * recall, precision + recall)

    membership_precision, membership_recall, membership_f1, benign_contamination_rate = _compute_membership_metrics(
        cluster_df, best_cluster, ground_truth_campaign_vehicles
    )

    positive_clusters = cluster_df[cluster_df["cluster_id"] >= 0]["cluster_id"].nunique()
    clustered_nodes = int((cluster_df["cluster_id"] >= 0).sum())
    fragmentation_rate = safe_div(clustered_nodes - positive_clusters, max(clustered_nodes, 1))

    return {
        "scenario_type": scenario_type,
        "local_or_incident_detected": local_or_incident_detected,
        "fleet_campaign_detected": fleet_campaign_detected,
        "false_campaign": false_campaign,
        "incorrect_merge_rate": incorrect_merge_rate,
        "campaign_precision": precision,
        "campaign_recall": recall,
        "campaign_f1": f1,
        "membership_precision": membership_precision,
        "membership_recall": membership_recall,
        "membership_f1": membership_f1,
        "fragmentation_rate": fragmentation_rate,
        "benign_contamination_rate": benign_contamination_rate,
        "best_cluster": best_cluster,
        "n_clusters": int(cluster_df["cluster_id"].nunique()),
    }


def write_fleet_transfer_policy(output_root: Path = OUTPUT_ROOT) -> None:
    write_markdown(
        output_root / "audit" / "fleet_model_transfer_policy.md",
        "Fleet Model Transfer Policy",
        {
            "Decision": "Option B — Cross-dataset framework validation",
            "Rationale": (
                "The OCSLab frozen GraphSAGE is not directly transferred. "
                "A reproducible GraphSAGE is trained on CTT descriptor graphs "
                "for cross-dataset framework validation."
            ),
            "Temporal edges": "None — all edges are behavioural similarity only.",
        },
    )
