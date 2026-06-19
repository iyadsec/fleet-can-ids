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


def evaluate_campaign(
    cluster_df: pd.DataFrame,
    scenario_type: str,
    ground_truth_campaign_vehicles: set[str] | None = None,
    ground_truth_attack_family: str | None = None,
) -> dict:
    """Evaluate campaign detection for a scenario."""
    clusters = cluster_df[cluster_df["cluster_id"] >= 0].groupby("cluster_id")
    campaign_detected = False
    best_cluster = -1
    best_score = 0.0

    for cid, group in clusters:
        vehicles = set(group["vehicle_id"].unique())
        n_vehicles = len(vehicles)
        if n_vehicles < 2:
            continue
        attack_families = set(group["attack_type"].unique()) - {"benign"}
        cohesion = 1.0 / len(attack_families) if attack_families else 0.0
        score = n_vehicles * cohesion
        if score > best_score:
            best_score = score
            best_cluster = cid
            campaign_detected = True

    false_campaign = False
    if scenario_type == "benign_fleet_control" and campaign_detected:
        false_campaign = True
    if scenario_type == "isolated_attack" and campaign_detected:
        # Campaign with >1 vehicle is false
        if best_cluster >= 0:
            cv = cluster_df[cluster_df["cluster_id"] == best_cluster]["vehicle_id"].nunique()
            false_campaign = cv > 1

    precision = recall = f1 = 0.0
    if ground_truth_campaign_vehicles and campaign_detected and best_cluster >= 0:
        detected = set(cluster_df[cluster_df["cluster_id"] == best_cluster]["vehicle_id"])
        tp = len(detected & ground_truth_campaign_vehicles)
        prec = safe_div(tp, len(detected))
        rec = safe_div(tp, len(ground_truth_campaign_vehicles))
        precision, recall = prec, rec
        f1 = safe_div(2 * prec * rec, prec + rec)

    return {
        "scenario_type": scenario_type,
        "campaign_detected": int(campaign_detected),
        "false_campaign": int(false_campaign),
        "campaign_precision": precision,
        "campaign_recall": recall,
        "campaign_f1": f1,
        "best_cluster": best_cluster,
        "n_clusters": int(cluster_df["cluster_id"].nunique()),
        "fragmentation": int((cluster_df["cluster_id"] >= 0).sum() - cluster_df["cluster_id"].nunique()),
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
