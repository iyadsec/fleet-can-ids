"""M1–M4 method implementations for the controlled reviewer ablation."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple  # Dict used in gate remaps

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from src.evaluation.campaign_clustering import run_dbscan, summarize_clusters
from src.graph.scenario_graph import build_scenario_graph_from_features
from src.models.gnn_models import train_graphsage_fleet_correlation


def set_all_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def edge_signature(edge_index: torch.Tensor) -> Tuple[Tuple[int, int], ...]:
    e = edge_index.detach().cpu().numpy()
    pairs = sorted({(int(min(u, v)), int(max(u, v))) for u, v in zip(e[0], e[1])})
    return tuple(pairs)


def _pyg_from_nx(result: Any, X: np.ndarray, meta: pd.DataFrame) -> Data:
    """Build PyG Data without depending on missing src.experiments helpers."""
    id_list = meta["event_id"].astype(str).tolist()
    id_to_idx = {eid: i for i, eid in enumerate(id_list)}
    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    for u, v, attrs in result.graph.edges(data=True):
        iu, iv = id_to_idx[str(u)], id_to_idx[str(v)]
        w = float(attrs.get("weight", attrs.get("similarity", 1.0)))
        src.extend([iu, iv])
        dst.extend([iv, iu])
        weights.extend([w, w])
    if not src:
        n = len(id_list)
        edge_index = torch.arange(n, dtype=torch.long).repeat(2, 1)
        edge_attr = torch.ones(n, dtype=torch.float32)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float32)

    y = (
        meta["local_alert"].astype(int).to_numpy()
        | meta["weak_signal"].astype(int).to_numpy()
    )
    data = Data(
        x=torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(y, dtype=torch.long),
        local_alert=torch.tensor(meta["local_alert"].to_numpy(), dtype=torch.long),
        weak_signal=torch.tensor(meta["weak_signal"].to_numpy(), dtype=torch.long),
        anomaly_score=torch.tensor(
            meta["anomaly_score"].to_numpy(), dtype=torch.float32
        ),
    )
    data.num_nodes = int(data.x.size(0))
    data.event_ids = id_list
    return data


def build_shared_graph(
    X: np.ndarray,
    vehicles: Sequence[str],
    meta: pd.DataFrame,
    *,
    similarity_threshold: float,
    max_same_vehicle_neighbors: int,
    max_cross_vehicle_neighbors: int,
    seed: int,
) -> Dict[str, Any]:
    result = build_scenario_graph_from_features(
        X,
        np.asarray(list(vehicles)),
        meta,
        similarity_threshold=similarity_threshold,
        max_same_vehicle_neighbors=max_same_vehicle_neighbors,
        max_cross_vehicle_neighbors=max_cross_vehicle_neighbors,
        seed=seed,
        build_pyg=False,
    )
    assert len(result.meta) == len(meta), (
        f"Graph node subset mismatch: {len(result.meta)} vs {len(meta)}"
    )
    assert result.meta["event_id"].tolist() == meta["event_id"].tolist()
    X_use = np.asarray(result.behavior_features, dtype=np.float64)
    pyg = _pyg_from_nx(result, X_use, result.meta)
    return {
        "pyg_data": pyg,
        "edge_index": pyg.edge_index.clone(),
        "edge_sig": edge_signature(pyg.edge_index),
        "stats": result.stats,
        "nx_graph": result.graph,
        "meta": result.meta,
        "X": X_use,
    }


def apply_campaign_gate(
    labels: np.ndarray,
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    *,
    min_vehicles: int,
    cohesion_threshold: float,
    fragment_merge_threshold: float,
) -> np.ndarray:
    """
    Campaign gate = current runnable ``summarize_clusters`` plus frozen fragment merge.

    Accept cluster if not noise, >= min_vehicles distinct vehicle_model values,
    and mean intra-cluster cosine similarity >= cohesion_threshold.
    Then merge accepted campaigns whose embedding centroids have cosine
    similarity >= fragment_merge_threshold.
    """
    _events, summary = summarize_clusters(
        labels,
        embeddings,
        meta,
        algorithm="dbscan",
        similarity_threshold=float(cohesion_threshold),
        min_vehicles=int(min_vehicles),
    )
    suspicious = {
        int(r["cluster_id"])
        for _, r in summary.iterrows()
        if bool(r["is_suspicious_campaign"]) and int(r["cluster_id"]) != -1
    }
    out = np.full(len(labels), -1, dtype=int)
    remap: Dict[int, int] = {}
    nid = 0
    for cid in sorted(suspicious):
        remap[cid] = nid
        out[labels == cid] = nid
        nid += 1

    campaign_ids = sorted(set(out.tolist()) - {-1})
    if len(campaign_ids) < 2:
        return out

    centroids = []
    for cid in campaign_ids:
        emb = embeddings[out == cid]
        c = emb.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
    C = np.stack(centroids)
    sim = C @ C.T
    parent = {cid: cid for cid in campaign_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(campaign_ids):
        for j, b in enumerate(campaign_ids):
            if j <= i:
                continue
            if float(sim[i, j]) >= float(fragment_merge_threshold):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
    final: Dict[int, int] = {}
    nid = 0
    for cid in campaign_ids:
        r = find(cid)
        if r not in final:
            final[r] = nid
            nid += 1
    out2 = np.full_like(out, -1)
    for i, v in enumerate(out):
        if v >= 0:
            out2[i] = final[find(v)]
    return out2


def _cluster_and_gate(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    cfg: Dict[str, Any],
    seed: int,
) -> np.ndarray:
    cl = cfg["clustering"]
    labels, _projector = run_dbscan(
        embeddings,
        eps=float(cl["eps"]),
        min_samples=int(cl["min_samples"]),
        pca_components=int(cl["pca_components"]),
        random_state=int(seed),
    )
    gate = cfg["campaign_gate"]
    return apply_campaign_gate(
        labels,
        embeddings,
        meta,
        min_vehicles=int(gate["min_vehicles"]),
        cohesion_threshold=float(gate["cohesion_threshold"]),
        fragment_merge_threshold=float(gate["fragment_merge_threshold"]),
    )


class GCNFleetCorrelator(nn.Module):
    """2-layer GCN mirror of GraphSAGEFleetCorrelator (same heads / dims)."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.campaign_scorer = nn.Sequential(nn.Linear(embedding_dim, 1), nn.Sigmoid())

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x, edge_index))
        z = self.conv2(h, edge_index)
        logits = self.classifier(z)
        campaign_score = self.campaign_scorer(z).squeeze(-1)
        return z, logits, campaign_score


def train_gcn_structure(
    data: Data,
    *,
    hidden_channels: int,
    embedding_dim: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    campaign_loss_weight: float,
    seed: int,
) -> np.ndarray:
    """Train GCN with the same structure + campaign-score objective as GraphSAGE."""
    set_all_seeds(seed)
    device = torch.device("cpu")
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    if edge_index.numel() == 0:
        n = x.size(0)
        idx = torch.arange(n, dtype=torch.long, device=device)
        edge_index = torch.stack([idx, idx], dim=0)

    model = GCNFleetCorrelator(
        in_channels=int(x.size(1)),
        hidden_channels=hidden_channels,
        embedding_dim=embedding_dim,
        num_classes=2,
    ).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    anom = x[:, 0]
    camp_target = (anom - anom.min()) / (anom.max() - anom.min() + 1e-9)

    best_state = None
    best_metric = float("inf")
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        z, _logits, campaign_score = model(x, edge_index)
        z_norm = F.normalize(z, dim=-1)
        src, dst = edge_index
        link_loss = (1.0 - (z_norm[src] * z_norm[dst]).sum(dim=-1)).mean()
        camp_loss = F.mse_loss(campaign_score, camp_target)
        loss = link_loss + float(campaign_loss_weight) * camp_loss
        loss.backward()
        opt.step()
        metric = float(link_loss.detach().item())
        if metric <= best_metric:
            best_metric = metric
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        emb, _, _ = model(x, edge_index)
    return emb.cpu().numpy().astype(np.float64)


def run_m1_local(frozen: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    del cfg
    meta = frozen["meta"]
    return {
        "method": "M1_local_if",
        "pred_campaign_id": np.full(len(meta), -1, dtype=int),
        "local_alert": meta["local_alert"].to_numpy(dtype=bool),
        "embeddings": None,
        "edge_sig": None,
        "produces_campaigns": False,
    }


def run_m2_descriptor(
    frozen: Dict[str, Any], cfg: Dict[str, Any], seed: int
) -> Dict[str, Any]:
    set_all_seeds(seed)
    pred = _cluster_and_gate(frozen["X"], frozen["meta"], cfg, seed)
    return {
        "method": "M2_descriptor_clustering",
        "pred_campaign_id": pred,
        "local_alert": frozen["meta"]["local_alert"].to_numpy(dtype=bool),
        "embeddings": frozen["X"],
        "edge_sig": None,
        "produces_campaigns": True,
    }


def run_m3_gcn(
    frozen: Dict[str, Any],
    cfg: Dict[str, Any],
    seed: int,
    shared_graph: Dict[str, Any],
) -> Dict[str, Any]:
    set_all_seeds(seed)
    gnn = cfg["gnn"]
    emb = train_gcn_structure(
        shared_graph["pyg_data"],
        hidden_channels=int(gnn["hidden_channels"]),
        embedding_dim=int(gnn["embedding_dim"]),
        epochs=int(gnn["epochs"]),
        learning_rate=float(gnn["learning_rate"]),
        weight_decay=float(gnn["weight_decay"]),
        campaign_loss_weight=float(gnn["campaign_loss_weight"]),
        seed=seed,
    )
    pred = _cluster_and_gate(emb, frozen["meta"], cfg, seed)
    return {
        "method": "M3_gcn",
        "pred_campaign_id": pred,
        "local_alert": frozen["meta"]["local_alert"].to_numpy(dtype=bool),
        "embeddings": emb,
        "edge_sig": shared_graph["edge_sig"],
        "produces_campaigns": True,
    }


def run_m4_graphsage(
    frozen: Dict[str, Any],
    cfg: Dict[str, Any],
    seed: int,
    shared_graph: Dict[str, Any],
) -> Dict[str, Any]:
    set_all_seeds(seed)
    gnn = cfg["gnn"]
    _model, _metrics, emb, _scores = train_graphsage_fleet_correlation(
        shared_graph["pyg_data"],
        hidden_channels=int(gnn["hidden_channels"]),
        embedding_dim=int(gnn["embedding_dim"]),
        epochs=int(gnn["epochs"]),
        learning_rate=float(gnn["learning_rate"]),
        weight_decay=float(gnn["weight_decay"]),
        campaign_loss_weight=float(gnn["campaign_loss_weight"]),
        supervision=str(gnn.get("supervision", "structure")),
        seed=seed,
        device="cpu",
    )
    emb = np.asarray(emb, dtype=np.float64)
    pred = _cluster_and_gate(emb, frozen["meta"], cfg, seed)
    return {
        "method": "M4_graphsage",
        "pred_campaign_id": pred,
        "local_alert": frozen["meta"]["local_alert"].to_numpy(dtype=bool),
        "embeddings": emb,
        "edge_sig": shared_graph["edge_sig"],
        "produces_campaigns": True,
    }
