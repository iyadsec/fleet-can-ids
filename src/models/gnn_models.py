"""Graph neural network training on fleet anomaly graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from src.utils.logging import get_logger

logger = get_logger(__name__)

GnnArchitecture = Literal["graphsage", "gcn", "gat"]


class GCNEncoder(nn.Module):
    """Two-layer GCN encoder with node classification head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x, edge_index))
        z = self.conv2(h, edge_index)
        logits = self.classifier(z)
        return z, logits


class GraphSAGEEncoder(nn.Module):
    """Two-layer GraphSAGE encoder with node classification head."""

    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.conv1(x, edge_index))
        z = self.conv2(h, edge_index)
        logits = self.classifier(z)
        return z, logits


class GraphSAGEFleetCorrelator(nn.Module):
    """GraphSAGE encoder with attack logits and per-node campaign score."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, embedding_dim)
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


class GATEncoder(nn.Module):
    """Two-layer GAT encoder for optional experiments."""

    def __init__(self, in_channels: int, hidden_channels: int, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
        self.conv2 = GATConv(hidden_channels, embedding_dim, heads=1)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.elu(self.conv1(x, edge_index))
        z = self.conv2(h, edge_index)
        logits = self.classifier(z)
        return z, logits


def load_fleet_graph(path: Path | str) -> Any:
    """Load PyG ``Data`` from ``fleet_graph.pt`` bundle."""
    bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
    data = bundle["pyg_data"]
    if not hasattr(data, "event_ids") and bundle.get("event_ids"):
        data.event_ids = bundle["event_ids"]
    return data


def _ensure_edges(data: Any) -> None:
    if data.edge_index is None or data.edge_index.numel() == 0:
        n = data.num_nodes
        idx = torch.arange(n, dtype=torch.long)
        data.edge_index = torch.stack([idx, idx], dim=0)
        logger.warning("Graph has no edges; using self-loops for GNN training.")


def _random_masks(
    num_nodes: int,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_nodes)
    n_train = int(num_nodes * train_ratio)
    n_val = int(num_nodes * val_ratio)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def _accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.sum() == 0:
        return 0.0
    pred = logits[mask].argmax(dim=-1)
    return float((pred == y[mask]).float().mean().item())


def train_gnn(
    data: Any,
    *,
    hidden_channels: int = 64,
    embedding_dim: int = 32,
    epochs: int = 30,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str | None = None,
    architecture: GnnArchitecture = "graphsage",
) -> tuple[nn.Module, dict[str, Any], np.ndarray]:
    """
    Train a GCN on node labels; return model, metrics, and node embeddings.
    """
    torch.manual_seed(seed)
    _ensure_edges(data)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x = data.x.to(dev)
    edge_index = data.edge_index.to(dev)
    y = data.y.long().to(dev)
    num_classes = int(y.max().item()) + 1 if y.numel() else 2
    num_classes = max(num_classes, 2)

    train_mask, val_mask, test_mask = _random_masks(
        data.num_nodes, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    train_mask = train_mask.to(dev)
    val_mask = val_mask.to(dev)
    test_mask = test_mask.to(dev)

    model_cls: type[nn.Module]
    if architecture == "gcn":
        model_cls = GCNEncoder
    elif architecture == "gat":
        model_cls = GATEncoder
    else:
        model_cls = GraphSAGEEncoder
    model = model_cls(x.size(1), hidden_channels, embedding_dim, num_classes).to(dev)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history: list[dict[str, float]] = []
    best_val = -1.0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        _, logits = model(x, edge_index)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z, logits = model(x, edge_index)
            train_acc = _accuracy(logits, y, train_mask)
            val_acc = _accuracy(logits, y, val_mask)
            test_acc = _accuracy(logits, y, test_mask)

        history.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.item()),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
            }
        )
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            logger.info(
                "Epoch %d/%d loss=%.4f train_acc=%.4f val_acc=%.4f test_acc=%.4f",
                epoch,
                epochs,
                loss.item(),
                train_acc,
                val_acc,
                test_acc,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        embeddings, logits = model(x, edge_index)
    emb_np = embeddings.cpu().numpy().astype(np.float32)

    metrics = {
        "epochs": epochs,
        "architecture": architecture,
        "hidden_channels": hidden_channels,
        "embedding_dim": embedding_dim,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "num_classes": num_classes,
        "best_val_acc": float(best_val),
        "final_test_acc": history[-1]["test_acc"] if history else 0.0,
        "final_train_loss": history[-1]["loss"] if history else 0.0,
        "history": history,
    }
    return model, metrics, emb_np


def train_graphsage_fleet_correlation(
    data: Any,
    *,
    hidden_channels: int = 64,
    embedding_dim: int = 32,
    epochs: int = 30,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str | None = None,
    campaign_loss_weight: float = 0.25,
) -> tuple[GraphSAGEFleetCorrelator, dict[str, Any], np.ndarray, np.ndarray]:
    """Train GraphSAGE for fleet correlation embeddings and campaign scores."""
    torch.manual_seed(seed)
    _ensure_edges(data)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    x = data.x.to(dev)
    edge_index = data.edge_index.to(dev)
    y = data.y.long().to(dev)
    num_classes = max(int(y.max().item()) + 1 if y.numel() else 2, 2)
    attack_target = (y > 0).float()

    train_mask, val_mask, test_mask = _random_masks(
        data.num_nodes, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )
    train_mask = train_mask.to(dev)
    val_mask = val_mask.to(dev)

    model = GraphSAGEFleetCorrelator(x.size(1), hidden_channels, embedding_dim, num_classes).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    best_val = -1.0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        _, logits, campaign_score = model(x, edge_index)
        cls_loss = F.cross_entropy(logits[train_mask], y[train_mask])
        camp_loss = F.binary_cross_entropy(campaign_score[train_mask], attack_target[train_mask])
        loss = cls_loss + campaign_loss_weight * camp_loss
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, logits, _ = model(x, edge_index)
            val_acc = _accuracy(logits, y, val_mask)

        history.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.item()),
                "train_acc": _accuracy(logits, y, train_mask),
                "val_acc": val_acc,
            }
        )
        if val_acc >= best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % max(1, epochs // 5) == 0 or epoch == epochs:
            logger.info("GraphSAGE epoch %d/%d loss=%.4f val_acc=%.4f", epoch, epochs, loss.item(), val_acc)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        embeddings, _, campaign_scores = model(x, edge_index)

    metrics = {
        "epochs": epochs,
        "architecture": "graphsage",
        "hidden_channels": hidden_channels,
        "embedding_dim": embedding_dim,
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "best_val_acc": float(best_val),
        "history": history,
    }
    return (
        model,
        metrics,
        embeddings.cpu().numpy().astype(np.float32),
        campaign_scores.cpu().numpy().astype(np.float32),
    )


def save_node_embeddings(
    embeddings: np.ndarray,
    event_ids: list[str],
    path: Path | str,
    *,
    gnn_type: str = "gcn",
    metrics: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(embeddings, columns=[f"embedding_{i}" for i in range(embeddings.shape[1])])
    df.insert(0, "event_id", event_ids)
    df.to_csv(out, index=False)
    logger.info("Saved %d node embeddings to %s", len(event_ids), out)
    return out


def save_training_metrics(metrics: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(metrics.get("history", []))
    if history.empty:
        history = pd.DataFrame([metrics])
    for key, value in metrics.items():
        if key != "history" and key not in history.columns:
            history[key] = value
    history.to_csv(out, index=False)
    logger.info("Saved GNN training metrics to %s", out)
    return out


def plot_training_loss(metrics: dict[str, Any], output_path: Path | str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    hist = pd.DataFrame(metrics.get("history", []))
    fig, ax = plt.subplots(figsize=(7, 4))
    if not hist.empty:
        ax.plot(hist["epoch"], hist["loss"], marker="o", linewidth=1.5)
    ax.set_title("GNN training loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_embedding_tsne(embeddings: np.ndarray, event_ids: list[str], output_path: Path | str, *, seed: int = 42) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(embeddings)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n, 5000), replace=False) if n > 5000 else np.arange(n)
    X = StandardScaler().fit_transform(embeddings[idx])
    perplexity = max(5, min(30, len(idx) - 1))
    xy = TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.6)
    ax.set_title("GNN node embeddings (t-SNE)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def train_gnn_from_graph_file(
    graph_path: Path | str,
    embeddings_path: Path | str,
    checkpoint_dir: Path | str | None = None,
    *,
    config: dict[str, Any] | None = None,
    seed: int = 42,
    metrics_path: Path | str | None = None,
    loss_plot_path: Path | str | None = None,
    tsne_plot_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load fleet graph, train a GNN, save CSV embeddings/metrics/figures."""
    cfg = config or {}
    data = load_fleet_graph(graph_path)
    event_ids = list(getattr(data, "event_ids", []))
    if len(event_ids) != data.num_nodes:
        event_ids = [f"node_{i}" for i in range(data.num_nodes)]

    arch = str(cfg.get("architecture", cfg.get("gnn_model", "graphsage"))).lower()
    if arch in {"sage", "graph_sage"}:
        arch = "graphsage"
    model, metrics, emb = train_gnn(
        data,
        hidden_channels=int(cfg.get("hidden_channels", 64)),
        embedding_dim=int(cfg.get("embedding_dim", 32)),
        epochs=int(cfg.get("epochs", 30)),
        learning_rate=float(cfg.get("learning_rate", 0.01)),
        weight_decay=float(cfg.get("weight_decay", 5e-4)),
        train_ratio=float(cfg.get("train_ratio", 0.7)),
        val_ratio=float(cfg.get("val_ratio", 0.15)),
        seed=seed,
        device=cfg.get("device"),
        architecture=arch if arch in {"graphsage", "gcn", "gat"} else "graphsage",
    )

    save_node_embeddings(
        emb, event_ids, embeddings_path, gnn_type=str(metrics.get("architecture", "graphsage")), metrics=metrics
    )
    if np.isnan(emb).any():
        raise ValueError("GNN embeddings contain NaN values")
    if len(emb) != data.num_nodes:
        raise ValueError("Embedding count does not match graph node count")
    if metrics_path is not None:
        save_training_metrics(metrics, metrics_path)
    if loss_plot_path is not None:
        plot_training_loss(metrics, loss_plot_path)
    if tsne_plot_path is not None:
        plot_embedding_tsne(emb, event_ids, tsne_plot_path, seed=seed)

    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f"{metrics.get('architecture', 'graphsage')}_best.pt"
        torch.save({"model_state": model.state_dict(), "metrics": metrics}, ckpt_path)
        logger.info("Wrote checkpoint to %s", ckpt_path)

    return metrics
