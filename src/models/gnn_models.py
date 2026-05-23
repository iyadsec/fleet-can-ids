"""Graph neural network training on fleet anomaly graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv

from src.utils.logging import get_logger

logger = get_logger(__name__)

GnnArchitecture = Literal["gcn"]


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
) -> tuple[GCNEncoder, dict[str, Any], np.ndarray]:
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

    model = GCNEncoder(
        in_channels=x.size(1),
        hidden_channels=hidden_channels,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
    ).to(dev)
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
    payload: dict[str, Any] = {
        "embeddings": embeddings,
        "event_ids": event_ids,
        "gnn_type": gnn_type,
    }
    if metrics is not None:
        payload["metrics"] = metrics
    torch.save(payload, out)
    logger.info("Saved %d node embeddings to %s", len(event_ids), out)
    return out


def train_gnn_from_graph_file(
    graph_path: Path | str,
    embeddings_path: Path | str,
    checkpoint_dir: Path | str | None = None,
    *,
    config: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Load fleet graph, train GCN, save embeddings and optional checkpoint."""
    cfg = config or {}
    data = load_fleet_graph(graph_path)
    event_ids = list(getattr(data, "event_ids", []))
    if len(event_ids) != data.num_nodes:
        event_ids = [f"node_{i}" for i in range(data.num_nodes)]

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
    )

    save_node_embeddings(
        emb, event_ids, embeddings_path, gnn_type=str(cfg.get("architecture", "gcn")), metrics=metrics
    )

    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "gcn_best.pt"
        torch.save({"model_state": model.state_dict(), "metrics": metrics}, ckpt_path)
        logger.info("Wrote checkpoint to %s", ckpt_path)

    return metrics
