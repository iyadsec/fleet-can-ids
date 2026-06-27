"""Cluster GNN embeddings to detect multi-vehicle suspicious attack campaigns."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from src.utils.logging import get_logger

logger = get_logger(__name__)

ClusterAlgorithm = Literal["kmeans", "dbscan"]

TIMING_FEATURE_SUBSTRINGS = ("inter_arrival", "timestamp")


def clustering_feature_columns(columns: list[str]) -> list[str]:
    """Behavioural columns for clustering (excludes timing)."""
    return [
        c
        for c in columns
        if not any(tok in c.lower() for tok in TIMING_FEATURE_SUBSTRINGS)
    ]


def event_id_to_window_id(event_id: str) -> int:
    return int(str(event_id).rsplit("-", 1)[-1])


def _cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    Xn = X / norms
    return Xn @ Xn.T


def mean_intra_cluster_similarity(X: np.ndarray, max_pairs: int = 2000) -> float:
    """Mean pairwise cosine similarity within a cluster (no timing)."""
    n = len(X)
    if n < 2:
        return 1.0
    if n > 80:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=min(n, 80), replace=False)
        X = X[idx]
        n = len(X)
    sim = _cosine_similarity_matrix(X.astype(np.float64))
    triu = sim[np.triu_indices(n, k=1)]
    return float(np.mean(triu)) if triu.size else 1.0


def _load_behavioural_matrix_from_features(
    descriptors_path: Path,
    features_path: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Join anomaly descriptors to ``window_features.csv`` (no JSON parse, no timing)."""
    from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS

    desc = pd.read_csv(
        descriptors_path,
        usecols=[
            "event_id",
            "vehicle_model",
            "attack_type",
            "evidence_level",
            "anomaly_score",
            "local_alert",
            "weak_signal",
        ],
    )
    feat_cols = clustering_feature_columns(list(BEHAVIOURAL_FEATURE_COLUMNS))

    usecols = ["window_id", *feat_cols]
    features = pd.read_csv(features_path, usecols=usecols)
    desc = desc.copy()
    desc["window_id"] = desc["event_id"].map(event_id_to_window_id)
    merged = desc.merge(features, on="window_id", how="inner", suffixes=("", "_feat"))
    if merged.empty:
        raise ValueError("No descriptor rows matched window_features (check event_id / window_id).")
    if len(merged) < len(desc):
        logger.warning(
            "Dropped %d descriptors without matching window features.",
            len(desc) - len(merged),
        )
    X = merged[feat_cols].to_numpy(dtype=np.float32)
    meta = merged[
        [
            "event_id",
            "window_id",
            "vehicle_model",
            "attack_type",
            "evidence_level",
            "anomaly_score",
            "local_alert",
            "weak_signal",
        ]
    ].copy()
    meta["embedding_source"] = "behavioural_fallback"
    return X, meta


def load_embedding_table(
    embeddings_path: Path | str,
    descriptors_path: Path | str | None = None,
    features_path: Path | str | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Load node embeddings and metadata.

    Primary: ``*_node_embeddings.pt`` from GNN training.
    Fallback: behavioural features from ``window_features.csv`` joined to descriptors.
    """
    emb_path = Path(embeddings_path)
    meta = pd.DataFrame()

    if emb_path.exists() and emb_path.suffix == ".csv":
        df = pd.read_csv(emb_path)
        emb_cols = [c for c in df.columns if c.startswith("embedding_")]
        if not emb_cols:
            raise ValueError(f"No embedding_* columns found in {emb_path}")
        X = df[emb_cols].to_numpy(dtype=np.float32)
        meta = df[["event_id"]].copy()
        meta["embedding_source"] = "gnn_csv"
        logger.info("Loaded %d GNN embeddings from %s", len(X), emb_path)
    elif emb_path.exists() and emb_path.suffix == ".pt":
        import torch

        bundle = torch.load(emb_path, map_location="cpu", weights_only=False)
        X = np.asarray(bundle["embeddings"], dtype=np.float32)
        event_ids = list(bundle.get("event_ids", []))
        meta = pd.DataFrame({"event_id": event_ids})
        meta["embedding_source"] = bundle.get(
            "gnn_type", emb_path.stem.replace("_node_embeddings", "")
        )
        logger.info("Loaded %d GNN embeddings from %s", len(X), emb_path)
    elif descriptors_path and Path(descriptors_path).exists():
        desc_p = Path(descriptors_path)
        feat_p = Path(features_path) if features_path else desc_p.parent / "window_features.csv"
        if feat_p.exists():
            logger.warning("GNN embeddings not found; using window features (no timing).")
            X, meta = _load_behavioural_matrix_from_features(desc_p, feat_p)
        else:
            import json

            logger.warning("Using JSON behavioural vectors from descriptors (slow).")
            desc = pd.read_csv(desc_p)
            vectors = [
                json.loads(raw) if isinstance(raw, str) else raw
                for raw in desc["behavioural_feature_vector"]
            ]
            X = np.asarray(vectors, dtype=np.float32)
            meta = desc[
                [
                    "event_id",
                    "vehicle_model",
                    "attack_type",
                    "evidence_level",
                    "anomaly_score",
                    "local_alert",
                    "weak_signal",
                ]
            ].copy()
            meta["embedding_source"] = "behavioural_fallback"
    else:
        raise FileNotFoundError(
            f"No embeddings at {emb_path} and no descriptors fallback at {descriptors_path}"
        )

    if emb_path.exists() and emb_path.suffix in {".pt", ".csv"} and descriptors_path and Path(descriptors_path).exists():
        extra = pd.read_csv(
            descriptors_path,
            usecols=[
                "event_id",
                "window_id",
                "vehicle_model",
                "source_file",
                "attack_type",
                "evidence_level",
                "anomaly_score",
                "local_alert",
                "weak_signal",
            ],
        )
        meta = meta.merge(extra, on="event_id", how="left", suffixes=("", "_dup"))
        meta = meta[[c for c in meta.columns if not c.endswith("_dup")]]

    if len(meta) != len(X):
        raise ValueError(f"Metadata rows ({len(meta)}) != embeddings ({len(X)})")

    return X, meta


def subsample_indices(
    meta: pd.DataFrame,
    max_samples: int,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Stratified subsample indices (by vehicle) for scalable clustering."""
    n = len(meta)
    if n <= max_samples:
        return np.arange(n)
    frac = max_samples / n
    idx_parts = []
    for _, group in meta.groupby("vehicle_model", dropna=False):
        k = max(1, int(round(len(group) * frac)))
        idx_parts.append(
            group.sample(n=min(k, len(group)), random_state=seed).index.to_numpy()
        )
    idx = np.unique(np.concatenate(idx_parts))
    if len(idx) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_samples, replace=False)
    return np.sort(idx)


def run_kmeans(
    X: np.ndarray,
    *,
    n_clusters: int = 12,
    random_state: int = 42,
) -> tuple[np.ndarray, StandardScaler, KMeans | MiniBatchKMeans]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    if len(X) > 10_000:
        model: KMeans | MiniBatchKMeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=4096,
            n_init=10,
        )
    else:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(Xs)
    return labels.astype(int), scaler, model


def predict_kmeans_labels(
    model: KMeans | MiniBatchKMeans,
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    return model.predict(scaler.transform(X)).astype(int)


class DbscanProjector:
    """Scale + PCA projection for DBSCAN (density only; similarity uses raw *X*)."""

    def __init__(self, n_components: int = 8, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, X_fit: np.ndarray) -> "DbscanProjector":
        Xs = self.scaler.fit_transform(X_fit)
        n_comp = min(self.n_components, Xs.shape[1], max(1, Xs.shape[0] - 1))
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca.fit(Xs)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("DbscanProjector.fit must be called before transform.")
        return self.pca.transform(self.scaler.transform(X))


def run_dbscan(
    X: np.ndarray,
    *,
    eps: float = 1.2,
    min_samples: int = 10,
    pca_components: int = 8,
    random_state: int = 42,
) -> tuple[np.ndarray, DbscanProjector]:
    """DBSCAN on scaled PCA space (no timing in *X*)."""
    projector = DbscanProjector(n_components=pca_components, random_state=random_state)
    Z = projector.fit(X).transform(X)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(Z)
    return labels.astype(int), projector


def extend_dbscan_labels(
    X: np.ndarray,
    fit_labels: np.ndarray,
    X_fit: np.ndarray,
    projector: DbscanProjector,
    *,
    eps: float = 1.2,
) -> np.ndarray:
    """Assign each point to nearest DBSCAN centroid in PCA space within *eps*, else noise."""
    cluster_ids = [int(c) for c in np.unique(fit_labels) if int(c) != -1]
    full_labels = np.full(len(X), -1, dtype=int)
    if not cluster_ids:
        return full_labels

    Z_fit = projector.transform(X_fit)
    Z = projector.transform(X)
    centroids = np.stack([Z_fit[fit_labels == cid].mean(axis=0) for cid in cluster_ids], axis=0)
    dists = pairwise_distances(Z, centroids)
    nearest = np.argmin(dists, axis=1)
    min_d = dists[np.arange(len(X)), nearest]
    valid = min_d <= eps
    full_labels[valid] = np.asarray(cluster_ids, dtype=int)[nearest[valid]]
    return full_labels


def summarize_clusters(
    labels: np.ndarray,
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    algorithm: str,
    similarity_threshold: float = 0.85,
    min_vehicles: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-event assignments and per-cluster summary.

    Suspicious campaign cluster:
    - anomalies from >= *min_vehicles* distinct vehicle models
    - mean intra-cluster cosine similarity >= *similarity_threshold*
    """
    meta = meta.reset_index(drop=True)
    cluster_stats: dict[int, dict[str, Any]] = {}
    unique_ids = sorted({int(c) for c in np.unique(labels)})

    for cid in unique_ids:
        mask = labels == cid
        cluster_size = int(mask.sum())
        n_vehicles = int(meta.loc[mask, "vehicle_model"].nunique(dropna=True))
        is_noise = cid == -1
        if is_noise or cluster_size < 2:
            mean_sim = 0.0 if is_noise else 1.0
        else:
            mean_sim = mean_intra_cluster_similarity(X[mask])
        is_suspicious = (
            (not is_noise)
            and n_vehicles >= min_vehicles
            and mean_sim >= similarity_threshold
        )
        vehicles = ",".join(sorted(meta.loc[mask, "vehicle_model"].dropna().unique()))
        dominant_attack = (
            meta.loc[mask, "attack_type"].mode(dropna=True).iloc[0]
            if not meta.loc[mask, "attack_type"].dropna().empty
            else "unknown"
        )
        cluster_stats[cid] = {
            "cluster_size": cluster_size,
            "num_unique_vehicles": n_vehicles,
            "vehicles_in_cluster": vehicles,
            "dominant_attack_type": dominant_attack,
            "mean_behavioural_similarity": round(mean_sim, 6),
            "is_cross_vehicle_cluster": bool((not is_noise) and n_vehicles > 1),
            "is_suspicious_campaign": bool(is_suspicious),
        }

    events = meta.copy()
    events["algorithm"] = algorithm
    events["cluster_id"] = labels.astype(int)
    events["cluster_size"] = events["cluster_id"].map(lambda c: cluster_stats[c]["cluster_size"])
    events["vehicles_in_cluster"] = events["cluster_id"].map(lambda c: cluster_stats[c]["vehicles_in_cluster"])
    events["num_unique_vehicles"] = events["cluster_id"].map(lambda c: cluster_stats[c]["num_unique_vehicles"])
    events["dominant_attack_type"] = events["cluster_id"].map(lambda c: cluster_stats[c]["dominant_attack_type"])
    events["is_cross_vehicle_cluster"] = events["cluster_id"].map(
        lambda c: cluster_stats[c]["is_cross_vehicle_cluster"]
    )
    events["mean_cluster_similarity"] = events["cluster_id"].map(
        lambda c: cluster_stats[c]["mean_behavioural_similarity"]
    )
    events["is_suspicious_campaign"] = events["cluster_id"].map(
        lambda c: cluster_stats[c]["is_suspicious_campaign"]
    )
    required = [
        "event_id",
        "window_id",
        "vehicle_model",
        "attack_type",
        "evidence_level",
        "anomaly_score",
        "local_alert",
        "weak_signal",
        "algorithm",
        "cluster_id",
        "cluster_size",
        "vehicles_in_cluster",
        "num_unique_vehicles",
        "dominant_attack_type",
        "is_cross_vehicle_cluster",
        "mean_cluster_similarity",
    ]

    summary_rows = [
        {
            "algorithm": algorithm,
            "cluster_id": cid,
            "cluster_size": st["cluster_size"],
            "num_unique_vehicles": st["num_unique_vehicles"],
            "mean_cluster_similarity": st["mean_behavioural_similarity"],
            "is_suspicious_campaign": st["is_suspicious_campaign"],
            "is_noise": cid == -1,
        }
        for cid, st in cluster_stats.items()
    ]
    return events[required], pd.DataFrame(summary_rows)


def run_campaign_clustering(
    X: np.ndarray,
    meta: pd.DataFrame,
    *,
    similarity_threshold: float = 0.85,
    min_vehicles: int = 2,
    kmeans_clusters: int = 12,
    dbscan_eps: float = 1.2,
    dbscan_min_samples: int = 10,
    dbscan_pca_components: int = 8,
    random_state: int = 42,
    max_clustering_samples: int | None = 20_000,
    method: str = "dbscan",
) -> pd.DataFrame:
    """Run DBSCAN by default; KMeans can be enabled for optional comparison."""
    parts: list[pd.DataFrame] = []

    if max_clustering_samples and len(X) > max_clustering_samples:
        fit_idx = subsample_indices(meta, max_clustering_samples, seed=random_state)
        logger.info("Clustering fit on %d / %d samples", len(fit_idx), len(X))
    else:
        fit_idx = np.arange(len(X))

    X_fit, meta_fit = X[fit_idx], meta.iloc[fit_idx].reset_index(drop=True)

    if method.lower() in {"kmeans", "both"}:
        km_fit, scaler, km_model = run_kmeans(
            X_fit, n_clusters=kmeans_clusters, random_state=random_state
        )
        km_labels = predict_kmeans_labels(km_model, scaler, X)
        km_events, _ = summarize_clusters(
            km_labels,
            X,
            meta.reset_index(drop=True),
            algorithm="kmeans",
            similarity_threshold=similarity_threshold,
            min_vehicles=min_vehicles,
        )
        parts.append(km_events)

    if method.lower() in {"dbscan", "both"}:
        db_fit, db_projector = run_dbscan(
            X_fit,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
            pca_components=dbscan_pca_components,
            random_state=random_state,
        )
        db_labels = extend_dbscan_labels(
            X, db_fit, X_fit, db_projector, eps=dbscan_eps
        )
        db_events, _ = summarize_clusters(
            db_labels,
            X,
            meta.reset_index(drop=True),
            algorithm="dbscan",
            similarity_threshold=similarity_threshold,
            min_vehicles=min_vehicles,
        )
        parts.append(db_events)

    return pd.concat(parts, ignore_index=True)


def build_cluster_summary_table(assignments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate suspicious campaign stats per algorithm."""
    return (
        assignments.groupby(["algorithm", "cluster_id"], as_index=False)
        .agg(
            cluster_size=("event_id", "count"),
            num_unique_vehicles=("num_unique_vehicles", "first"),
            mean_cluster_similarity=("mean_cluster_similarity", "first"),
            is_cross_vehicle_cluster=("is_cross_vehicle_cluster", "first"),
            vehicles_in_cluster=("vehicles_in_cluster", "first"),
            dominant_attack_type=("dominant_attack_type", "first"),
        )
    )


def save_campaign_clusters(df: pd.DataFrame, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved %d cluster rows to %s", len(df), out)
    return out


def plot_tsne_clusters(
    X: np.ndarray,
    assignments: pd.DataFrame,
    meta: pd.DataFrame,
    output_path: Path | str,
    *,
    algorithm: str,
    max_points: int = 5000,
    seed: int = 42,
) -> Path:
    """t-SNE plot colored by cluster; highlight suspicious multi-vehicle campaigns."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    sub = assignments[assignments["algorithm"] == algorithm].copy()
    labels = sub["cluster_id"].to_numpy()
    suspicious = sub["is_cross_vehicle_cluster"].to_numpy()

    n = len(X)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        Xp, lab, sus = X[idx], labels[idx], suspicious[idx]
    else:
        idx = np.arange(n)
        Xp, lab, sus = X, labels, suspicious

    xy = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(Xp) - 1)).fit_transform(
        StandardScaler().fit_transform(Xp)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    noise = lab == -1
    ax.scatter(xy[~sus & ~noise, 0], xy[~sus & ~noise, 1], c=lab[~sus & ~noise], cmap="tab20", s=12, alpha=0.5, label="other")
    if noise.any():
        ax.scatter(xy[noise, 0], xy[noise, 1], c="lightgray", s=8, alpha=0.3, label="noise")
    if sus.any():
        ax.scatter(xy[sus, 0], xy[sus, 1], c="red", s=40, alpha=0.9, edgecolors="black", linewidths=0.5, label="suspicious campaign")
    ax.set_title(f"t-SNE clusters ({algorithm}) — multi-vehicle suspicious campaigns")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote t-SNE cluster plot to %s", out)
    return out


def plot_cluster_summaries(
    assignments: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """Summary bar charts: cluster sizes and suspicious multi-vehicle counts."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    summary = build_cluster_summary_table(assignments)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for ax, algo in zip(axes[0], ["kmeans", "dbscan"]):
        sub = summary[summary["algorithm"] == algo].sort_values("cluster_id")
        colors = sub["is_cross_vehicle_cluster"].map({True: "crimson", False: "steelblue"})
        ax.bar(sub["cluster_id"].astype(str), sub["cluster_size"], color=colors)
        ax.set_title(f"{algo}: cluster size")
        ax.set_xlabel("cluster_id")
        ax.tick_params(axis="x", rotation=45)

    for ax, algo in zip(axes[1], ["kmeans", "dbscan"]):
        sub = summary[summary["algorithm"] == algo]
        sus = sub[sub["is_cross_vehicle_cluster"]]
        if sus.empty:
            ax.text(0.5, 0.5, "No cross-vehicle clusters", ha="center", va="center")
        else:
            ax.barh(sus["cluster_id"].astype(str), sus["num_unique_vehicles"], color="crimson")
            ax.set_title(f"{algo}: cross-vehicle — vehicle count")
        ax.set_xlabel("distinct vehicles")

    fig.suptitle("Campaign cluster summaries (no temporal features)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote cluster summary plot to %s", out)
    return out


def print_cluster_report(assignments: pd.DataFrame) -> None:
    """Print suspicious multi-vehicle campaign clusters."""
    print("\n=== Suspicious Multi-Vehicle Campaign Clusters ===")
    for algo in assignments["algorithm"].unique():
        sub = assignments[(assignments["algorithm"] == algo) & assignments["is_cross_vehicle_cluster"]]
        clusters = sub.drop_duplicates(subset=["cluster_id"])
        print(f"\n  [{algo}] {len(clusters)} suspicious cluster(s)")
        for _, row in clusters.iterrows():
            print(
                f"    cluster {row['cluster_id']}: size={row['cluster_size']}, "
                f"vehicles={row['num_unique_vehicles']}, sim={row['mean_cluster_similarity']:.4f}"
            )
    print("================================================\n")
