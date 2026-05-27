"""Research-ready figures, tables, and experiment report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.preprocessing import StandardScaler

from src.data.dataset_loader import dataset_statistics, dataset_statistics_chunked
from src.evaluation.campaign_clustering import (
    clustering_feature_columns,
    load_embedding_table,
    plot_cluster_summaries,
    plot_tsne_clusters,
    run_campaign_clustering,
    save_campaign_clusters,
)
from src.features.descriptor_generator import descriptor_statistics
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from sklearn.neighbors import NearestNeighbors

from src.graph.fleet_graph_builder import (
    _prepare_features,
    _similarity_from_distance,
    load_anomaly_descriptors,
    parse_feature_matrix,
)
from src.models.vehicle_ids import (
    VEHICLE_MODELS,
    load_feature_dataset,
    plot_vehicle_confusion_matrices,
    predict_autoencoder,
    predict_isolation_forest,
    predict_logistic_regression,
    predict_random_forest,
    prepare_vehicle_split,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    sub = df.head(max_rows)
    cols = sub.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(sub.iloc[i][c]) for c in cols) + " |" for i in range(len(sub))]
    text = "\n".join([header, sep, *rows])
    if len(df) > max_rows:
        text += f"\n\n_({len(df) - max_rows} additional rows omitted)_"
    return text


RESEARCH_FIGURES = {
    "confusion": "confusion_matrices.png",
    "roc": "roc_curves.png",
    "feature_importance": "feature_importance.png",
    "graph": "fleet_graph_visualization.png",
    "embedding_tsne": "gnn_embedding_tsne.png",
    "cluster_kmeans": "campaign_tsne_kmeans.png",
    "cluster_dbscan": "campaign_tsne_dbscan.png",
    "cluster_summary": "campaign_cluster_summary.png",
    "feature_corr": "feature_correlation_heatmap.png",
    "feature_dist": "feature_distributions.png",
}


def _fig_path(figures_dir: Path, name: str) -> Path:
    return figures_dir / RESEARCH_FIGURES.get(name, name)


def plot_vehicle_roc_curves(
    features_path: Path | str,
    output_path: Path | str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    include_autoencoder: bool = True,
    ae_epochs: int = 15,
) -> Path | None:
    """ROC curves per vehicle (held-out test split)."""
    df = load_feature_dataset(features_path)
    predictors: list[tuple[str, Any]] = [
        ("random_forest", predict_random_forest),
        ("logistic_regression", predict_logistic_regression),
        ("isolation_forest", predict_isolation_forest),
    ]
    if include_autoencoder:
        predictors.append(("autoencoder", predict_autoencoder))

    vehicles = [v for v in VEHICLE_MODELS if v in df["vehicle_model"].unique()]
    nrows = len(vehicles)
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 4 * max(nrows, 1)))
    axes = np.atleast_1d(axes)

    for ax, vehicle in zip(axes, vehicles):
        split = prepare_vehicle_split(
            df, vehicle, test_size=test_size, random_state=random_state
        )
        for model_name, predict_fn in predictors:
            try:
                if model_name in ("isolation_forest", "autoencoder"):
                    kwargs = {"random_state": random_state}
                    if model_name == "autoencoder":
                        kwargs["epochs"] = ae_epochs
                    y_pred, y_score = predict_fn(
                        split.X_train_benign, split.X_test, **kwargs
                    )
                else:
                    y_pred, y_score = predict_fn(
                        split.X_train,
                        split.y_train,
                        split.X_test,
                        random_state=random_state,
                    )
                fpr, tpr, _ = roc_curve(split.y_test, y_score)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
            except Exception as exc:
                logger.warning("ROC skipped %s/%s: %s", vehicle, model_name, exc)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title(f"{vehicle} — ROC (test split)")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Vehicle-level IDS — ROC curves", fontsize=13)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote ROC curves to %s", out)
    return out


def plot_feature_importance(
    features_path: Path | str,
    output_path: Path | str,
    *,
    random_state: int = 42,
    top_k: int = 12,
) -> Path:
    """Random Forest feature importance per vehicle (train split)."""
    df = load_feature_dataset(features_path)
    feat_cols = clustering_feature_columns(list(BEHAVIOURAL_FEATURE_COLUMNS))
    vehicles = [v for v in VEHICLE_MODELS if v in df["vehicle_model"].unique()]

    fig, axes = plt.subplots(1, len(vehicles), figsize=(5 * len(vehicles), 5))
    if len(vehicles) == 1:
        axes = [axes]

    for ax, vehicle in zip(axes, vehicles):
        sub = df[df["vehicle_model"] == vehicle]
        X = sub[feat_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y = sub["label"].to_numpy(dtype=np.int64)
        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state, n_jobs=-1
        )
        clf.fit(X, y)
        imp = clf.feature_importances_
        order = np.argsort(imp)[-top_k:]
        ax.barh(
            [feat_cols[i] for i in order],
            imp[order],
            color="steelblue",
        )
        ax.set_title(f"{vehicle}")
        ax.set_xlabel("Importance")

    fig.suptitle("Random Forest feature importance (no timing features)", fontsize=12)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote feature importance to %s", out)
    return out


def _build_sparse_viz_graph(
    descriptors: pd.DataFrame,
    *,
    metric: str = "cosine",
    threshold: float = 0.85,
    max_nodes: int = 800,
    k_neighbours: int = 12,
    seed: int = 42,
) -> nx.Graph:
    """Lightweight kNN graph for visualization (avoids dense radius graphs)."""
    X, node_ids = parse_feature_matrix(descriptors)
    df = descriptors.reset_index(drop=True)
    if len(X) > max_nodes:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=max_nodes, replace=False)
        idx.sort()
        X, df = X[idx], df.iloc[idx].reset_index(drop=True)
        node_ids = [node_ids[i] for i in idx]

    X_prep = _prepare_features(X, metric)  # type: ignore[arg-type]
    nn = NearestNeighbors(n_neighbors=min(k_neighbours + 1, len(X_prep)), metric=metric)
    nn.fit(X_prep)
    dists, indices = nn.kneighbors(X_prep)
    sims = _similarity_from_distance(dists[:, 1:], metric)  # type: ignore[arg-type]

    G = nx.Graph()
    for i, eid in enumerate(node_ids):
        G.add_node(
            eid,
            vehicle_model=str(df.loc[i, "vehicle_model"]),
            attack_type=str(df.loc[i, "attack_type"]),
        )
    for i in range(len(X_prep)):
        for j, sim in zip(indices[i, 1:], sims[i]):
            if sim >= threshold:
                u, v = node_ids[i], node_ids[int(j)]
                if u != v:
                    G.add_edge(u, v, weight=float(sim), similarity=float(sim))
    return G


def plot_fleet_graph_visualization(
    descriptors_path: Path | str,
    output_path: Path | str,
    *,
    graphml_path: Path | str | None = None,
    metric: str = "cosine",
    threshold: float = 0.85,
    max_nodes: int = 800,
    seed: int = 42,
) -> Path:
    """Spring-layout fleet graph coloured by vehicle model."""
    out = Path(output_path)
    G: nx.Graph | None = None

    if graphml_path and Path(graphml_path).exists():
        G = nx.read_graphml(graphml_path)
        if G.number_of_nodes() > max_nodes:
            rng = np.random.default_rng(seed)
            nodes = list(G.nodes())
            keep = rng.choice(len(nodes), size=max_nodes, replace=False)
            G = G.subgraph([nodes[i] for i in keep]).copy()

    if G is None or G.number_of_nodes() == 0:
        descriptors = load_anomaly_descriptors(descriptors_path)
        G = _build_sparse_viz_graph(
            descriptors,
            metric=metric,
            threshold=threshold,
            max_nodes=max_nodes,
            seed=seed,
        )

    vehicle_colors = {"Hyundai": "#1f77b4", "Kia": "#ff7f0e", "Chevrolet": "#2ca02c"}
    node_colors = []
    for node in G.nodes():
        data = G.nodes[node]
        vm = data.get("vehicle_model", "unknown")
        node_colors.append(vehicle_colors.get(str(vm), "#888888"))

    fig, ax = plt.subplots(figsize=(11, 9))
    pos = nx.spring_layout(G, seed=seed, k=0.15, iterations=50)
    weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, pos, ax=ax, alpha=0.08, width=[0.3 + 1.5 * w for w in weights], edge_color="gray"
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors, node_size=18, alpha=0.75, linewidths=0
    )
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8, label=v)
        for v, c in vehicle_colors.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.set_title(
        f"Fleet anomaly graph (behavioural similarity ≥ {threshold}, n={G.number_of_nodes()})"
    )
    ax.axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote graph visualization to %s", out)
    return out


def plot_gnn_embedding_tsne(
    embeddings_path: Path | str,
    descriptors_path: Path | str,
    features_path: Path | str,
    output_path: Path | str,
    *,
    max_points: int = 5000,
    seed: int = 42,
) -> Path:
    """t-SNE of GNN (or fallback) embeddings coloured by vehicle."""
    X, meta = load_embedding_table(embeddings_path, descriptors_path, features_path)
    n = len(X)
    if n > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_points, replace=False)
        Xp, meta_p = X[idx], meta.iloc[idx].reset_index(drop=True)
    else:
        Xp, meta_p = X, meta.reset_index(drop=True)

    xy = TSNE(n_components=2, random_state=seed, perplexity=min(30, len(Xp) - 1)).fit_transform(
        StandardScaler().fit_transform(Xp)
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    vehicles = meta_p["vehicle_model"].astype(str)
    for vehicle in sorted(vehicles.unique()):
        mask = vehicles == vehicle
        ax.scatter(xy[mask, 0], xy[mask, 1], s=14, alpha=0.55, label=vehicle)
    source = meta_p.get("embedding_source", pd.Series(["unknown"])).iloc[0]
    ax.set_title(f"t-SNE of node embeddings ({source})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote embedding t-SNE to %s", out)
    return out


def write_metrics_summary_tables(
    *,
    paths: dict[str, Path],
    metrics_dir: Path,
    seed: int = 42,
) -> dict[str, Path]:
    """Export consolidated CSV summary tables for the report."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if paths.get("vehicle_results") and paths["vehicle_results"].exists():
        ids = pd.read_csv(paths["vehicle_results"])
        summary = ids[
            ["vehicle_model", "model", "task", "accuracy", "precision", "recall", "f1", "roc_auc"]
        ].round(4)
        p = metrics_dir / "summary_vehicle_ids.csv"
        summary.to_csv(p, index=False)
        written["vehicle_ids"] = p

    if paths.get("fleet_cluster_results") and paths["fleet_cluster_results"].exists():
        cdf = pd.read_csv(paths["fleet_cluster_results"])
        sus = (
            cdf[cdf["is_suspicious_campaign"]]
            .drop_duplicates(subset=["algorithm", "cluster_id"])
            .sort_values(["algorithm", "cluster_size"], ascending=[True, False])
        )
        p = metrics_dir / "summary_suspicious_campaigns.csv"
        sus.to_csv(p, index=False)
        written["campaigns"] = p

    if paths.get("descriptors") and paths["descriptors"].exists():
        desc = pd.read_csv(paths["descriptors"], nrows=500_000)
        stats = descriptor_statistics(desc)
        p = metrics_dir / "summary_descriptors.json"
        p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        written["descriptors"] = p

    if paths.get("clean_can_data") and paths["clean_can_data"].exists():
        stats = dataset_statistics_chunked(paths["clean_can_data"])
        p = metrics_dir / "summary_dataset.json"
        p.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        written["dataset"] = p

    if paths.get("graph_stats") and paths["graph_stats"].exists():
        written["graph"] = paths["graph_stats"]

    if paths.get("gnn_metrics") and paths["gnn_metrics"].exists():
        written["gnn"] = paths["gnn_metrics"]

    if paths.get("final_outcome_summary") and paths["final_outcome_summary"].exists():
        written["final_outcomes"] = paths["final_outcome_summary"]

    return written


def ensure_clustering_outputs(
    config: dict[str, Any],
    paths: dict[str, Path],
    figures_dir: Path,
    *,
    seed: int = 42,
    regenerate: bool = False,
) -> pd.DataFrame | None:
    """Run or reload campaign clustering and figures."""
    cluster_csv = paths.get("fleet_cluster_results")
    if cluster_csv is None:
        return None

    cluster_cfg = config.get("clustering", {})
    emb = paths.get("gcn_embeddings")
    desc = paths.get("descriptors")
    feat = paths.get("window_features")

    if not regenerate and cluster_csv.exists():
        return pd.read_csv(cluster_csv)

    if desc is None or not desc.exists() or feat is None or not feat.exists():
        logger.warning("Cannot run clustering — missing descriptors or features.")
        return pd.read_csv(cluster_csv) if cluster_csv.exists() else None

    X, meta = load_embedding_table(emb, desc, feat)
    max_samples = cluster_cfg.get("max_clustering_samples")
    assignments = run_campaign_clustering(
        X,
        meta,
        similarity_threshold=float(cluster_cfg.get("similarity_threshold", 0.85)),
        min_vehicles=int(cluster_cfg.get("min_vehicles", 2)),
        kmeans_clusters=int(cluster_cfg.get("kmeans_clusters", 12)),
        dbscan_eps=float(cluster_cfg.get("dbscan_eps", 1.2)),
        dbscan_min_samples=int(cluster_cfg.get("dbscan_min_samples", 10)),
        dbscan_pca_components=int(cluster_cfg.get("dbscan_pca_components", 8)),
        random_state=seed,
        max_clustering_samples=int(max_samples) if max_samples else None,
    )
    save_campaign_clusters(assignments, cluster_csv)
    for algo in ("kmeans", "dbscan"):
        sub = assignments[assignments["algorithm"] == algo]
        plot_tsne_clusters(
            X, sub, meta, figures_dir / RESEARCH_FIGURES[f"cluster_{algo}"], algorithm=algo, seed=seed
        )
    plot_cluster_summaries(assignments, figures_dir / RESEARCH_FIGURES["cluster_summary"])
    return assignments


def generate_all_research_outputs(
    config: dict[str, Any],
    root: Path,
    *,
    regenerate_clustering: bool = False,
) -> dict[str, Any]:
    """Generate figures, summary tables, and experiment report markdown."""
    pipeline = config.get("pipeline", {})
    artifacts = pipeline.get("artifacts", {})
    research_cfg = config.get("research", {})

    def art(key: str, default: str) -> Path:
        return root / artifacts.get(key, default)

    paths = {
        "clean_can_data": art("clean_can_data", "data/processed/clean_can_data.csv"),
        "window_metadata": art("window_metadata", "data/processed/window_metadata.csv"),
        "window_features": art("window_features", "data/processed/window_features.csv"),
        "vehicle_results": art("vehicle_results", "outputs/metrics/vehicle_level_results.csv"),
        "descriptors": art("anomaly_descriptors", "data/processed/anomaly_descriptors.csv"),
        "fleet_graph": art("fleet_graph", "data/processed/fleet_graph.pt"),
        "fleet_graph_graphml": art("fleet_graph_graphml", "outputs/fleet_graph.graphml"),
        "graph_stats": art("graph_stats", "outputs/metrics/fleet_graph_stats.json"),
        "gcn_embeddings": art("gcn_embeddings", "outputs/embeddings/gcn_node_embeddings.pt"),
        "gnn_metrics": art("gnn_metrics", "outputs/metrics/gnn_training_metrics.json"),
        "fleet_cluster_results": art("fleet_cluster_results", "data/processed/fleet_cluster_results.csv"),
        "final_detection_outcomes": art("final_detection_outcomes", "outputs/metrics/final_detection_outcomes.csv"),
        "final_outcome_summary": art("final_outcome_summary", "outputs/metrics/final_outcome_summary.csv"),
    }

    figures_dir = root / config.get("paths", {}).get("figures_dir", "outputs/figures")
    metrics_dir = root / config.get("paths", {}).get("metrics_dir", "outputs/metrics")
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config.get("project", {}).get("seed", 42))
    graph_cfg = config.get("graph", {})
    ids_cfg = config.get("vehicle_ids", {})
    generated: dict[str, str] = {}

    if paths["vehicle_results"].exists():
        results = pd.read_csv(paths["vehicle_results"])
        plot_vehicle_confusion_matrices(results, _fig_path(figures_dir, "confusion"))
        generated["confusion"] = str(_fig_path(figures_dir, "confusion"))

    if paths["window_features"].exists():
        plot_vehicle_roc_curves(
            paths["window_features"],
            _fig_path(figures_dir, "roc"),
            test_size=float(ids_cfg.get("test_size", 0.2)),
            random_state=seed,
            include_autoencoder=bool(ids_cfg.get("include_autoencoder", True)),
            ae_epochs=int(research_cfg.get("autoencoder_epochs_report", 15)),
        )
        generated["roc"] = str(_fig_path(figures_dir, "roc"))
        plot_feature_importance(paths["window_features"], _fig_path(figures_dir, "feature_importance"))
        generated["feature_importance"] = str(_fig_path(figures_dir, "feature_importance"))

    if paths["descriptors"].exists():
        plot_fleet_graph_visualization(
            paths["descriptors"],
            _fig_path(figures_dir, "graph"),
            graphml_path=paths["fleet_graph_graphml"],
            metric=str(graph_cfg.get("similarity_metric", "cosine")),
            threshold=float(graph_cfg.get("similarity_threshold", 0.85)),
            max_nodes=int(research_cfg.get("graph_viz_max_nodes", 1200)),
            seed=seed,
        )
        generated["graph"] = str(_fig_path(figures_dir, "graph"))

    if paths["descriptors"].exists() and paths["window_features"].exists():
        plot_gnn_embedding_tsne(
            paths["gcn_embeddings"],
            paths["descriptors"],
            paths["window_features"],
            _fig_path(figures_dir, "embedding_tsne"),
            seed=seed,
        )
        generated["embedding_tsne"] = str(_fig_path(figures_dir, "embedding_tsne"))

    assignments = ensure_clustering_outputs(
        config, paths, figures_dir, seed=seed, regenerate=regenerate_clustering
    )
    for key in ("cluster_kmeans", "cluster_dbscan", "cluster_summary"):
        p = _fig_path(figures_dir, key)
        if p.exists():
            generated[key] = str(p)

    tables = write_metrics_summary_tables(paths=paths, metrics_dir=metrics_dir, seed=seed)

    report_path = root / research_cfg.get("report_path", "outputs/experiment_report.md")
    generate_experiment_report(
        config=config,
        paths=paths,
        figures_dir=figures_dir,
        metrics_dir=metrics_dir,
        generated_figures=generated,
        output_path=report_path,
        assignments=assignments,
    )
    generated["report"] = str(report_path)
    return {"figures": generated, "tables": {k: str(v) for k, v in tables.items()}}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


_LEGACY_FIGURES = {
    "feature_corr": "feature_correlation_heatmap.png",
    "feature_dist": "feature_distributions.png",
    "confusion": "confusion_matrix_vehicle.png",
}


def _img(figures_dir: Path, root: Path, key: str) -> str:
    candidates = [_fig_path(figures_dir, key)]
    if key in _LEGACY_FIGURES:
        candidates.append(figures_dir / _LEGACY_FIGURES[key])
    for p in candidates:
        if p.exists():
            return f"![{key}]({_rel(p, root)})"
    return f"_Figure `{RESEARCH_FIGURES.get(key, key)}` not generated yet._"


def generate_experiment_report(
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    figures_dir: Path,
    metrics_dir: Path,
    generated_figures: dict[str, str],
    output_path: Path | str,
    assignments: pd.DataFrame | None = None,
) -> Path:
    """Write research narrative markdown at *output_path*."""
    root = figures_dir.parent.parent if figures_dir.name == "figures" else figures_dir.parent
    out = Path(output_path)
    seed = config.get("project", {}).get("seed", 42)
    feat_cfg = config.get("features", {})

    lines: list[str] = [
        "# Fleet-Aware CAN-Bus Intrusion Detection — Experiment Report",
        "",
        "Research summary for the end-to-end pipeline: per-vehicle IDS, anomaly descriptors, behavioural fleet graph, GNN / clustering, and final isolated-vs-coordinated anomaly classification.",
        "",
        f"**Project:** {config.get('project', {}).get('name', 'fleet-can-ids')}  ",
        f"**Random seed:** {seed}",
        "",
        "---",
        "",
        "## 1. Dataset",
        "",
        "We use the **Car Hacking / Car Track** CAN traffic corpus (Hyundai, Kia, Chevrolet) with labelled attack and benign drives. Raw traces are merged into a single standardized schema (`timestamp`, `can_id`, DLC, payload bytes, `label`, `attack_type`, `vehicle_model`, `source_file`).",
        "",
    ]

    ds_json = metrics_dir / "summary_dataset.json"
    if ds_json.exists():
        ds = json.loads(ds_json.read_text(encoding="utf-8"))
        lines.extend(
            [
                "| Statistic | Value |",
                "|-----------|-------|",
                f"| Vehicles | {', '.join(ds.get('vehicles', []))} |",
                f"| Attack types (excl. benign) | {ds.get('n_attack_types', '—')} |",
                f"| CAN frames | {ds.get('n_can_frames', 0):,} |",
                "",
            ]
        )

    lines.extend(
        [
            "## 2. Preprocessing",
            "",
            "Frames are sorted per vehicle and attack session. No cross-file windowing is applied. Validation checks (schema, DLC/byte ranges, duplicate keys) are recorded in `outputs/metrics/data_validation_report.csv`.",
            "",
            "## 3. Feature extraction",
            "",
            f"Sliding windows of **{feat_cfg.get('window_size', 100)}** frames with **{feat_cfg.get('overlap', 50)}**-frame overlap produce per-window behavioural descriptors: CAN ID entropy, DLC statistics, and byte mean/std (plus timing statistics stored but excluded from fleet similarity and campaign clustering).",
            "",
            _img(figures_dir, root, "feature_corr"),
            "",
            _img(figures_dir, root, "feature_dist"),
            "",
            _img(figures_dir, root, "feature_importance"),
            "",
            "## 4. Vehicle-level IDS",
            "",
            "Four models are trained **per OEM** on disjoint train/test splits (80/20 stratified): Random Forest and Logistic Regression (supervised), Isolation Forest and Autoencoder (benign-only anomaly detection).",
            "",
            _img(figures_dir, root, "confusion"),
            "",
            _img(figures_dir, root, "roc"),
            "",
        ]
    )

    ids_csv = metrics_dir / "summary_vehicle_ids.csv"
    if ids_csv.exists():
        lines.append("### Metrics summary (test split)\n")
        lines.append(_dataframe_to_markdown(pd.read_csv(ids_csv)))
        lines.append("")

    lines.extend(
        [
            "## 5. Descriptor abstraction",
            "",
            "Only rows with `is_anomaly = 1` from `data/processed/vehicle_anomaly_predictions.csv` are retained. Each descriptor stores `event_id`, vehicle, source file, attack type, anomaly score, and a JSON behavioural vector for graph construction.",
            "",
        ]
    )
    desc_json = metrics_dir / "summary_descriptors.json"
    if desc_json.exists():
        dst = json.loads(desc_json.read_text(encoding="utf-8"))
        lines.append(f"- **Total descriptors:** {dst.get('n_descriptors', 0):,}")
        if dst.get("by_vehicle"):
            lines.append("- **By vehicle:** " + ", ".join(f"{k}={v:,}" for k, v in dst["by_vehicle"].items()))
        lines.append("")

    graph_cfg = config.get("graph", {})
    lines.extend(
        [
            "## 6. Graph construction",
            "",
            f"Fleet graph **nodes** are anomaly descriptors; **edges** link pairs with cosine similarity ≥ **{graph_cfg.get('similarity_threshold', 0.85)}** on behavioural vectors. **No temporal edges** are used.",
            "",
            _img(figures_dir, root, "graph"),
            "",
        ]
    )
    if paths.get("graph_stats") and paths["graph_stats"].exists():
        gst = json.loads(paths["graph_stats"].read_text(encoding="utf-8"))
        lines.extend(
            [
                "| Graph metric | Value |",
                "|--------------|-------|",
                f"| Nodes | {int(gst.get('num_nodes', 0)):,} |",
                f"| Edges | {int(gst.get('num_edges', 0)):,} |",
                f"| Average degree | {gst.get('average_degree', 0):.4f} |",
                f"| Density | {gst.get('graph_density', 0):.6f} |",
                "",
            ]
        )

    gnn_cfg = config.get("gnn", {})
    use_gt = bool(gnn_cfg.get("use_ground_truth_labels", True))
    label_note = (
        "Node classification uses **ground-truth window labels** (`ground_truth_label`), "
        "not IDS predictions, so representation learning is not tied to per-vehicle detector bias."
        if use_gt
        else "Node classification uses IDS `predicted_label` (legacy mode)."
    )
    lines.extend(
        [
            "## 7. GNN learning",
            "",
            f"A **{gnn_cfg.get('architecture', 'gcn')}** encoder ({gnn_cfg.get('hidden_channels', 64)} hidden → {gnn_cfg.get('embedding_dim', 32)} embedding) is trained on the fleet graph. {label_note}",
            "",
            _img(figures_dir, root, "embedding_tsne"),
            "",
        ]
    )
    if paths.get("gnn_metrics") and paths["gnn_metrics"].exists():
        gm = json.loads(paths["gnn_metrics"].read_text(encoding="utf-8"))
        lines.append(
            f"- Best validation accuracy: **{gm.get('best_val_acc', 0):.4f}**  \n"
            f"- Final test accuracy: **{gm.get('final_test_acc', 0):.4f}**  \n"
            f"- Nodes / edges: **{int(gm.get('num_nodes', 0)):,}** / **{int(gm.get('num_edges', 0)):,}**"
        )
        lines.append("")
    else:
        lines.append(
            "_GNN metrics file not found — run `python experiments/07_train_gnn.py` after building the fleet graph._\n"
        )

    cluster_cfg = config.get("clustering", {})
    lines.extend(
        [
            "## 8. Campaign clustering",
            "",
            "**KMeans** and **DBSCAN** (PCA-reduced density space) group node embeddings. A cluster is flagged as a **suspicious multi-vehicle campaign** when it spans ≥ "
            f"**{cluster_cfg.get('min_vehicles', 2)}** vehicle models and mean intra-cluster cosine similarity ≥ **{cluster_cfg.get('similarity_threshold', 0.85)}**.",
            "",
            _img(figures_dir, root, "cluster_kmeans"),
            "",
            _img(figures_dir, root, "cluster_dbscan"),
            "",
            _img(figures_dir, root, "cluster_summary"),
            "",
        ]
    )

    sus_csv = metrics_dir / "summary_suspicious_campaigns.csv"
    if sus_csv.exists():
        lines.append("### Suspicious campaigns\n")
        lines.append(_dataframe_to_markdown(pd.read_csv(sus_csv)))
        lines.append("")

    final_csv = paths.get("final_detection_outcomes")
    final_summary = paths.get("final_outcome_summary")
    if final_summary and final_summary.exists():
        lines.extend(
            [
                "## 9. Final anomaly outcomes",
                "",
                "Each anomaly event is classified as either `Isolated anomaly` or `Fleet-level coordinated behavioural pattern`. A fleet-level outcome requires a cluster spanning more than one vehicle with high behavioural similarity; temporal proximity is not used.",
                "",
                _dataframe_to_markdown(pd.read_csv(final_summary)),
                "",
            ]
        )
    elif final_csv and final_csv.exists():
        lines.extend(
            [
                "## 9. Final anomaly outcomes",
                "",
                _dataframe_to_markdown(pd.read_csv(final_csv)),
                "",
            ]
        )

    lines.extend(
        [
            "## 10. Limitations",
            "",
            "- **OEM isolation:** IDS models do not share weights across manufacturers; fleet reasoning happens only after descriptor abstraction.",
            "- **Similarity graph cost:** All-pairs neighbourhood search scales with descriptor count; large fleets may require `graph.max_nodes` subsampling.",
            "- **Descriptor selection vs GNN labels:** IDS predictions still decide which windows become graph nodes (suspicious-only descriptors). GNN training uses **ground-truth labels** when `gnn.use_ground_truth_labels: true` (default). Re-run `05_generate_descriptors.py`, `06_build_graph.py`, and `07_train_gnn.py` after upgrading descriptor CSVs.",
            "- **Clustering:** Density parameters are sensitive; DBSCAN uses PCA for separation. Timing features are excluded from similarity but remain in raw window tables.",
            "- **Reproducibility:** Large artefacts (CSVs, `.pt` graphs) are gitignored; regenerate via `python experiments/run_full_pipeline.py`.",
            "",
            "## 11. Future work",
            "",
            "- Joint **cross-vehicle** representation learning with contrastive or graph-matching losses.",
            "- **Temporal** edges (session-aware) combined with behavioural similarity for attack progression analysis.",
            "- **Online** fleet scoring and incremental graph updates for deployment.",
            "- Explainability: GNN attention or SHAP on graph neighbourhoods for analyst-facing alerts.",
            "- Benchmarking against additional public CAN IDS datasets and hardware-in-the-loop traces.",
            "",
            "---",
            "",
            "### Generated artefacts",
            "",
        ]
    )
    for name, p in sorted(generated_figures.items()):
        lines.append(f"- **{name}:** `{p}`")
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote experiment report to %s", out)
    return out
