"""Run the full Fleet CAN-IDS experiment pipeline from YAML configuration."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.data.dataset_loader import load_and_merge, print_dataset_statistics, save_clean_dataset
from src.evaluation.campaign_clustering import (
    load_embedding_table,
    plot_cluster_summaries,
    plot_tsne_clusters,
    print_cluster_report,
    run_campaign_clustering,
    save_campaign_clusters,
)
from src.evaluation.pipeline_report import generate_pipeline_report
from src.features.descriptor_generator import (
    generate_anomaly_descriptors,
    load_or_generate_predictions,
    print_descriptor_summary,
    save_anomaly_descriptors,
)
from src.features.feature_extractor import (
    extract_features,
    load_frames,
    load_windows,
    plot_feature_correlation_heatmap,
    plot_feature_distributions,
    print_feature_summary,
    save_window_features,
)
from src.features.window_generator import (
    generate_windows,
    load_can_frames,
    print_window_statistics,
    save_window_metadata,
)
from src.graph.fleet_graph_builder import (
    build_fleet_anomaly_graph,
    load_anomaly_descriptors,
    print_graph_statistics,
    save_fleet_graph,
)
from src.models.vehicle_ids import (
    load_feature_dataset,
    plot_vehicle_confusion_matrices,
    print_results_summary,
    run_vehicle_level_training,
    save_results,
)
from src.utils.config import get_nested
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)

PIPELINE_STEPS = (
    "load_dataset",
    "generate_windows",
    "extract_features",
    "train_vehicle_ids",
    "generate_descriptors",
    "build_fleet_graph",
    "train_gnn",
    "cluster_campaigns",
    "generate_report",
)


class FullPipelineRunner:
    """Orchestrate dataset → IDS → graph → GNN → clustering → report."""

    def __init__(self, config: dict[str, Any], paths: ProjectPaths) -> None:
        self.config = config
        self.paths = paths
        self.root = paths.root
        self.seed = int(get_nested(config, "project", "seed", default=42))
        self.artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
        self.pipeline_cfg = get_nested(config, "pipeline", default={}) or {}
        self.step_results: dict[str, dict[str, Any]] = {}

    def artifact(self, key: str, default: str) -> Path:
        rel = self.artifacts.get(key, default)
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def _should_skip(self, path: Path) -> bool:
        if not self.pipeline_cfg.get("skip_if_outputs_exist", False):
            return False
        return path.exists() and path.stat().st_size > 0

    def _record(self, step: str, status: str, t0: float, **extra: Any) -> None:
        self.step_results[step] = {
            "status": status,
            "elapsed_sec": time.perf_counter() - t0,
            **extra,
        }

    def run(self, steps: list[str] | None = None) -> int:
        selected = steps or list(
            self.pipeline_cfg.get("steps", PIPELINE_STEPS)
        )
        stop_on_error = bool(self.pipeline_cfg.get("stop_on_error", True))

        for step in selected:
            if step not in PIPELINE_STEPS:
                logger.warning("Unknown pipeline step: %s (skipped)", step)
                continue
            handler = getattr(self, f"_step_{step}", None)
            if handler is None:
                logger.error("No handler for step: %s", step)
                return 1
            logger.info("========== Step: %s ==========", step)
            t0 = time.perf_counter()
            try:
                code = handler()
            except Exception as exc:
                logger.exception("Step %s failed: %s", step, exc)
                self._record(step, "failed", t0, error=str(exc))
                if stop_on_error:
                    return 1
                continue
            if code != 0:
                self._record(step, "failed", t0, exit_code=code)
                if stop_on_error:
                    return code
            else:
                self._record(step, "ok", t0)

        return 0

    def _step_load_dataset(self) -> int:
        out = self.artifact("clean_can_data", "data/processed/clean_can_data.csv")
        if self._should_skip(out):
            logger.info("Skipping load_dataset — output exists: %s", out)
            return 0

        data_cfg = self.config.get("data", {})
        external = Path(
            data_cfg.get(
                "external_dataset_dir",
                "/Users/iyadatieh/Library/CloudStorage/OneDrive-Personal/"
                "University of Reading/CodeRepo/Dataset",
            )
        )
        raw_root = self.root / get_nested(self.config, "paths", "raw_dir", default="data/raw")
        df = load_and_merge(external_root=external, raw_root=raw_root)
        if df.empty:
            logger.error("No data loaded.")
            return 1
        save_clean_dataset(df, out)
        print_dataset_statistics(df)
        return 0

    def _step_generate_windows(self) -> int:
        out = self.artifact("window_metadata", "data/processed/window_metadata.csv")
        if self._should_skip(out):
            logger.info("Skipping generate_windows — output exists: %s", out)
            return 0

        frames_path = self.artifact("clean_can_data", "data/processed/clean_can_data.csv")
        frames = load_can_frames(frames_path)
        if frames.empty:
            return 1
        meta = generate_windows(frames, self.config)
        if meta.empty:
            return 1
        save_window_metadata(meta, out)
        print_window_statistics(meta)
        return 0

    def _step_extract_features(self) -> int:
        out = self.artifact("window_features", "data/processed/window_features.csv")
        if self._should_skip(out):
            logger.info("Skipping extract_features — output exists: %s", out)
            return 0

        frames = load_frames(self.artifact("clean_can_data", "data/processed/clean_can_data.csv"))
        windows = load_windows(
            self.artifact("window_metadata", "data/processed/window_metadata.csv")
        )
        features = extract_features(frames, windows)
        if features.empty:
            return 1
        save_window_features(features, out)
        print_feature_summary(features)

        feat_cfg = self.config.get("features", {})
        if feat_cfg.get("generate_plots", True):
            plot_feature_correlation_heatmap(
                features, self.paths.figures_dir / "feature_correlation_heatmap.png"
            )
            plot_feature_distributions(
                features, self.paths.figures_dir / "feature_distributions.png"
            )
        return 0

    def _step_train_vehicle_ids(self) -> int:
        out = self.artifact("vehicle_results", "outputs/metrics/vehicle_level_results.csv")
        if self._should_skip(out):
            logger.info("Skipping train_vehicle_ids — output exists: %s", out)
            return 0

        ids_cfg = self.config.get("vehicle_ids", {})
        features_path = self.artifact("window_features", "data/processed/window_features.csv")
        results = run_vehicle_level_training(
            features_path,
            test_size=float(ids_cfg.get("test_size", 0.2)),
            random_state=self.seed,
            include_autoencoder=bool(ids_cfg.get("include_autoencoder", True)),
        )
        if results.empty:
            return 1
        save_results(results, out)
        plot_vehicle_confusion_matrices(
            results, self.paths.figures_dir / "confusion_matrix_vehicle.png"
        )
        print_results_summary(results)
        return 0

    def _step_generate_descriptors(self) -> int:
        out = self.artifact("anomaly_descriptors", "data/processed/anomaly_descriptors.csv")
        if self._should_skip(out):
            logger.info("Skipping generate_descriptors — output exists: %s", out)
            return 0

        desc_cfg = self.config.get("descriptors", {})
        features_path = self.artifact("window_features", "data/processed/window_features.csv")
        predictions_path = self.artifact(
            "window_predictions", "outputs/metrics/window_predictions.csv"
        )
        features = load_feature_dataset(features_path)
        predictions = load_or_generate_predictions(
            features,
            predictions_path,
            random_state=self.seed,
            test_size=float(
                desc_cfg.get("test_size", self.config.get("vehicle_ids", {}).get("test_size", 0.2))
            ),
            include_autoencoder=bool(
                desc_cfg.get(
                    "include_autoencoder",
                    self.config.get("vehicle_ids", {}).get("include_autoencoder", True),
                )
            ),
            regenerate=bool(desc_cfg.get("regenerate_predictions", False)),
        )
        descriptors = generate_anomaly_descriptors(
            features,
            predictions,
            primary_model=str(desc_cfg.get("primary_model", "random_forest")),
            score_threshold=float(desc_cfg.get("score_threshold", 0.5)),
        )
        if descriptors.empty:
            logger.error("No anomaly descriptors produced.")
            return 1
        save_anomaly_descriptors(descriptors, out)
        print_descriptor_summary(descriptors)
        return 0

    def _step_build_fleet_graph(self) -> int:
        pt_out = self.artifact("fleet_graph", "data/processed/fleet_graph.pt")
        if self._should_skip(pt_out):
            logger.info("Skipping build_fleet_graph — output exists: %s", pt_out)
            return 0

        graph_cfg = self.config.get("graph", {})
        desc_path = self.artifact("anomaly_descriptors", "data/processed/anomaly_descriptors.csv")
        descriptors = load_anomaly_descriptors(desc_path)
        max_nodes = graph_cfg.get("max_nodes")
        G, pyg_data, stats, _ = build_fleet_anomaly_graph(
            descriptors,
            metric=graph_cfg.get("similarity_metric", "cosine"),  # type: ignore[arg-type]
            threshold=float(graph_cfg.get("similarity_threshold", 0.85)),
            max_nodes=int(max_nodes) if max_nodes else None,
            seed=self.seed,
        )
        graphml = self.artifact("fleet_graph_graphml", "outputs/fleet_graph.graphml")
        save_fleet_graph(G, pyg_data, stats, pt_path=pt_out, graphml_path=graphml)
        stats_path = self.artifact("graph_stats", "outputs/metrics/fleet_graph_stats.json")
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print_graph_statistics(stats)
        return 0

    def _step_train_gnn(self) -> int:
        emb_out = self.artifact("gcn_embeddings", "outputs/embeddings/gcn_node_embeddings.pt")
        if self._should_skip(emb_out):
            logger.info("Skipping train_gnn — output exists: %s", emb_out)
            return 0

        graph_path = self.artifact("fleet_graph", "data/processed/fleet_graph.pt")
        if not graph_path.exists():
            logger.error("Fleet graph missing: %s", graph_path)
            return 1

        from src.models.gnn_models import train_gnn_from_graph_file

        gnn_cfg = self.config.get("gnn", {})
        ckpt_dir = self.artifact("gnn_checkpoints", "outputs/checkpoints")
        metrics = train_gnn_from_graph_file(
            graph_path,
            emb_out,
            ckpt_dir,
            config=gnn_cfg,
            seed=self.seed,
        )
        metrics_path = self.artifact("gnn_metrics", "outputs/metrics/gnn_training_metrics.json")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return 0

    def _step_cluster_campaigns(self) -> int:
        out = self.artifact("campaign_clusters", "outputs/metrics/campaign_clusters.csv")
        cluster_cfg = self.config.get("clustering", {})

        emb_path = self.artifact("gcn_embeddings", "outputs/embeddings/gcn_node_embeddings.pt")
        desc_path = self.artifact("anomaly_descriptors", "data/processed/anomaly_descriptors.csv")
        feat_path = self.artifact("window_features", "data/processed/window_features.csv")

        X, meta = load_embedding_table(emb_path, desc_path, features_path=feat_path)
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
            random_state=self.seed,
            max_clustering_samples=int(max_samples) if max_samples else None,
        )
        save_campaign_clusters(assignments, out)

        for algo in ("kmeans", "dbscan"):
            sub = assignments[assignments["algorithm"] == algo]
            plot_tsne_clusters(
                X,
                sub,
                meta,
                self.paths.figures_dir / f"campaign_tsne_{algo}.png",
                algorithm=algo,
                seed=self.seed,
            )
        plot_cluster_summaries(assignments, self.paths.figures_dir / "campaign_cluster_summary.png")
        print_cluster_report(assignments)
        return 0

    def _step_generate_report(self) -> int:
        report_path = self.artifact("report", "outputs/metrics/pipeline_report.md")
        cfg = dict(self.config)
        cfg["_project_root"] = str(self.root)
        generate_pipeline_report(
            config=cfg,
            step_results=self.step_results,
            output_path=report_path,
        )
        return 0
