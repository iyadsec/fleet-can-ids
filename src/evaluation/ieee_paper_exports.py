"""Publication-quality exports for the IEEE Experimental Evaluation section (4 contributions)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

IEEE_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


@dataclass(frozen=True)
class IeeePaperOutputs:
    paper_dir: Path
    results_dir: Path
    tables_dir: Path
    figures_dir: Path
    source_results: Path
    source_figures: Path


def _df_to_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True, float_format="%.4f")
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            body.strip(),
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )


def _df_to_md(df: pd.DataFrame, title: str) -> str:
    cols = list(df.columns)
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def _copy_figure(src: Path, dst_base: Path) -> None:
    dst_base.parent.mkdir(parents=True, exist_ok=True)
    copied = False
    for ext in (".pdf", ".png"):
        if src.with_suffix(ext).exists():
            shutil.copy2(src.with_suffix(ext), dst_base.with_suffix(ext))
            copied = True
    if not copied and not any(dst_base.with_suffix(ext).exists() for ext in (".pdf", ".png")):
        raise FileNotFoundError(f"Figure source missing and no existing output: {src}")


def _load_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(f"Required result missing: {path}")
    return pd.read_csv(path)


def _export_table_bundle(
    df: pd.DataFrame,
    *,
    stem: str,
    caption: str,
    label: str,
    title: str,
    outputs: IeeePaperOutputs,
) -> dict[str, Path]:
    csv_path = outputs.results_dir / f"{stem}.csv"
    md_path = outputs.tables_dir / f"{stem}.md"
    tex_path = outputs.tables_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    md_path.write_text(_df_to_md(df, title), encoding="utf-8")
    tex_path.write_text(_df_to_tex(df, caption, label), encoding="utf-8")
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


def _build_table_01_vehicle_ids(src: Path) -> pd.DataFrame:
    metrics = _load_csv(src, "vehicle_level_metrics.csv")
    return metrics.round(4)


def _build_table_02_descriptor_security(src: Path) -> pd.DataFrame:
    raw = _load_csv(src, "descriptor_security_comparison_table.csv")
    rows = [
        {
            "Category": "Compactness",
            "Metric": "Average raw CAN window size (bytes)",
            "Value": float(raw.loc[raw["Metric"].str.contains("raw CAN window", case=False), "Value"].iloc[0]),
        },
        {
            "Category": "Compactness",
            "Metric": "Average descriptor size (bytes)",
            "Value": float(raw.loc[raw["Metric"].str.contains("transmitted descriptor", case=False), "Value"].iloc[0]),
        },
        {
            "Category": "Compactness",
            "Metric": "Compression ratio (×)",
            "Value": float(raw.loc[raw["Metric"] == "Compression ratio", "Value"].iloc[0]),
        },
        {
            "Category": "Compactness",
            "Metric": "Bandwidth reduction (%)",
            "Value": float(raw.loc[raw["Metric"] == "Bandwidth reduction (%)", "Value"].iloc[0]),
        },
        {
            "Category": "Fleet scalability",
            "Metric": "Fleet bandwidth @ 100 vehicles — raw (MB)",
            "Value": float(
                raw.loc[raw["Metric"].str.contains("100 vehicles — raw", case=False), "Value"].iloc[0]
            ),
        },
        {
            "Category": "Fleet scalability",
            "Metric": "Fleet bandwidth @ 100 vehicles — descriptor (MB)",
            "Value": float(
                raw.loc[raw["Metric"].str.contains("100 vehicles — descriptor", case=False), "Value"].iloc[0]
            ),
        },
        {
            "Category": "Security",
            "Metric": "Payload-statistic reconstruction R² (descriptor attacker)",
            "Value": float(
                raw.loc[raw["Metric"].str.contains("Payload-statistic reconstruction R² — descriptor", case=False), "Value"].iloc[0]
            ),
        },
        {
            "Category": "Security",
            "Metric": "Vehicle fingerprinting accuracy (descriptor)",
            "Value": float(
                raw.loc[raw["Metric"].str.contains("Vehicle fingerprinting accuracy — descriptor", case=False), "Value"].iloc[0]
            ),
        },
        {
            "Category": "Information disclosure",
            "Metric": "Raw payload bytes transmitted",
            "Value": "Exposed",
        },
        {
            "Category": "Information disclosure",
            "Metric": "Descriptor payload bytes transmitted",
            "Value": "Not transmitted",
        },
    ]
    return pd.DataFrame(rows)


def _build_table_01_attack_breakdown(src: Path) -> pd.DataFrame:
    df = _load_csv(src, "vehicle_level_by_attack_type.csv")
    return df.round(2)


def _build_table_03_cross_vehicle(src: Path) -> pd.DataFrame:
    df = _load_csv(src, "cross_vehicle_generalisation.csv")
    rf = df[df["classifier"] == "random_forest"].copy()
    return rf[
        [
            "train_vehicle_display",
            "test_vehicle_display",
            "roc_auc",
            "pr_auc",
            "recall",
            "precision",
            "f1",
        ]
    ].rename(
        columns={
            "train_vehicle_display": "Train Vehicle",
            "test_vehicle_display": "Test Vehicle",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "recall": "Recall",
            "precision": "Precision",
            "f1": "F1-score",
        }
    ).round(4)


def _build_table_03_agnostic_score(src: Path) -> pd.DataFrame:
    return _load_csv(src, "vehicle_agnostic_score.csv").round(4)


def _build_table_04_campaign_detection(src: Path) -> pd.DataFrame:
    from src.evaluation.campaign_detection_experiment import build_campaign_results_table

    per_type = _load_csv(src, "campaign_detection_by_type.csv")
    metrics_raw = _load_csv(src, "campaign_detection_metrics.csv")
    metrics = {str(r["metric"]): float(r["value"]) for _, r in metrics_raw.iterrows()}
    return build_campaign_results_table(per_type, metrics)


def _build_table_04_local_vs_fleet_campaign(src: Path) -> pd.DataFrame:
    return _load_csv(src, "local_vs_fleet_campaign_comparison.csv")


def _ensure_campaign_detection_results(repo_root: Path) -> None:
    """Run campaign detection if IEEE Contribution 4 artifacts are missing."""
    required = repo_root / "results" / "campaign_ground_truth.csv"
    if required.exists():
        return
    logger.info("Campaign detection results missing; running evaluation before IEEE export.")
    from src.evaluation.campaign_detection_experiment import (
        CampaignDetectionConfig,
        CampaignDetectionOutputs,
        run_campaign_detection_experiment,
    )
    from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
    from src.utils.config import get_nested, load_config
    from src.utils.paths import ProjectPaths

    paths = ProjectPaths.from_root(repo_root)
    config = load_config(repo_root / "configs" / "fleet_ids.yaml")
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    cd = config.get("campaign_detection", {})
    fg = parse_fleet_graph_similarity_settings(config)
    fleet_graph_cfg = config.get("fleet_graph", {})
    seed = int(get_nested(config, "project", "seed", default=42))
    cfg = CampaignDetectionConfig(
        top_k_same_vehicle=int(cd.get("top_k_same_vehicle", fleet_graph_cfg.get("top_k_same_vehicle", 10))),
        top_k_cross_vehicle=int(cd.get("top_k_cross_vehicle", fleet_graph_cfg.get("top_k_cross_vehicle", 5))),
        similarity_threshold=float(cd.get("similarity_threshold", fleet_graph_cfg.get("similarity_threshold", 0.95))),
        similarity_feature_view=fg["similarity_feature_view"],
        feature_dominance_threshold=float(fg["feature_dominance_threshold"]),
        allowed_high_dominance_features=fg["allowed_high_dominance_features"],
        min_vehicles=int(cd.get("min_vehicles", 2)),
        min_cluster_size=int(cd.get("min_cluster_size", 10)),
        min_cohesion=float(cd.get("min_cohesion", 0.85)),
        min_dominant_attack_ratio=float(cd.get("min_dominant_attack_ratio", 0.60)),
        campaign_match_recall=float(cd.get("campaign_match_recall", 0.10)),
        campaign_match_min_nodes=int(cd.get("campaign_match_min_nodes", 15)),
        dbscan_eps=float(cd.get("dbscan_eps", config.get("clustering", {}).get("dbscan_eps", 1.2))),
        dbscan_min_samples=int(cd.get("dbscan_min_samples", config.get("clustering", {}).get("dbscan_min_samples", 10))),
        dbscan_pca_components=int(cd.get("dbscan_pca_components", config.get("clustering", {}).get("dbscan_pca_components", 8))),
        max_clustering_samples=int(cd.get("max_clustering_samples", config.get("clustering", {}).get("max_clustering_samples", 20000))),
        max_graph_viz_nodes=int(cd.get("max_graph_viz_nodes", 800)),
        max_embedding_samples=int(cd.get("max_embedding_samples", 5000)),
        embedding_method=str(cd.get("embedding_method", "tsne")),  # type: ignore[arg-type]
        seed=seed,
    )
    run_campaign_detection_experiment(
        descriptors_path=paths.root / artifacts.get("anomaly_descriptors", "data/processed/anomaly_descriptors.csv"),
        features_path=paths.root / artifacts.get("window_features", "data/processed/window_features.csv"),
        outputs=CampaignDetectionOutputs(
            results_dir=paths.root / str(pub.get("results_dir", "results")),
            tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
            figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
        ),
        cfg=cfg,
    )


def _copy_campaign_supporting_results(src: Path, outputs: IeeePaperOutputs) -> dict[str, Path]:
    """Mirror campaign CSVs into the paper bundle."""
    names = [
        "campaign_ground_truth.csv",
        "campaign_graph_statistics.csv",
        "detected_campaign_clusters.csv",
        "campaign_detection_metrics.csv",
        "campaign_detection_by_type.csv",
        "local_vs_fleet_campaign_comparison.csv",
        "campaign_detection_summary.md",
    ]
    copied: dict[str, Path] = {}
    for name in names:
        src_path = src / name
        if src_path.exists():
            dst = outputs.results_dir / name
            shutil.copy2(src_path, dst)
            copied[name] = dst
    return copied


def _write_interpretations(path: Path, sections: dict[str, str]) -> None:
    lines = [
        "# IEEE Experimental Evaluation — Interpretations",
        "",
        "This document supports the four validated contributions in the Experimental Evaluation section.",
        "Claims are limited to what the exported evidence supports.",
        "",
    ]
    for title, body in sections.items():
        lines.extend([f"## {title}", "", body.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ieee_paper_exports(*, repo_root: Path) -> dict[str, Any]:
    plt.rcParams.update(IEEE_RC)
    _ensure_campaign_detection_results(repo_root)
    src = repo_root / "results"
    src_fig = repo_root / "figures"
    paper = repo_root / "paper"
    outputs = IeeePaperOutputs(
        paper_dir=paper,
        results_dir=paper / "results",
        tables_dir=paper / "tables",
        figures_dir=paper / "figures",
        source_results=src,
        source_figures=src_fig,
    )
    for d in (outputs.results_dir, outputs.tables_dir, outputs.figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    written: dict[str, Any] = {}

    # --- Contribution 1: Vehicle-Level IDS ---
    t1 = _build_table_01_vehicle_ids(src)
    written["table_01"] = _export_table_bundle(
        t1,
        stem="table_01_vehicle_level_ids",
        caption="Vehicle-Level IDS Performance (Isolation Forest, FPR$\\leq$5\\% threshold).",
        label="tab:vehicle-level-ids",
        title="Table 1: Vehicle-Level IDS Performance",
        outputs=outputs,
    )
    t1_attack = _build_table_01_attack_breakdown(src)
    t1_attack.to_csv(outputs.results_dir / "table_01_vehicle_level_by_attack.csv", index=False)

    _copy_figure(src_fig / "local_ids_roc_curve", outputs.figures_dir / "figure_01_vehicle_ids_roc")
    _copy_figure(src_fig / "local_ids_anomaly_score_distribution", outputs.figures_dir / "figure_02_vehicle_ids_score_distribution")

    # --- Contribution 2: Descriptor Compactness and Security ---
    t2 = _build_table_02_descriptor_security(src)
    written["table_02"] = _export_table_bundle(
        t2,
        stem="table_02_descriptor_compactness_security",
        caption="Descriptor compactness, fleet scalability, and security/privacy metrics.",
        label="tab:descriptor-security",
        title="Table 2: Descriptor Compactness and Security",
        outputs=outputs,
    )
    _copy_figure(src_fig / "raw_vs_descriptor_exposure", outputs.figures_dir / "figure_03_descriptor_bandwidth_exposure")
    _copy_figure(src_fig / "payload_reconstruction_error", outputs.figures_dir / "figure_04_payload_reconstruction_risk")

    # --- Contribution 3: Cross-Vehicle Generalisation ---
    t3 = _build_table_03_cross_vehicle(src)
    written["table_03"] = _export_table_bundle(
        t3,
        stem="table_03_cross_vehicle_generalisation",
        caption="Cross-vehicle descriptor generalisation (Random Forest, leave-one-vehicle-out).",
        label="tab:cross-vehicle-generalisation",
        title="Table 3: Cross-Vehicle Descriptor Generalisation",
        outputs=outputs,
    )
    agnostic = _build_table_03_agnostic_score(src)
    agnostic.to_csv(outputs.results_dir / "table_03_vehicle_agnostic_score.csv", index=False)
    _copy_figure(src_fig / "descriptor_embedding_by_attack", outputs.figures_dir / "figure_05_cross_vehicle_embedding")

    # --- Contribution 4: Fleet Campaign Detection ---
    t4 = _build_table_04_campaign_detection(src)
    written["table_04"] = _export_table_bundle(
        t4,
        stem="table_04_campaign_detection_results",
        caption="Coordinated campaign detection on controlled multi-vehicle attack scenarios "
        "(behaviour-normalized descriptor similarity graph; scenarios constructed from labelled windows).",
        label="tab:campaign-detection-results",
        title="Table 4: Coordinated Campaign Detection Results",
        outputs=outputs,
    )
    t4b = _build_table_04_local_vs_fleet_campaign(src)
    written["table_04_local_vs_fleet"] = _export_table_bundle(
        t4b,
        stem="table_04_local_vs_fleet_campaign_detection",
        caption="Local IDS vs fleet-aware coordinated campaign detection.",
        label="tab:local-vs-fleet-campaign",
        title="Table 4 (supplementary): Local IDS vs Fleet-Aware Campaign Detection",
        outputs=outputs,
    )
    written["campaign_supporting"] = _copy_campaign_supporting_results(src, outputs)
    _copy_figure(src_fig / "fleet_campaign_graph", outputs.figures_dir / "figure_06_fleet_campaign_graph")
    _copy_figure(
        src_fig / "campaign_descriptor_embedding",
        outputs.figures_dir / "figure_07_campaign_descriptor_embedding",
    )
    written["figure_06"] = outputs.figures_dir / "figure_06_fleet_campaign_graph.pdf"
    written["figure_07"] = outputs.figures_dir / "figure_07_campaign_descriptor_embedding.pdf"

    # Mirror numbered tables to repo tables/ for convenience
    mirror_tables = repo_root / "tables"
    mirror_tables.mkdir(exist_ok=True)
    for key in ("table_01", "table_02", "table_03", "table_04"):
        for fmt in ("csv", "md", "tex"):
            p = written[key][fmt if fmt != "csv" else "csv"]
            if fmt == "csv":
                shutil.copy2(p, mirror_tables / p.name.replace("table_0", "ieee_table_0"))
            else:
                shutil.copy2(p, mirror_tables / p.name)
    for fmt in ("csv", "md", "tex"):
        p = written["table_04_local_vs_fleet"][fmt if fmt != "csv" else "csv"]
        if fmt == "csv":
            dst = mirror_tables / "ieee_table_04_local_vs_fleet_campaign_detection.csv"
        else:
            dst = mirror_tables / p.name
        shutil.copy2(p, dst)

    interpretations = {
        "Contribution 1 — Vehicle-Level IDS Effectiveness": """
The self-supervised Isolation Forest achieves **ROC-AUC 0.786** and **PR-AUC 0.927** at an operating point selected for **FPR ≤ 5%**.
Precision is high (**97.3%**) but recall is moderate (**46.0%**, F1 **62.4%**), indicating conservative strong-alert generation.
Per-attack F1 ranges from **39.8% (replay)** to **81.3% (fuzzy)**; replay remains the hardest class at the chosen threshold.

**Interpretation:** The vehicle-level IDS provides a usable local baseline with low false alarms, but cannot group cross-vehicle attack behaviour into coordinated campaigns.
**Limitation:** Threshold selection trades recall for FPR; campaign-level reasoning requires the fleet correlation layer.
        """,
        "Contribution 2 — Descriptor Compactness and Security": """
Descriptors compress raw CAN windows by **12.6×** (**92%** bandwidth reduction), with **94%** fleet bandwidth reduction at 100 vehicles.
Frame-level CAN IDs, payloads, and exact message order are not transmitted; only aggregated behavioural statistics and anomaly evidence are uplinked.

Payload-statistic reconstruction from descriptors yields **R² ≈ 0.44**, well below the raw-CAN baseline (**R² = 1.0**), indicating limited inference of payload-derived statistics from the uplink alone.
Vehicle fingerprinting remains high (**≈99.97%**) on behavioural descriptors — **anonymisation is not fully achieved**.

**Interpretation:** The descriptor layer substantially reduces data exposure and communication cost while preserving anomaly evidence.
**Limitation:** Residual vehicle-specific patterns remain; privacy-hardening is future work.
        """,
        "Contribution 3 — Vehicle-Agnostic Descriptor Generalisation": """
Leave-one-vehicle-out transfer (Random Forest on behavioural descriptor features) yields mean **ROC-AUC 0.745**, mean **F1 0.896**, and vehicle-agnostic score **0.821**.
Transfer is strongest for Kia→Chevrolet (**ROC-AUC 0.856**) and weakest when Chevrolet is the training platform (small sample size).

Cross-vehicle attack similarity gaps are smallest for **replay** and **fuzzy** (≈0.03 cosine gap), supporting behavioural alignment across platforms.
Embeddings (Figure 5) show attack-type structure spanning multiple vehicle markers.

**Interpretation:** Descriptor features encode attack behaviour that transfers across heterogeneous vehicles, supporting fleet deployment.
**Limitation:** ROC-AUC is moderate for linear models (0.60); Chevrolet's smaller corpus limits some pairs; descriptors still permit vehicle classification.
        """,
        "Contribution 4 — Fleet Campaign Detection": """
Controlled campaign scenarios were constructed from labelled attack windows across **four attack types** (flooding, fuzzy, replay, malfunction) spanning **2–3 vehicles** each.
These scenarios evaluate fleet-level campaign reasoning; they **do not** represent externally synchronized real-world campaigns.

The behaviour-normalized fleet graph achieves **≈42%** cross-vehicle edges.
DBSCAN clustering on behavioural descriptors yields **one valid cross-vehicle campaign cluster** (flooding, Hyundai+Kia, purity **100%**, mean similarity **0.99**).
Overall **campaign detection rate is 25%** (1/4 scenarios), with **campaign precision 100%** and **false campaign rate 0%** under current gates.
Fuzzy, replay, and malfunction campaigns were **not** recovered as distinct multi-vehicle clusters at the chosen similarity/cohesion thresholds.

Local IDS retains the same per-window attack recall (**≈79%** on campaign windows) but **cannot** perform campaign-level detection (**0%** vs **25%** fleet scenario detection rate).

**Interpretation:** The fleet-aware correlation layer enables campaign-level detection by grouping behaviourally similar anomaly descriptors across multiple vehicles — a capability unavailable to isolated vehicle-level IDS models.
**Limitation:** Detection is strongest for flooding; other attack types overlap behaviourally or form larger mixed clusters; campaign scenarios are synthetically defined from the public dataset labels.
        """,
    }
    interp_path = outputs.results_dir / "ieee_experimental_evaluation_interpretations.md"
    _write_interpretations(interp_path, interpretations)
    written["interpretations"] = interp_path

    index = paper / "IEEE_EXPERIMENTAL_EVALUATION_INDEX.md"
    index.write_text(
        "\n".join(
            [
                "# IEEE Experimental Evaluation — Export Index",
                "",
                "## Tables",
                "| Table | File | Contribution |",
                "|-------|------|--------------|",
                "| Table 1 | `paper/tables/table_01_vehicle_level_ids.tex` | Vehicle-Level IDS |",
                "| Table 2 | `paper/tables/table_02_descriptor_compactness_security.tex` | Descriptor Compactness & Security |",
                "| Table 3 | `paper/tables/table_03_cross_vehicle_generalisation.tex` | Cross-Vehicle Generalisation |",
                "| Table 4 | `paper/tables/table_04_campaign_detection_results.tex` | Fleet Campaign Detection |",
                "| Table 4 (supp.) | `paper/tables/table_04_local_vs_fleet_campaign_detection.tex` | Fleet Campaign Detection |",
                "",
                "## Figures",
                "| Figure | File | Contribution |",
                "|--------|------|--------------|",
                "| Figure 1 | `paper/figures/figure_01_vehicle_ids_roc.pdf` | Vehicle-Level IDS |",
                "| Figure 2 | `paper/figures/figure_02_vehicle_ids_score_distribution.pdf` | Vehicle-Level IDS |",
                "| Figure 3 | `paper/figures/figure_03_descriptor_bandwidth_exposure.pdf` | Descriptor Security |",
                "| Figure 4 | `paper/figures/figure_04_payload_reconstruction_risk.pdf` | Descriptor Security |",
                "| Figure 5 | `paper/figures/figure_05_cross_vehicle_embedding.pdf` | Cross-Vehicle Generalisation |",
                "| Figure 6 | `paper/figures/figure_06_fleet_campaign_graph.pdf` | Fleet Campaign Detection |",
                "| Figure 7 | `paper/figures/figure_07_campaign_descriptor_embedding.pdf` | Fleet Campaign Detection |",
                "",
                "## Supporting CSVs",
                "All under `paper/results/`.",
                "",
                "## Interpretations",
                "`paper/results/ieee_experimental_evaluation_interpretations.md`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    written["index"] = index

    logger.info("IEEE paper exports written to %s", paper)
    return written
