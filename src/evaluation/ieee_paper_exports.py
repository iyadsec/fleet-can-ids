"""Publication-quality exports for the IEEE Experimental Evaluation section (H1–H4)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.descriptor_security_experiment import PAPER_FIGURE_CAPTIONS, export_h2_publication_figures
from src.utils.logging import get_logger

IEEE_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

OBSOLETE_PAPER_STEMS = (
    "figure_01_vehicle_ids_roc",
    "figure_02_vehicle_ids_score_distribution",
    "figure_02_vehicle_level_roc",
    "figure_03_anomaly_score_distribution",
    "figure_03_local_ids_f1_by_attack",
    "figure_03_descriptor_bandwidth_exposure",
    "figure_04_payload_reconstruction_risk",
    "figure_05_payload_reconstruction_risk",
    "figure_05_cross_vehicle_embedding",
    "figure_06_fleet_campaign_graph",
    "figure_07_campaign_descriptor_embedding",
    "figure_09_campaign_detection_by_attack_type",
    "table_04_campaign_detection_results",
    "table_04_local_vs_fleet_campaign_detection",
)

LEGACY_CAMPAIGN_RESULT_FILES = (
    "campaign_ground_truth.csv",
    "campaign_graph_statistics.csv",
    "detected_campaign_clusters.csv",
    "campaign_detection_metrics.csv",
    "campaign_detection_by_type.csv",
    "local_vs_fleet_campaign_comparison.csv",
    "campaign_detection_summary.md",
)


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
    if not copied:
        raise FileNotFoundError(f"Figure source missing: {src}")


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


def _cleanup_obsolete_paper_artifacts(outputs: IeeePaperOutputs) -> None:
    for stem in OBSOLETE_PAPER_STEMS:
        for folder in (outputs.figures_dir, outputs.tables_dir, outputs.results_dir):
            for ext in (".pdf", ".png", ".md", ".tex", ".csv"):
                path = folder / f"{stem}{ext}"
                if path.exists():
                    path.unlink()
    for name in LEGACY_CAMPAIGN_RESULT_FILES:
        path = outputs.results_dir / name
        if path.exists():
            path.unlink()


def _build_table_01_vehicle_ids(src: Path) -> pd.DataFrame:
    return _load_csv(src, "vehicle_level_metrics.csv").round(4)


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


def _build_table_04_local_vs_gnn_fleet(src: Path) -> pd.DataFrame:
    df = _load_csv(src, "table_final_local_vs_gnn_fleet_ids.csv")
    numeric_cols = ["Local IDS", "GNN-Based Fleet IDS"]
    out = df.copy()
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.round(4)


def _build_table_05_coordinated_campaigns(src: Path) -> pd.DataFrame:
    return _load_csv(src, "table_final_campaign_detection_by_attack_type.csv").round(4)


def _ensure_prerequisite_experiments(repo_root: Path) -> None:
    """Run upstream experiments when required CSVs or figures are missing."""
    src = repo_root / "results"
    fig = repo_root / "figures"
    config_path = repo_root / "configs" / "fleet_ids.yaml"

    need_vehicle = not (src / "vehicle_level_metrics.csv").exists() or not (
        fig / "local_ids_pr_curve.pdf"
    ).exists()
    need_descriptor = not (src / "descriptor_security_comparison_table.csv").exists() or not (
        fig / "information_disclosure_comparison.pdf"
    ).exists() or not (fig / "bandwidth_scaling_fleet_sizes.pdf").exists()
    need_cross = not (src / "cross_vehicle_generalisation.csv").exists() or not (
        fig / "descriptor_embedding_by_attack.pdf"
    ).exists()

    if need_vehicle:
        logger.info("Running vehicle-level evaluation for IEEE H1 artifacts.")
        import run_vehicle_level_evaluation

        run_vehicle_level_evaluation.main()
    if need_descriptor:
        logger.info("Running descriptor security experiment for IEEE H2 artifacts.")
        import run_descriptor_security_experiment

        run_descriptor_security_experiment.main()
    if need_cross:
        logger.info("Running cross-vehicle generalisation for IEEE H3 artifacts.")
        import run_cross_vehicle_generalisation

        run_cross_vehicle_generalisation.main()


def _ensure_final_gnn_fleet_decision_results(repo_root: Path) -> None:
    """Run final GNN fleet decision pipeline if artifacts are missing."""
    required = repo_root / "results" / "final_attack_decisions.csv"
    if required.exists():
        return
    logger.info("Final GNN fleet decision results missing; running evaluation before IEEE export.")
    from src.evaluation.final_gnn_fleet_decision_experiment import (
        FinalGnnFleetConfig,
        FinalGnnFleetOutputs,
        run_final_gnn_fleet_decision_experiment,
    )
    from src.graph.fleet_similarity_features import parse_fleet_graph_similarity_settings
    from src.utils.config import get_nested, load_config
    from src.utils.paths import ProjectPaths

    paths = ProjectPaths.from_root(repo_root)
    config = load_config(repo_root / "configs" / "fleet_ids.yaml")
    pub = config.get("publication", {})
    artifacts = get_nested(config, "pipeline", "artifacts", default={}) or {}
    fg = parse_fleet_graph_similarity_settings(config)
    fleet_graph_cfg = config.get("fleet_graph", {})
    gnn_cfg = config.get("gnn", {})
    fd = config.get("final_gnn_fleet_decision", {})
    seed = int(get_nested(config, "project", "seed", default=42))
    ckpt = fd.get("checkpoint_path", "outputs/models/final_graphsage_fleet.pt")
    cfg = FinalGnnFleetConfig(
        top_k_same_vehicle=int(fd.get("top_k_same_vehicle", fleet_graph_cfg.get("top_k_same_vehicle", 10))),
        top_k_cross_vehicle=int(fd.get("top_k_cross_vehicle", fleet_graph_cfg.get("top_k_cross_vehicle", 5))),
        similarity_threshold=float(fd.get("similarity_threshold", fleet_graph_cfg.get("similarity_threshold", 0.95))),
        feature_dominance_threshold=float(fg["feature_dominance_threshold"]),
        gnn_hidden_channels=int(fd.get("gnn_hidden_channels", gnn_cfg.get("hidden_channels", 64))),
        gnn_embedding_dim=int(fd.get("gnn_embedding_dim", gnn_cfg.get("embedding_dim", 32))),
        gnn_epochs=int(fd.get("gnn_epochs", gnn_cfg.get("epochs", 30))),
        gnn_learning_rate=float(fd.get("gnn_learning_rate", gnn_cfg.get("learning_rate", 0.01))),
        gnn_weight_decay=float(fd.get("gnn_weight_decay", gnn_cfg.get("weight_decay", 5e-4))),
        campaign_score_threshold=float(fd.get("campaign_score_threshold", 0.55)),
        min_cluster_size=int(fd.get("min_cluster_size", 10)),
        min_vehicles=int(fd.get("min_vehicles", 2)),
        min_behavioral_cohesion=float(fd.get("min_behavioral_cohesion", 0.85)),
        dbscan_eps=float(fd.get("dbscan_eps", 0.8)),
        dbscan_min_samples=int(fd.get("dbscan_min_samples", 10)),
        dbscan_pca_components=int(fd.get("dbscan_pca_components", 8)),
        max_clustering_samples=int(fd.get("max_clustering_samples", 20000)),
        max_graph_viz_nodes=int(fd.get("max_graph_viz_nodes", 800)),
        max_embedding_samples=int(fd.get("max_embedding_samples", 5000)),
        embedding_method=str(fd.get("embedding_method", "tsne")),  # type: ignore[arg-type]
        gnn_supervision=str(fd.get("gnn_supervision", "structure")),  # type: ignore[arg-type]
        checkpoint_path=paths.root / ckpt,
        retrain_gnn=bool(fd.get("retrain_gnn", False)),
        seed=seed,
    )
    run_final_gnn_fleet_decision_experiment(
        descriptors_path=paths.root / artifacts.get("anomaly_descriptors", "data/processed/anomaly_descriptors.csv"),
        features_path=paths.root / artifacts.get("window_features", "data/processed/window_features.csv"),
        outputs=FinalGnnFleetOutputs(
            results_dir=paths.root / str(pub.get("results_dir", "results")),
            tables_dir=paths.root / str(pub.get("tables_dir", "tables")),
            figures_dir=paths.root / str(pub.get("figures_dir", "figures")),
        ),
        cfg=cfg,
    )


def _copy_gnn_supporting_results(src: Path, outputs: IeeePaperOutputs) -> dict[str, Path]:
    names = [
        "final_gnn_graph_statistics.csv",
        "final_gnn_campaign_clusters.csv",
        "final_attack_decisions.csv",
        "final_gnn_fleet_decision_summary.md",
        "table_final_attack_decision_summary.csv",
        "final_local_vs_gnn_fleet_metrics.csv",
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
        "Evidence supports hypotheses **H1–H4** under the deployment-realistic fleet architecture:",
        "Vehicle IDS → Anomaly Descriptors → Behaviour Graph → GraphSAGE (structure-only) → DBSCAN →",
        "`isolated_attack` / `coordinated_attack`. Attack-type labels are used **only** for evaluation plots and tables.",
        "",
    ]
    for title, body in sections.items():
        lines.extend([f"## {title}", "", body.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_ieee_paper_exports(*, repo_root: Path) -> dict[str, Any]:
    plt.rcParams.update(IEEE_RC)
    _ensure_prerequisite_experiments(repo_root)
    _ensure_final_gnn_fleet_decision_results(repo_root)
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
    _cleanup_obsolete_paper_artifacts(outputs)

    written: dict[str, Any] = {}

    # --- H1: Vehicle-Level IDS ---
    t1 = _build_table_01_vehicle_ids(src)
    written["table_01"] = _export_table_bundle(
        t1,
        stem="table_01_vehicle_level_ids",
        caption="Vehicle-Level IDS Performance (Isolation Forest, FPR$\\leq$5\\% threshold).",
        label="tab:vehicle-level-ids",
        title="Table 1: Vehicle-Level IDS Performance",
        outputs=outputs,
    )
    if (src / "vehicle_level_by_attack_type.csv").exists():
        shutil.copy2(src / "vehicle_level_by_attack_type.csv", outputs.results_dir / "table_01_vehicle_level_by_attack.csv")

    _copy_figure(src_fig / "local_ids_pr_curve", outputs.figures_dir / "figure_02_vehicle_level_pr")

    # --- H2: Descriptor Compactness and Security ---
    t2 = _build_table_02_descriptor_security(src)
    written["table_02"] = _export_table_bundle(
        t2,
        stem="table_02_descriptor_compactness_security",
        caption="Descriptor compactness, fleet scalability, and security/privacy metrics.",
        label="tab:descriptor-security",
        title="Table 2: Descriptor Compactness and Security",
        outputs=outputs,
    )
    h2_figs = export_h2_publication_figures(
        scalability_csv=src / "descriptor_fleet_scalability.csv",
        paper_figures_dir=outputs.figures_dir,
        source_figures_dir=src_fig,
    )
    written["figure_04"] = h2_figs["figure_04_bandwidth_scaling"]
    written["figure_05"] = h2_figs["figure_05_descriptor_information_disclosure"]

    # --- H3: Cross-Vehicle Generalisation ---
    t3 = _build_table_03_cross_vehicle(src)
    written["table_03"] = _export_table_bundle(
        t3,
        stem="table_03_cross_vehicle_generalisation",
        caption="Cross-vehicle descriptor generalisation (Random Forest, leave-one-vehicle-out).",
        label="tab:cross-vehicle-generalisation",
        title="Table 3: Cross-Vehicle Descriptor Generalisation",
        outputs=outputs,
    )
    if (src / "vehicle_agnostic_score.csv").exists():
        shutil.copy2(src / "vehicle_agnostic_score.csv", outputs.results_dir / "table_03_vehicle_agnostic_score.csv")
    _copy_figure(src_fig / "descriptor_embedding_by_attack", outputs.figures_dir / "figure_06_cross_vehicle_descriptor_embedding")

    # --- H4: Fleet-Aware GNN IDS ---
    t4 = _build_table_04_local_vs_gnn_fleet(src)
    written["table_04"] = _export_table_bundle(
        t4,
        stem="table_04_local_vs_gnn_fleet_ids",
        caption="Local IDS vs fleet-aware GNN IDS (GraphSAGE + DBSCAN; final decisions "
        "use behavioural cohesion, not attack-type metadata).",
        label="tab:local-vs-gnn-fleet",
        title="Table 4: Local IDS vs Fleet-Aware GNN IDS",
        outputs=outputs,
    )
    t5 = _build_table_05_coordinated_campaigns(src)
    written["table_05"] = _export_table_bundle(
        t5,
        stem="table_05_coordinated_campaign_detection",
        caption="Coordinated campaign detection by attack type (evaluation scenarios; "
        "attack types used for scoring only).",
        label="tab:coordinated-campaign-detection",
        title="Table 5: Coordinated Campaign Detection Results",
        outputs=outputs,
    )
    written["gnn_supporting"] = _copy_gnn_supporting_results(src, outputs)

    _copy_figure(src_fig / "final_gnn_fleet_campaign_graph", outputs.figures_dir / "figure_07_gnn_fleet_campaign_graph")
    _copy_figure(src_fig / "final_attack_decision_distribution", outputs.figures_dir / "figure_08_final_attack_decision_distribution")
    written["figure_07"] = outputs.figures_dir / "figure_07_gnn_fleet_campaign_graph.pdf"
    written["figure_08"] = outputs.figures_dir / "figure_08_final_attack_decision_distribution.pdf"

    mirror_tables = repo_root / "tables"
    mirror_tables.mkdir(exist_ok=True)
    for key in ("table_01", "table_02", "table_03", "table_04", "table_05"):
        for fmt in ("csv", "md", "tex"):
            p = written[key][fmt if fmt != "csv" else "csv"]
            if fmt == "csv":
                shutil.copy2(p, mirror_tables / p.name.replace("table_0", "ieee_table_0"))
            else:
                shutil.copy2(p, mirror_tables / p.name)

    interpretations = {
        "H1 — Vehicle-Level IDS Effectiveness (Table 1; Figure 2)": """
We introduce a **lightweight, self-supervised** Isolation Forest vehicle IDS that flags suspicious CAN windows without attack labels at training time.
On the held-out test split it achieves **PR-AUC 0.927** (Figure 2) and **ROC-AUC 0.786** (Table 1) with **FPR ≤ 5%**, **97.3% precision**, and **46.0% recall** (F1 **62.4%**).
This produces a compact local alert stream uplinked as anomaly descriptors to the fleet correlation layer.

**Interpretation:** H1 is supported — self-supervised local detection is effective enough to feed the fleet pipeline but does not classify coordinated campaigns.
        """,
        "H2 — Descriptor Compactness and Security (Table 2; Figures 4–5)": """
H2 is evaluated on **two complementary axes** — less data sent, and safer content in what is sent:

1. **Compactness (Figure 4; Table 2)** — *How much* leaves the vehicle?
   Per-window uplink drops from **~2,076 bytes** (raw CAN window) to **~165 bytes** (descriptor), i.e. **12.6×** compression and **92%** bandwidth reduction; at 100 vehicles, fleet uplink falls from **~7.6 GB** to **~425 MB** (**94%** reduction, Figure 4).

2. **Security / privacy (Figure 5; Table 2)** — *What sensitive content* is in that uplink?
   Figure 5 compares raw CAN vs descriptor uplink element-by-element: payload bytes and per-frame CAN IDs are **not transmitted**; message order is summarised; anomaly evidence is **preserved** for fleet IDS.

Together, Figure 4 shows the **volume** reduction; Figure 5 shows the **content** reduction. Table 2 reports the numeric compactness metrics plus disclosure rows (e.g. raw payload bytes: Exposed → descriptor: Not transmitted).

**Interpretation:** H2 is supported — descriptors are a smaller *and* safer fleet uplink than raw CAN, while keeping anomaly evidence for correlation.
**Limitation:** Residual vehicle fingerprinting from behavioural patterns remains high (~99.97%); this is a linkability limit, not a claim of full anonymisation.
        """,
        "H3 — Cross-Vehicle Descriptor Generalisation (Table 3; Figure 6)": """
Leave-one-vehicle-out transfer yields mean **ROC-AUC 0.745**, mean **F1 0.896**, and vehicle-agnostic score **0.821**.
Cross-vehicle descriptor embeddings (Figure 6) show attack-behaviour structure spanning vehicle platforms without transmitting raw CAN payloads.

**Interpretation:** H3 is supported — behavioural descriptors generalise across heterogeneous vehicles, enabling fleet-scale correlation without vehicle-identity features in the GNN input.
        """,
        "H4 — Fleet-Aware GNN Correlation (Tables 4–5; Figures 7–8)": """
The fleet layer follows: **Vehicle IDS → anomaly descriptors → behaviour-normalized graph → GraphSAGE (structure-only) → DBSCAN → final decision** (`isolated_attack` | `coordinated_attack`).
Runtime decisions use **behavioural cluster cohesion** and multi-vehicle structure — **not** attack-type labels.

On evaluation scenarios (four multi-vehicle attack families), the GNN fleet IDS achieves **100% campaign recall (4/4)** vs **0%** for local IDS alone (Table 5).
**7,267** locally suspicious events are classified as `coordinated_attack`; **49,760** as `isolated_attack` (Figure 8).
Campaign precision is **80%** with behavioural cohesion **0.984** (Table 4); one qualifying cluster is unmatched under evaluation mapping (false campaign rate **20%**).

Figure 7 colours nodes by final fleet decision. Per-attack-type evaluation metrics are reported in Table 5 only.

**Interpretation:** H4 is supported — GraphSAGE fleet correlation adds coordinated-campaign classification beyond isolated local detection.
**Limitation:** Evaluation scenarios are synthetically defined from labelled windows; attack-type names appear only in evaluation tables.
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
                "## Architecture (final)",
                "Vehicle IDS → Anomaly Descriptors → Behaviour Graph → GraphSAGE (structure-only) → DBSCAN →",
                "`isolated_attack` / `coordinated_attack`",
                "",
                "## Tables",
                "| Table | File | Hypothesis |",
                "|-------|------|------------|",
                "| Table 1 | `paper/tables/table_01_vehicle_level_ids.tex` | H1 — Vehicle-Level IDS |",
                "| Table 2 | `paper/tables/table_02_descriptor_compactness_security.tex` | H2 — Descriptor Security |",
                "| Table 3 | `paper/tables/table_03_cross_vehicle_generalisation.tex` | H3 — Cross-Vehicle Generalisation |",
                "| Table 4 | `paper/tables/table_04_local_vs_gnn_fleet_ids.tex` | H4 — Fleet GNN IDS |",
                "| Table 5 | `paper/tables/table_05_coordinated_campaign_detection.tex` | H4 — Campaign Detection |",
                "",
                "## Figures",
                "| Figure | File | Hypothesis |",
                "|--------|------|------------|",
                "| Figure 2 | `paper/figures/figure_02_vehicle_level_pr.pdf` | H1 — PR curve (self-supervised local IDS) |",
                "| Figure 4 | `paper/figures/figure_04_bandwidth_scaling.pdf` | H2 — bandwidth scaling (color line chart) |",
                "| Figure 5 | `paper/figures/figure_05_descriptor_information_disclosure.pdf` | H2 — information disclosure matrix (color) |",
                "",
                "## Figure captions (H2)",
                "",
                f"**Figure 4.** {PAPER_FIGURE_CAPTIONS['figure_04_bandwidth_scaling']}",
                "",
                f"**Figure 5.** {PAPER_FIGURE_CAPTIONS['figure_05_descriptor_information_disclosure']}",
                "| Figure 6 | `paper/figures/figure_06_cross_vehicle_descriptor_embedding.pdf` | H3 |",
                "| Figure 7 | `paper/figures/figure_07_gnn_fleet_campaign_graph.pdf` | H4 |",
                "| Figure 8 | `paper/figures/figure_08_final_attack_decision_distribution.pdf` | H4 |",
                "",
                "Figure 7 node colour = `isolated_attack` vs `coordinated_attack` (runtime decision).",
                "Per-attack-type evaluation metrics are in Table 5 only (no Figure 9).",
                "Attack types in Figure 6 are evaluation/visualisation only.",
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
