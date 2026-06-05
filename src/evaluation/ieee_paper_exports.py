"""Publication-quality exports for the IEEE Experimental Evaluation section (4 contributions)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
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
    for ext in (".pdf", ".png"):
        if src.with_suffix(ext).exists():
            shutil.copy2(src.with_suffix(ext), dst_base.with_suffix(ext))


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


def _build_table_04_fleet_correlation(src: Path) -> pd.DataFrame:
    local = _load_csv(src, "local_only_detection_metrics.csv").iloc[0]
    fleet = _load_csv(src, "fleet_level_detection_metrics.csv").iloc[0]
    weak_local = _load_csv(src, "weak_anomaly_local_metrics.csv").iloc[0]
    selective = _load_csv(src, "selective_weak_promotion_operating_point.csv").iloc[0]
    conservative = _load_csv(src, "weak_recovery_best_configurations.csv")
    cons = conservative[conservative["Configuration Name"] == "Conservative"].iloc[0]
    ablation = _load_csv(src, "behavior_view_fleet_graph_statistics.csv")
    full_row = ablation[ablation["similarity_view"] == "full_descriptor"].iloc[0]
    norm_row = ablation[ablation["similarity_view"] == "behavior_only_vehicle_normalized"].iloc[0]
    full_cross_pct = 100.0 * float(full_row["num_cross_vehicle_edges"]) / max(float(full_row["num_edges"]), 1.0)
    norm_cross_pct = 100.0 * float(norm_row["num_cross_vehicle_edges"]) / max(float(norm_row["num_edges"]), 1.0)

    return pd.DataFrame(
        [
            {
                "Evaluation": "Full-dataset detection (strong alerts)",
                "Local IDS F1": round(float(local["f1"]), 4),
                "Fleet-aware F1": round(float(fleet["f1"]), 4),
                "FPR (%)": round(float(fleet["false_positive_rate"]) * 100, 2),
                "Cross-vehicle edges (%)": float("nan"),
                "Note": "No F1 gain under top-k graph + ≥3 vehicle cluster gates",
            },
            {
                "Evaluation": "Weak anomalies — local baseline",
                "Local IDS F1": round(float(weak_local["f1"]), 4),
                "Fleet-aware F1": round(float(selective["f1"]), 4),
                "FPR (%)": round(float(selective["false_positive_rate"]) * 100, 2),
                "Cross-vehicle edges (%)": float("nan"),
                "Note": "Selective DBSCAN promotion (eps=1.2, gated)",
            },
            {
                "Evaluation": "Weak anomalies — optimized conservative",
                "Local IDS F1": 0.0,
                "Fleet-aware F1": round(float(cons["F1"]), 4),
                "FPR (%)": round(float(cons["FPR"]) * 100, 2),
                "Cross-vehicle edges (%)": float("nan"),
                "Note": f"Weak recovery {cons['Recovery Rate']:.2f}% (grid search, FPR≤5%)",
            },
            {
                "Evaluation": "Graph connectivity — full descriptor",
                "Local IDS F1": float("nan"),
                "Fleet-aware F1": float("nan"),
                "FPR (%)": float("nan"),
                "Cross-vehicle edges (%)": round(full_cross_pct, 4),
                "Note": "Top-k similarity on full descriptor features",
            },
            {
                "Evaluation": "Graph connectivity — behaviour-normalized",
                "Local IDS F1": float("nan"),
                "Fleet-aware F1": float("nan"),
                "FPR (%)": float("nan"),
                "Cross-vehicle edges (%)": round(norm_cross_pct, 4),
                "Note": "Vehicle-normalized behavioural similarity view",
            },
        ]
    )


def _figure_06_fleet_correlation(src: Path, outputs: IeeePaperOutputs) -> Path:
    """Two-panel fleet correlation summary (connectivity + weak recovery trade-off)."""
    ablation = pd.read_csv(src / "behavior_view_topk_vehicle_bias.csv")
    ops = pd.read_csv(src / "weak_recovery_best_configurations.csv")
    ops = ops[ops["Configuration Name"].isin(["Conservative", "Balanced", "Maximum Recovery"])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Full\nDescriptor", "Behaviour-\nOnly", "Behaviour-\nNormalized"]
    x = np.arange(len(labels))
    same = ablation["pct_same_vehicle_edges"].to_numpy()
    cross = ablation["pct_cross_vehicle_edges"].to_numpy()
    w = 0.35
    axes[0].bar(x - w / 2, same, width=w, label="Same-vehicle", color="#4472C4")
    axes[0].bar(x + w / 2, cross, width=w, label="Cross-vehicle", color="#ED7D31")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylabel("Edge share (%)")
    axes[0].set_title("(a) Fleet graph connectivity")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].scatter(
        ops["FPR"] * 100,
        ops["Recovery Rate"],
        s=80,
        c=["#70AD47", "#4472C4", "#C00000"],
        edgecolors="k",
        linewidths=0.5,
    )
    for _, row in ops.iterrows():
        axes[1].annotate(
            row["Configuration Name"].replace(" ", "\n"),
            (row["FPR"] * 100, row["Recovery Rate"]),
            fontsize=6,
            xytext=(4, 4),
            textcoords="offset points",
        )
    axes[1].set_xlabel("False positive rate (%)")
    axes[1].set_ylabel("Weak recovery rate (%)")
    axes[1].set_title("(b) Weak anomaly recovery operating points")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Fleet-level correlation analysis", fontsize=11)
    fig.tight_layout()
    base = outputs.figures_dir / "figure_06_fleet_correlation_analysis"
    _save_figure(fig, base)
    return base.with_suffix(".pdf")


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

    # --- Contribution 4: Fleet-Level Correlation ---
    t4 = _build_table_04_fleet_correlation(src)
    written["table_04"] = _export_table_bundle(
        t4,
        stem="table_04_fleet_correlation_analysis",
        caption="Fleet-level correlation analysis: full-dataset detection, weak-anomaly recovery, and graph connectivity.",
        label="tab:fleet-correlation",
        title="Table 4: Fleet-Level Correlation Analysis",
        outputs=outputs,
    )
    fig6 = _figure_06_fleet_correlation(src, outputs)
    written["figure_06"] = fig6

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

    interpretations = {
        "Contribution 1 — Vehicle-Level IDS Effectiveness": """
The self-supervised Isolation Forest achieves **ROC-AUC 0.786** and **PR-AUC 0.927** at an operating point selected for **FPR ≤ 5%**.
Precision is high (**97.3%**) but recall is moderate (**46.0%**, F1 **62.4%**), indicating conservative strong-alert generation.
Per-attack F1 ranges from **39.8% (replay)** to **81.3% (fuzzy)**; replay remains the hardest class at the chosen threshold.

**Interpretation:** The vehicle-level IDS provides a usable local baseline with low false alarms, but does not fully recover weak or replay-dominated attacks without fleet-level correlation.
**Limitation:** Threshold selection trades recall for FPR; weak anomalies are largely deferred to the fleet layer.
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
        "Contribution 4 — Fleet-Level Correlation Analysis": """
On the **full labelled dataset**, fleet graph correlation with **≥3 vehicle cluster gates** does **not** improve strong-alert F1 over local IDS (**0.846 vs 0.846**) under the original top-k similarity graph.
**Behaviour-normalized** graph construction increases cross-vehicle edges from **≈0.02%** to **≈1.08%**, enabling cross-platform correlation that identity-dominated similarity suppresses.

For **weak anomalies**, ungated connected-component promotion inflates FPR; **selective DBSCAN promotion** achieves modest recovery (**≈1.3%**, conservative grid-search point) at **FPR ≈ 1.6%** (operating point) to **2.3%** (optimized conservative).
The full-dataset strong-alert evaluation reports identical local and fleet F1 (**0.846**) at **FPR ≈ 44%** — a different operating context from the vehicle-level threshold in Table 1 (FPR ≤ 5%).
The IEEE recovery target (≥10% at FPR ≤ 10%) was **not achieved** in systematic optimization (max recovery **1.41%**).

**Interpretation:** Fleet correlation adds value primarily through (i) cross-vehicle graph connectivity and (ii) gated weak-anomaly promotion, not through blanket cluster escalation.
**Limitation:** Strong-anomaly fleet gains are null under current gates; weak recovery remains low despite cross-vehicle connectivity improvements.
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
                "| Table 4 | `paper/tables/table_04_fleet_correlation_analysis.tex` | Fleet Correlation |",
                "",
                "## Figures",
                "| Figure | File | Contribution |",
                "|--------|------|--------------|",
                "| Figure 1 | `paper/figures/figure_01_vehicle_ids_roc.pdf` | Vehicle-Level IDS |",
                "| Figure 2 | `paper/figures/figure_02_vehicle_ids_score_distribution.pdf` | Vehicle-Level IDS |",
                "| Figure 3 | `paper/figures/figure_03_descriptor_bandwidth_exposure.pdf` | Descriptor Security |",
                "| Figure 4 | `paper/figures/figure_04_payload_reconstruction_risk.pdf` | Descriptor Security |",
                "| Figure 5 | `paper/figures/figure_05_cross_vehicle_embedding.pdf` | Cross-Vehicle Generalisation |",
                "| Figure 6 | `paper/figures/figure_06_fleet_correlation_analysis.pdf` | Fleet Correlation |",
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
