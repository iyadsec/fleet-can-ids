"""Tables and figures for evaluation correction."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_LABELS = {
    "local_ids": "M1 Local IDS",
    "descriptor_clustering": "M2 Clustering",
    "standard_gnn": "M3 Standard GNN",
    "fcgnn": "M4 FCGNN",
}
METHOD_ORDER = ["local_ids", "descriptor_clustering", "standard_gnn", "fcgnn"]


def _write_table_bundle(df: pd.DataFrame, stem: Path, *, caption: str, label: str) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    md = f"# {caption}\n\n{df.to_markdown(index=False)}\n"
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    tex_body = df.to_latex(index=False, escape=True, float_format="%.4f")
    tex = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            tex_body,
            "\\end{table}",
        ]
    )
    stem.with_suffix(".tex").write_text(tex, encoding="utf-8")


def _mean_std_str(s: pd.Series, d: int = 3) -> str:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return "N/A"
    if len(x) == 1:
        return f"{x.mean():.{d}f}"
    return f"{x.mean():.{d}f} ± {x.std():.{d}f}"


def export_tables(
    *,
    confusion: pd.DataFrame,
    vehicle: pd.DataFrame,
    campaign_err: pd.DataFrame,
    weak_results: pd.DataFrame,
    stats: pd.DataFrame,
    fixed_evidence: pd.DataFrame,
    original_vs_corrected: pd.DataFrame,
    tables_dir: Path,
) -> list[str]:
    created: list[str] = []
    bundles = [
        ("table_E1_event_confusion_metrics", confusion, "Event-level confusion metrics (corrected)"),
        ("table_E2_vehicle_level_detailed_metrics", vehicle, "Vehicle-level detailed metrics"),
        ("table_E3_campaign_error_breakdown", campaign_err, "Campaign error breakdown"),
        ("table_E4_corrected_weak_campaign_results", weak_results, "Corrected weak campaign results"),
        ("table_E5_corrected_statistical_tests", stats, "Corrected statistical tests"),
        ("table_E6_fixed_evidence_control", fixed_evidence, "Fixed-evidence control results"),
        ("table_E7_original_vs_corrected_evaluation", original_vs_corrected, "Original vs corrected evaluation"),
    ]
    for name, df, caption in bundles:
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            continue
        _write_table_bundle(df, tables_dir / name, caption=caption, label=name)
        created.append(name)
    return created


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)


def export_figures(
    corrected_metrics: pd.DataFrame,
    campaign_err: pd.DataFrame,
    fixed_evidence: pd.DataFrame,
    figures_dir: Path,
) -> list[str]:
    created: list[str] = []
    if corrected_metrics.empty:
        return created

    for strength, tag, fname in (
        ("strong", "E1", "figure_E1_event_precision_recall_by_method"),
        ("weak", "E1w", "figure_E1_event_precision_recall_by_method_weak"),
    ):
        sub = corrected_metrics[corrected_metrics["attack_strength"] == strength]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in METHOD_ORDER:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue
            ax.scatter(
                msub["precision"].mean(),
                msub["recall"].mean(),
                label=METHOD_LABELS.get(method, method),
                s=80,
            )
        ax.set_xlabel("Event precision")
        ax.set_ylabel("Event recall")
        ax.set_title(f"Event precision–recall ({strength})")
        ax.legend()
        _save_fig(fig, figures_dir / (fname if strength == "weak" else "figure_E1_event_precision_recall_by_method"))
        created.append(fname)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHOD_ORDER:
        msub = corrected_metrics[corrected_metrics["method"] == method]
        if msub.empty:
            continue
        ax.bar(
            METHOD_LABELS.get(method, method),
            msub["fpr"].mean(),
            yerr=msub["fpr"].std(),
            capsize=4,
        )
    ax.set_ylabel("Event FPR")
    ax.set_title("Event FPR by method (corrected)")
    plt.xticks(rotation=20, ha="right")
    _save_fig(fig, figures_dir / "figure_E2_event_FPR_by_method")
    created.append("figure_E2_event_FPR_by_method")

    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHOD_ORDER:
        msub = corrected_metrics[corrected_metrics["method"] == method]
        if msub.empty:
            continue
        ax.scatter(
            msub["vehicle_precision"].mean(),
            msub["vehicle_recall"].mean(),
            label=METHOD_LABELS.get(method, method),
            s=80,
        )
    ax.set_xlabel("Attacked-vehicle precision")
    ax.set_ylabel("Attacked-vehicle recall")
    ax.set_title("Vehicle precision–recall (corrected)")
    ax.legend()
    _save_fig(fig, figures_dir / "figure_E3_vehicle_precision_recall")
    created.append("figure_E3_vehicle_precision_recall")

    fcgnn = corrected_metrics[corrected_metrics["method"] == "fcgnn"]
    if not fcgnn.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for metric, ylab in (("campaign_precision", "Campaign precision"), ("campaign_recall", "Campaign recall")):
            by_size = fcgnn.groupby("campaign_size")[metric].agg(["mean", "std"])
            ax.errorbar(
                by_size.index,
                by_size["mean"],
                yerr=by_size["std"],
                marker="o",
                label=ylab,
                capsize=4,
            )
        ax.set_xlabel("Campaign size")
        ax.set_title("FCGNN campaign precision/recall by size")
        ax.legend()
        _save_fig(fig, figures_dir / "figure_E4_campaign_precision_recall_by_size")
        created.append("figure_E4_campaign_precision_recall_by_size")

    if not campaign_err.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        agg = (
            campaign_err.groupby(["method", "campaign_size"])
            .agg(
                false_clusters=("false_campaign_clusters", "mean"),
                missed=("missed_campaigns", "mean"),
                benign_veh=("benign_vehicles_incorrectly_included", "mean"),
            )
            .reset_index()
        )
        x = np.arange(len(agg))
        ax.bar(x - 0.2, agg["false_clusters"], width=0.2, label="False campaign clusters")
        ax.bar(x, agg["missed"], width=0.2, label="Missed campaigns")
        ax.bar(x + 0.2, agg["benign_veh"], width=0.2, label="Benign vehicles included")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{METHOD_LABELS.get(m, m)}\nn={int(c)}" for m, c in zip(agg["method"], agg["campaign_size"])],
            rotation=45,
            ha="right",
        )
        ax.legend()
        ax.set_title("Campaign error breakdown")
        _save_fig(fig, figures_dir / "figure_E5_campaign_error_breakdown")
        created.append("figure_E5_campaign_error_breakdown")

    if not fixed_evidence.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for method in ("descriptor_clustering", "standard_gnn", "fcgnn"):
            msub = fixed_evidence[fixed_evidence["method"] == method]
            if msub.empty:
                continue
            by_size = msub.groupby("campaign_size")["campaign_f1"].mean()
            ax.plot(by_size.index, by_size.values, marker="o", label=METHOD_LABELS.get(method, method))
        ax.set_xlabel("Campaign size (fixed 20 malicious descriptors)")
        ax.set_ylabel("Campaign F1")
        ax.set_title("Fixed-evidence control: campaign F1 vs size")
        ax.legend()
        _save_fig(fig, figures_dir / "figure_E6_fixed_evidence_campaign_F1_vs_size")
        created.append("figure_E6_fixed_evidence_campaign_F1_vs_size")

        fig, ax = plt.subplots(figsize=(7, 5))
        for method in ("descriptor_clustering", "standard_gnn", "fcgnn"):
            msub = fixed_evidence[fixed_evidence["method"] == method]
            if msub.empty:
                continue
            by_size = msub.groupby("campaign_size")["campaign_detection_rate"].mean()
            ax.plot(by_size.index, by_size.values, marker="o", label=METHOD_LABELS.get(method, method))
        ax.set_xlabel("Campaign size")
        ax.set_ylabel("Campaign detection rate")
        ax.set_title("Fixed-evidence control: detection vs size")
        ax.legend()
        _save_fig(fig, figures_dir / "figure_E7_fixed_evidence_detection_vs_size")
        created.append("figure_E7_fixed_evidence_detection_vs_size")

    return created


def summarize_weak_campaign_results(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "method",
        "campaign_size",
        "precision",
        "recall",
        "f1",
        "fpr",
        "vehicle_precision",
        "vehicle_recall",
        "vehicle_event_coverage_mean",
        "campaign_detection_rate",
        "campaign_precision",
        "campaign_recall",
        "campaign_f1",
        "weak_malicious_promoted",
        "benign_incorrectly_promoted",
    ]
    sub = df[df["attack_strength"] == "weak"].copy()
    if sub.empty:
        return pd.DataFrame()
    present = [c for c in cols if c in sub.columns]
    rows: list[dict] = []
    for (method, cs), grp in sub.groupby(["method", "campaign_size"]):
        row = {"Method": METHOD_LABELS.get(method, method), "Campaign size": int(cs)}
        for c in present:
            if c in ("method", "campaign_size"):
                continue
            if c in ("weak_malicious_promoted", "benign_incorrectly_promoted"):
                row[c.replace("_", " ").title()] = int(grp[c].sum())
            else:
                row[c.replace("_", " ").title()] = _mean_std_str(grp[c])
        rows.append(row)
    return pd.DataFrame(rows)
