"""Generate publication-ready tables (CSV, Markdown, LaTeX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments.publication_metrics import NA, aggregate_validated_metrics
from src.experiments.scenario_registry import METHOD_LABELS, SCENARIO_REGISTRY
from src.experiments.statistical_testing import run_paired_comparisons

METHOD_ORDER = ["local_ids", "descriptor_clustering", "standard_gnn", "fcgnn"]


def _escape_tex(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _mean_std_str(df: pd.DataFrame, col: str, d: int = 3) -> str:
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        return NA
    if len(s) == 1:
        return f"{s.mean():.{d}f}"
    return f"{s.mean():.{d}f} $\\pm$ {s.std():.{d}f}"


def _write_table(df: pd.DataFrame, stem: str, out_dir: Path, caption: str, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    cols = list(df.columns)
    md = [f"# {caption}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, r in df.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    (out_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    body_rows = []
    for _, r in df.iterrows():
        body_rows.append(" & ".join(_escape_tex(str(r[c])) for c in cols) + r" \\")
    tex = "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            "\\begin{tabular}{" + "l" * len(cols) + "}",
            "\\toprule",
            " & ".join(_escape_tex(c) for c in cols) + r" \\",
            "\\midrule",
            *body_rows,
            "\\bottomrule",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )
    (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")


def table_01_experimental_design(config: dict, out_dir: Path) -> None:
    g = config.get("general", {})
    c = config.get("campaign", {})
    l = config.get("local_ids", {})
    gr = config.get("graph", {})
    panel_a = pd.DataFrame(
        [
            {"Parameter": "Split strategy", "Value": "trace-grouped train/val/test (70/15/15)"},
            {"Parameter": "Random seeds", "Value": str(g.get("seeds", []))},
            {"Parameter": "Campaign sizes", "Value": str(c.get("campaign_sizes", []))},
            {"Parameter": "Coordination strengths", "Value": str(c.get("coordination_strengths", []))},
            {"Parameter": "Weak threshold", "Value": str(l.get("weak_threshold", 0.55))},
            {"Parameter": "Strong threshold", "Value": str(l.get("strong_threshold", 0.80))},
            {"Parameter": "Similarity metric", "Value": gr.get("similarity_metric", "cosine")},
            {"Parameter": "Temporal edges", "Value": str(gr.get("use_temporal_edges", False))},
            {"Parameter": "Min campaign vehicles", "Value": str(c.get("minimum_unique_vehicles", 2))},
        ]
    )
    panel_b = pd.DataFrame(
        [{"Method": METHOD_LABELS[m], "ID": m} for m in METHOD_ORDER]
    )
    _write_table(panel_a, "table_01_experimental_design_panel_a", out_dir, "Experimental design parameters", "tab:experimental_design")
    _write_table(panel_b, "table_01_experimental_design_panel_b", out_dir, "Compared detection methods", "tab:experimental_design_methods")


def table_02_scenario_definitions(out_dir: Path) -> None:
    rows = []
    defs = {
        "S0_benign_control": ("Benign fleet only", "0", "No campaign", "False campaign alert rate", "Safety control"),
        "S1_isolated": ("Single-vehicle attack", "1", "Isolated incident", "Isolated vs campaign distinction", "Single-vehicle control"),
        "S2_non_coordinated": ("Distinct attack families", "2,5,10", "Separate incidents", "Incorrect campaign merging", "Merge resistance"),
        "S3_strong_campaign": ("Coordinated strong attacks", "2,5,10", "One campaign", "Campaign F1 / detection rate", "Strong campaign correlation"),
        "S4_weak_campaign": ("Coordinated weak attacks", "2,5,10", "One campaign", "Weak-event recovery", "Primary contribution"),
    }
    for key, (fleet, sizes, gt, metric, purpose) in defs.items():
        rows.append(
            {
                "Scenario": SCENARIO_REGISTRY[key].scenario_id,
                "Fleet condition": fleet,
                "Campaign sizes": sizes,
                "Ground-truth decision": gt,
                "Primary metric": metric,
                "Research purpose": purpose,
            }
        )
    _write_table(pd.DataFrame(rows), "table_02_scenario_definitions", out_dir, "Controlled scenario definitions", "tab:scenario_definitions")


def table_03_safety_controls(metrics: pd.DataFrame, out_dir: Path) -> None:
    sub = metrics[metrics["scenario_key"].isin(["S0_benign_control", "S1_isolated", "S2_non_coordinated"])]
    rows = []
    for (sc, meth, n), g in sub.groupby(["scenario_key", "method", "campaign_size"]):
        rows.append(
            {
                "Scenario": sc.replace("_", " ").split()[0],
                "Method": METHOD_LABELS.get(meth, meth),
                "Campaign size": int(n),
                "Vehicle recall": _mean_std_str(g, "vehicle_recall"),
                "False campaign alert rate": _mean_std_str(g, "false_campaign_alert_rate"),
                "Incorrect campaign merging": _mean_std_str(g, "incorrect_campaign_merging"),
                "Benign vehicles incorrectly included": _mean_std_str(g, "benign_vehicles_incorrectly_included"),
                "Detected campaign clusters": _mean_std_str(g, "n_detected_campaign_clusters"),
                "Campaign precision": NA if sc in ("S0_benign_control", "S1_isolated") else _mean_std_str(g, "campaign_precision"),
                "Campaign recall": NA if sc in ("S0_benign_control", "S1_isolated") else _mean_std_str(g, "campaign_recall"),
            }
        )
    _write_table(pd.DataFrame(rows), "table_03_safety_controls", out_dir, "Safety controls (S0--S2)", "tab:safety_controls")


def table_04_strong_campaign(metrics: pd.DataFrame, out_dir: Path) -> None:
    sub = metrics[metrics["scenario_key"] == "S3_strong_campaign"]
    rows = []
    for (meth, n), g in sub.groupby(["method", "campaign_size"]):
        rows.append(
            {
                "Method": METHOD_LABELS.get(meth, meth),
                "Campaign size": int(n),
                "Event precision": _mean_std_str(g, "precision"),
                "Event recall": _mean_std_str(g, "recall"),
                "Event F1": _mean_std_str(g, "f1"),
                "Attacked-vehicle recall": _mean_std_str(g, "vehicle_recall"),
                "Campaign detection rate": _mean_std_str(g, "campaign_detection_rate"),
                "Campaign F1": _mean_std_str(g, "campaign_f1"),
                "Incorrect merging": _mean_std_str(g, "incorrect_campaign_merging"),
                "Latency (s)": _mean_std_str(g, "runtime_total_sec"),
            }
        )
    _write_table(pd.DataFrame(rows), "table_04_strong_campaign_results", out_dir, "Strong coordinated campaign results (S3)", "tab:strong_campaign")


def table_05_weak_campaign(metrics: pd.DataFrame, out_dir: Path) -> None:
    sub = metrics[metrics["scenario_key"] == "S4_weak_campaign"]
    rows = []
    for (meth, n), g in sub.groupby(["method", "campaign_size"]):
        rows.append(
            {
                "Method": METHOD_LABELS.get(meth, meth),
                "Campaign size": int(n),
                "Event precision": _mean_std_str(g, "precision"),
                "Event recall": _mean_std_str(g, "recall"),
                "Event F1": _mean_std_str(g, "f1"),
                "Event FPR": _mean_std_str(g, "fpr"),
                "Attacked-vehicle recall": _mean_std_str(g, "vehicle_recall"),
                "Campaign detection rate": _mean_std_str(g, "campaign_detection_rate"),
                "Campaign F1": _mean_std_str(g, "campaign_f1"),
                "Weak malicious recovered": _mean_std_str(g, "weak_malicious_recovered"),
                "Weak benign promoted": _mean_std_str(g, "weak_benign_promoted"),
                "Latency (s)": _mean_std_str(g, "runtime_total_sec"),
            }
        )
    _write_table(pd.DataFrame(rows), "table_05_weak_campaign_results", out_dir, "Weak coordinated campaign results (S4)", "tab:weak_campaign")


def table_06_ablation(metrics: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for sc in ["S3_strong_campaign", "S4_weak_campaign"]:
        sub = metrics[metrics["scenario_key"] == sc]
        for meth in METHOD_ORDER:
            g = sub[sub["method"] == meth]
            if g.empty:
                continue
            rows.append(
                {
                    "Scenario": sc.split("_")[0],
                    "Method": METHOD_LABELS.get(meth, meth),
                    "Event recall": _mean_std_str(g, "recall"),
                    "Event F1": _mean_std_str(g, "f1"),
                    "Vehicle recall": _mean_std_str(g, "vehicle_recall"),
                    "Campaign detection rate": _mean_std_str(g, "campaign_detection_rate"),
                    "Campaign F1": _mean_std_str(g, "campaign_f1"),
                    "False campaign rate": _mean_std_str(g, "false_campaign_alert_rate"),
                }
            )
        loc = sub[sub["method"] == "local_ids"]
        fcg = sub[sub["method"] == "fcgnn"]
        if not loc.empty and not fcg.empty:
            rows.append(
                {
                    "Scenario": sc.split("_")[0],
                    "Method": "Δ FCGNN − Local",
                    "Event recall": f"{fcg['recall'].mean() - loc['recall'].mean():+.3f}",
                    "Event F1": f"{fcg['f1'].mean() - loc['f1'].mean():+.3f}",
                    "Vehicle recall": f"{fcg['vehicle_recall'].mean() - loc['vehicle_recall'].mean():+.3f}",
                    "Campaign detection rate": f"{fcg['campaign_detection_rate'].mean() - loc['campaign_detection_rate'].mean():+.3f}",
                    "Campaign F1": f"{fcg['campaign_f1'].mean() - loc['campaign_f1'].mean():+.3f}",
                    "False campaign rate": f"{fcg['false_campaign_alert_rate'].mean() - loc['false_campaign_alert_rate'].mean():+.3f}",
                }
            )
    _write_table(pd.DataFrame(rows), "table_06_method_ablation", out_dir, "Method ablation and primary improvement", "tab:method_ablation")


def table_07_statistics(metrics: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for sc in ["S3_strong_campaign", "S4_weak_campaign"]:
        sub = metrics[metrics["scenario_key"] == sc]
        if sub.empty:
            continue
        stats = run_paired_comparisons(sub, scenario_key=sc)
        for _, r in stats.iterrows():
            rows.append(
                {
                    "Scenario": sc.split("_")[0],
                    "Metric": r["metric"],
                    "Comparison": f"FCGNN vs {r['compare_method']}",
                    "Paired seeds": int(r["n_pairs"]),
                    "Test": r["test"],
                    "Mean paired difference": round(float(r["mean_difference"]), 3) if pd.notna(r["mean_difference"]) else NA,
                    "95% CI": f"[{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]" if pd.notna(r["ci95_low"]) else NA,
                    "p-value": round(float(r["p_value"]), 4) if pd.notna(r["p_value"]) else NA,
                    "Holm p-value": round(float(r["p_value_holm"]), 4) if "p_value_holm" in r and pd.notna(r["p_value_holm"]) else NA,
                    "Effect size": round(float(r["effect_size"]), 3) if pd.notna(r["effect_size"]) else NA,
                }
            )
    _write_table(pd.DataFrame(rows), "table_07_statistical_significance", out_dir, "Paired statistical significance", "tab:statistical_significance")


def table_08_edge_sensitivity(edge_df: pd.DataFrame, out_dir: Path) -> None:
    if edge_df.empty:
        df = pd.DataFrame([{"Status": "Edge sensitivity sweep not yet executed"}])
    else:
        cols = [
            "method", "similarity_threshold", "top_k_same_vehicle", "unique_undirected_edges",
            "pyg_stored_edges", "average_degree", "isolated_node_percentage",
            "cross_vehicle_edge_percentage", "campaign_precision", "campaign_recall",
            "campaign_f1", "false_campaign_alert_rate", "graph_construction_time_sec",
            "runtime_gnn_inference_sec", "runtime_total_sec",
        ]
        avail = [c for c in cols if c in edge_df.columns]
        df = edge_df[avail].copy()
        if "method" in df.columns:
            df["method"] = df["method"].map(lambda m: METHOD_LABELS.get(m, m))
        for c in df.select_dtypes(include="number").columns:
            df[c] = df[c].round(3)
    _write_table(df, "table_08_edge_sensitivity", out_dir, "Edge connectivity sensitivity", "tab:edge_sensitivity")


def generate_all_tables(
    validated: pd.DataFrame,
    config: dict,
    out_dir: Path,
    edge_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metrics = aggregate_validated_metrics(validated)
    metrics.to_csv(out_dir.parent / "data" / "validated_metrics.csv", index=False)
    table_01_experimental_design(config, out_dir)
    table_02_scenario_definitions(out_dir)
    table_03_safety_controls(metrics, out_dir)
    table_04_strong_campaign(metrics, out_dir)
    table_05_weak_campaign(metrics, out_dir)
    table_06_ablation(metrics, out_dir)
    table_07_statistics(metrics, out_dir)
    table_08_edge_sensitivity(edge_df if edge_df is not None else pd.DataFrame(), out_dir)
    return metrics
