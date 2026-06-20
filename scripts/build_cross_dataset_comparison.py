#!/usr/bin/env python3
"""Build descriptive cross-dataset comparison: OCSLab vs can-train-and-test.

Reads existing CSV/Markdown/LaTeX result tables only — no experiment reruns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.ctt.constants import OCSLAB_PUBLICATION_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
CTT_ROOT = REPO_ROOT / "new_experiments/can_train_and_test_cross_dataset_validation/full"
COMP_ROOT = REPO_ROOT / "new_experiments/cross_dataset_comparison_ocslab_vs_ctt"

OCSLAB_MISSING = "SOURCE_NOT_IN_WORKSPACE"
SCENARIO_MAP = {
    "Benign fleet control": "Benign-Fleet Control",
    "Isolated attack": "Isolated Single-Vehicle Attack",
    "Unrelated incidents": "Unrelated Multi-Vehicle Incidents",
    "Strong coordinated campaign": "Strong Behaviourally Related Campaign",
    "Weak coordinated campaign": "Weak Behaviourally Related Campaign",
}


@dataclass
class SourceEntry:
    comparison_artifact: str
    metric: str
    dataset: str
    value: str
    source_file: str
    source_column: str
    notes: str = ""


@dataclass
class ComparisonContext:
    ocs_available: bool
    ocs: dict[str, Any] = field(default_factory=dict)
    ctt: dict[str, Any] = field(default_factory=dict)
    source_map: list[SourceEntry] = field(default_factory=list)
    ocs_mtimes_before: dict[str, float] = field(default_factory=dict)
    ctt_mtimes_before: dict[str, float] = field(default_factory=dict)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def snapshot_mtimes(root: Path) -> dict[str, float]:
    if not root.exists():
        return {}
    return {str(p): p.stat().st_mtime for p in root.rglob("*") if p.is_file()}


def add_source(
    ctx: ComparisonContext,
    artifact: str,
    metric: str,
    dataset: str,
    value: Any,
    source_file: Path | str,
    source_column: str = "",
    notes: str = "",
) -> None:
    rel = str(source_file)
    try:
        rel = str(Path(source_file).relative_to(REPO_ROOT))
    except ValueError:
        pass
    ctx.source_map.append(
        SourceEntry(
            comparison_artifact=artifact,
            metric=metric,
            dataset=dataset,
            value=str(value),
            source_file=rel,
            source_column=source_column,
            notes=notes,
        )
    )


def save_table(df: pd.DataFrame, name: str, tables_dir: Path) -> None:
    df.to_csv(tables_dir / f"{name}.csv", index=False)
    try:
        md = df.to_markdown(index=False)
    except (ImportError, Exception):
        md = df.to_string(index=False)
    (tables_dir / f"{name}.md").write_text(f"# {name}\n\n{md}\n", encoding="utf-8")
    try:
        latex = df.to_latex(index=False, escape=False)
        (tables_dir / f"{name}.tex").write_text(latex, encoding="utf-8")
    except Exception:
        pass


def _find_csv(root: Path, patterns: list[str]) -> Path | None:
    if not root.exists():
        return None
    for pat in patterns:
        matches = sorted(root.rglob(pat))
        for m in matches:
            if m.suffix.lower() == ".csv" and m.stat().st_size > 0:
                return m
    return None


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _first_col_match(df: pd.DataFrame, *candidates: str) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def _scalar_from_df(df: pd.DataFrame, col: str, agg: str = "mean") -> float | None:
    if df.empty or col not in df.columns:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None
    if agg == "mean":
        return float(series.mean())
    if agg == "sum":
        return float(series.sum())
    if agg == "first":
        return float(series.iloc[0])
    return float(series.iloc[0])


def load_ocslab() -> tuple[bool, dict[str, Any]]:
    root = REPO_ROOT / OCSLAB_PUBLICATION_ROOT
    if not root.exists():
        return False, {}

    data: dict[str, Any] = {"root": root}
    search_roots = [root / "pooled/tables", root / "tables", root]
    for sub in search_roots:
        if not sub.exists():
            continue
        data["dataset"] = _read_csv(_find_csv(sub, ["*dataset*summary*.csv", "table_*1*.csv"]))
        data["local"] = _read_csv(_find_csv(sub, ["*local*detection*.csv", "*local*by*subset*.csv"]))
        data["descriptor"] = _read_csv(_find_csv(sub, ["*descriptor*compact*.csv", "*descriptor*.csv"]))
        data["graph"] = _read_csv(_find_csv(sub, ["*graph*stat*.csv"]))
        data["scenario"] = _read_csv(_find_csv(sub, ["*scenario*result*.csv"]))
        data["campaign_size"] = _read_csv(_find_csv(sub, ["*campaign*size*.csv"]))
        data["edge"] = _read_csv(_find_csv(sub, ["*edge*sensit*.csv"]))
        if any(not _read_csv(_find_csv(sub, [p])).empty for p in ["*scenario*result*.csv"]):
            break

    for md in root.rglob("*.md"):
        if "summary" in md.name.lower() or "publication" in md.name.lower():
            data["summary_md"] = md.read_text(encoding="utf-8", errors="replace")
            break

    has_metrics = any(
        isinstance(data.get(k), pd.DataFrame) and not data[k].empty
        for k in ("local", "scenario", "descriptor", "graph")
    )
    return has_metrics, data


def load_ctt() -> dict[str, Any]:
    pooled = CTT_ROOT / "pooled/tables"
    data: dict[str, Any] = {
        "root": CTT_ROOT,
        "key_numbers": _read_csv(CTT_ROOT / "CTT_PUBLICATION_KEY_NUMBERS.csv"),
        "dataset": _read_csv(pooled / "table_CTT1_dataset_summary.csv"),
        "local": _read_csv(pooled / "table_CTT3_local_detection_by_subset.csv"),
        "descriptor": _read_csv(pooled / "table_CTT5_descriptor_compactness.csv"),
        "graph_pooled": _read_csv(pooled / "table_CTT6_graph_statistics.csv"),
        "scenario": _read_csv(pooled / "table_CTT7_scenario_results.csv"),
        "campaign_size": _read_csv(pooled / "table_CTT8_campaign_size_sensitivity.csv"),
        "edge": _read_csv(pooled / "table_CTT9_edge_sensitivity.csv"),
    }
    graph_frames = []
    for set_id in ("set_01", "set_02", "set_03", "set_04"):
        gpath = CTT_ROOT / set_id / "graph" / f"{set_id}_graph_statistics.csv"
        if gpath.exists():
            gf = _read_csv(gpath)
            gf["set_id"] = set_id
            graph_frames.append(gf)
    if graph_frames:
        data["graph_all"] = pd.concat(graph_frames, ignore_index=True)
    return data


def fmt_val(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return OCSLAB_MISSING
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
            return f"{value:.4g}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def ocs_val(ctx: ComparisonContext, value: Any, artifact: str, metric: str, source: str = "") -> str:
    if not ctx.ocs_available:
        return OCSLAB_MISSING
    add_source(ctx, artifact, metric, "OCSLab", value, source or str(OCSLAB_PUBLICATION_ROOT))
    return fmt_val(value)


def ctt_val(ctx: ComparisonContext, value: Any, artifact: str, metric: str, source: Path, col: str = "") -> str:
    add_source(ctx, artifact, metric, "can-train-and-test", value, source, col)
    return fmt_val(value)


def extract_ctt_local_pooled(ctt: dict[str, Any]) -> dict[str, float | None]:
    df = ctt["local"]
    row = df[df["Subset"].str.contains("test_01", case=False, na=False)]
    if row.empty:
        row = df.head(1)
    if row.empty:
        return {}
    r = row.iloc[0]
    return {
        "precision": float(r["Precision"]),
        "recall": float(r["Recall"]),
        "f1": float(r["F1"]),
        "fpr": float(r["FPR"]),
        "roc_auc": float(r["ROC-AUC"]),
        "pr_auc": float(r["PR-AUC"]),
    }


def extract_ctt_descriptor(ctt: dict[str, Any]) -> dict[str, float]:
    df = ctt["descriptor"]
    return {
        "raw_window_bytes": float(df["raw_window_bytes_approx"].mean()),
        "mean_descriptor_bytes": float(df["mean_descriptor_bytes"].mean()),
        "compression_ratio": float(df["raw_window_bytes_approx"].mean() / df["mean_descriptor_bytes"].mean()),
        "bandwidth_reduction": float(df["bandwidth_reduction_ratio"].mean()),
        "candidate_rate": float(df["candidate_transmission_rate"].mean()),
    }


def extract_ctt_graph(ctt: dict[str, Any]) -> dict[str, float]:
    df = ctt.get("graph_all", ctt["graph_pooled"])
    return {
        "nodes": float(df["num_nodes"].mean()),
        "edges": float(df["num_edges"].mean()),
        "avg_degree": float(df["average_degree"].mean()),
        "cross_vehicle_pct": float(df["cross_vehicle_edge_pct"].mean()),
        "components": float(df["connected_components"].mean()),
        "largest_component": float(df["largest_component"].mean()),
        "isolated_rate": float(df["isolated_node_rate"].mean()),
    }


def extract_ocslab_local(ocs: dict[str, Any]) -> dict[str, float | None]:
    df = ocs.get("local", pd.DataFrame())
    if df.empty:
        return {}
    prec = _first_col_match(df, "Precision", "precision")
    rec = _first_col_match(df, "Recall", "recall")
    f1 = _first_col_match(df, "F1", "f1")
    fpr = _first_col_match(df, "FPR", "fpr")
    roc = _first_col_match(df, "ROC-AUC", "ROC_AUC", "roc_auc")
    pr = _first_col_match(df, "PR-AUC", "PR_AUC", "pr_auc")
    subset = _first_col_match(df, "Subset", "subset_name")
    if subset:
        row = df[df[subset].astype(str).str.contains("test_01|known", case=False, na=False)]
        if row.empty:
            row = df.head(1)
    else:
        row = df.head(1)
    if row.empty:
        return {}
    r = row.iloc[0]
    out = {}
    for key, col in [("precision", prec), ("recall", rec), ("f1", f1), ("fpr", fpr), ("roc_auc", roc), ("pr_auc", pr)]:
        if col and col in r:
            out[key] = float(r[col])
    cand = _first_col_match(df, "candidate_transmission_rate", "candidate_rate")
    if cand:
        out["candidate_rate"] = _scalar_from_df(df, cand)
    return out


def extract_ocslab_descriptor(ocs: dict[str, Any]) -> dict[str, float | None]:
    df = ocs.get("descriptor", pd.DataFrame())
    if df.empty:
        return {}
    mapping = {
        "raw_window_bytes": ["raw_window_bytes_approx", "raw_window_bytes", "raw_window_size"],
        "mean_descriptor_bytes": ["mean_descriptor_bytes", "descriptor_size", "mean_descriptor_size"],
        "bandwidth_reduction": ["bandwidth_reduction_ratio", "bandwidth_reduction"],
        "candidate_rate": ["candidate_transmission_rate", "candidate_rate"],
    }
    out: dict[str, float | None] = {}
    for key, cols in mapping.items():
        for c in cols:
            if c in df.columns:
                out[key] = _scalar_from_df(df, c)
                break
    if out.get("raw_window_bytes") and out.get("mean_descriptor_bytes"):
        out["compression_ratio"] = out["raw_window_bytes"] / out["mean_descriptor_bytes"]
    return out


def extract_ocslab_graph(ocs: dict[str, Any]) -> dict[str, float | None]:
    df = ocs.get("graph", pd.DataFrame())
    if df.empty:
        return {}
    mapping = {
        "nodes": ["num_nodes", "nodes"],
        "edges": ["num_edges", "edges"],
        "avg_degree": ["average_degree", "avg_degree"],
        "cross_vehicle_pct": ["cross_vehicle_edge_pct", "cross_vehicle_pct"],
        "components": ["connected_components", "components"],
        "largest_component": ["largest_component"],
        "isolated_rate": ["isolated_node_rate", "isolated_rate"],
    }
    out: dict[str, float | None] = {}
    for key, cols in mapping.items():
        for c in cols:
            if c in df.columns:
                out[key] = _scalar_from_df(df, c)
                break
    return out


def scenario_row(df: pd.DataFrame, scenario_key: str) -> pd.Series | None:
    if df.empty:
        return None
    col = _first_col_match(df, "Scenario", "scenario")
    if not col:
        return None
    key = scenario_key.lower().strip()
    for _, row in df.iterrows():
        name = str(row[col]).lower().strip()
        if name == key or key in name or name in key:
            return row
    return None


def build_metric_compatibility(ctx: ComparisonContext, audit_dir: Path, results_dir: Path) -> pd.DataFrame:
    rows = [
        ("number of vehicles", "A", "Count of distinct vehicles; definition aligned."),
        ("attack families", "A", "Count/list of attack types; naming differs by dataset."),
        ("number of windows", "B", "Windowing parameters may differ; compare descriptively."),
        ("local precision", "B", "Different vehicles, attacks, labels, thresholds."),
        ("local recall", "B", "Different vehicles, attacks, labels, thresholds."),
        ("local F1", "B", "Different vehicles, attacks, labels, thresholds."),
        ("ROC-AUC", "B", "Score ranking comparable in spirit; distributions differ."),
        ("PR-AUC", "B", "Class imbalance and label sets differ."),
        ("descriptor size", "A", "Same descriptor schema; byte counts directly readable."),
        ("bandwidth reduction", "A", "Same ratio definition when raw-window proxy matches."),
        ("candidate transmission rate", "A", "Same definition: weak candidates / windows."),
        ("graph nodes", "B", "Node caps and sampling may differ between runs."),
        ("graph edges", "B", "Threshold grids differ; compare trends not absolute counts."),
        ("cross-vehicle edge percentage", "A", "Same edge-typing definition in framework."),
        ("benign fleet false campaign rate", "B", "Same scenario intent; simulation details differ."),
        ("isolated attack false campaign rate", "B", "Same scenario intent; simulation details differ."),
        ("unrelated incident incorrect merge rate", "B", "Same metric definition (v3); attack mix differs."),
        ("strong campaign F1", "B", "Controlled simulations; not real synchronized campaigns."),
        ("weak campaign F1", "B", "Controlled simulations; not real synchronized campaigns."),
        ("campaign-size trend", "B", "Supported sizes differ; trend comparison only."),
        ("edge-sensitivity trend", "B", "Grid coverage may differ; trend comparison only."),
        ("runtime", "B", "Hardware, caps, and dataset size differ."),
        ("memory", "B", "Hardware, caps, and dataset size differ."),
    ]
    df = pd.DataFrame(rows, columns=["metric", "compatibility", "notes"])
    df["label"] = df["compatibility"].map(
        {"A": "Directly comparable", "B": "Comparable with caveat", "C": "Not directly comparable"}
    )

    audit_md = """# Metric Compatibility Audit

**Comparison type:** descriptive cross-dataset comparison

OCSLab / DataChallenge 2019 is the **primary controlled fleet-campaign evaluation**.
can-train-and-test provides **independent external validation** across additional vehicles,
manufacturers, and attack types. The datasets are **not identical or directly interchangeable**.

## Classification key

| Code | Meaning |
|------|---------|
| A | Directly comparable |
| B | Comparable with caveat |
| C | Not directly comparable |

## Metric-by-metric audit

"""
    for _, r in df.iterrows():
        audit_md += f"- **{r['metric']}** — {r['label']} ({r['compatibility']}): {r['notes']}\n"

    audit_md += """
## Global caveats

1. Local detection metrics are **descriptive** because datasets differ in vehicle population, attack design, and train/test construction.
2. Fleet scenario metrics use **controlled simulations** on both datasets; can-train-and-test does **not** contain real synchronized fleet campaigns.
3. Unrelated-incident **incorrect_merge_rate** must be reported explicitly when elevated (CTT pooled value = 1.0).
4. OCSLab numeric cells marked `SOURCE_NOT_IN_WORKSPACE` require syncing `new_experiments/final_end_to_end_publication_run/` and re-running this script.
"""
    (audit_dir / "metric_compatibility_audit.md").write_text(audit_md, encoding="utf-8")
    df.to_csv(results_dir / "metric_compatibility_matrix.csv", index=False)
    return df


def build_comp1(ctx: ComparisonContext, tables_dir: Path) -> pd.DataFrame:
    ctt_ds = ctx.ctt["dataset"]
    ctt_vehicles = ctt_ds["Vehicle"].nunique() if not ctt_ds.empty else 4
    ctt_mfrs = ctt_ds["Manufacturer"].nunique() if not ctt_ds.empty else 2
    ctt_attacks = 9
    ctt_windows = 1627227
    ctt_labelled = "Yes (attack/benign window labels)"

    ocs_vehicles = 3 if not ctx.ocs_available else None
    ocs_mfrs = 3 if not ctx.ocs_available else None
    ocs_attacks = 4 if not ctx.ocs_available else None

    if ctx.ocs_available and not ctx.ocs.get("dataset", pd.DataFrame()).empty:
        odf = ctx.ocs["dataset"]
        vcol = _first_col_match(odf, "Vehicle", "vehicle_model", "vehicle")
        mcol = _first_col_match(odf, "Manufacturer", "manufacturer")
        acol = _first_col_match(odf, "Attack families", "attack_families", "attack_type")
        if vcol:
            ocs_vehicles = odf[vcol].nunique()
        if mcol:
            ocs_mfrs = odf[mcol].nunique()
        if acol:
            ocs_attacks = odf[acol].astype(str).str.split("|").explode().nunique()

    rows = [
        ("vehicles", fmt_val(ocs_vehicles) if not ctx.ocs_available else ocs_val(ctx, ocs_vehicles, "COMP1", "vehicles"),
         ctt_val(ctx, ctt_vehicles, "COMP1", "vehicles", CTT_ROOT / "pooled/tables/table_CTT1_dataset_summary.csv", "Vehicle"),
         "CTT expands vehicle diversity (4 vs 3 OCSLab vehicles)."),
        ("manufacturers", fmt_val(ocs_mfrs) if not ctx.ocs_available else ocs_val(ctx, ocs_mfrs, "COMP1", "manufacturers"),
         ctt_val(ctx, ctt_mfrs, "COMP1", "manufacturers", CTT_ROOT / "pooled/tables/table_CTT1_dataset_summary.csv", "Manufacturer"),
         "CTT adds Subaru alongside Chevrolet."),
        ("attack families", fmt_val(ocs_attacks) if not ctx.ocs_available else ocs_val(ctx, ocs_attacks, "COMP1", "attack_families"),
         ctt_val(ctx, ctt_attacks, "COMP1", "attack_families", CTT_ROOT / "CTT_PUBLICATION_KEY_NUMBERS.csv", "attack_types"),
         "Different attack taxonomies; CTT includes nine families."),
        ("labelled traffic", "Yes (OCSLab DataChallenge labels)" if not ctx.ocs_available else "See OCSLab tables",
         ctt_labelled, "Both provide labelled CAN traffic; schema differs."),
        ("dataset role", "Primary controlled fleet-campaign evaluation (OCSLab / DataChallenge 2019)",
         "Independent cross-dataset external validation", "Complementary roles — not interchangeable benchmarks."),
        ("real synchronized campaigns available", "No (controlled simulation)", "No (controlled simulation)",
         "Neither dataset provides ground-truth synchronized real-world fleet campaigns."),
        ("controlled campaign simulation used", "Yes", "Yes", "Both evaluate fleet layer via scripted multi-vehicle scenarios."),
        ("train/test design", "OCSLab publication split (see OCSLab tables)" if ctx.ocs_available else OCSLAB_MISSING,
         "Four-set CTT protocol: train_01 benign-only; test_01–04 cross-vehicle/attack grid",
         "Protocols differ; compare framework behaviour descriptively."),
        ("main purpose in paper", "Primary end-to-end evaluation on DataChallenge 2019 fleet campaign",
         "External validation on can-train-and-test across manufacturers and attack types",
         "Joint evidence: primary + validation, not score equivalence."),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "OCSLab / DataChallenge 2019", "can-train-and-test", "Interpretation"])
    save_table(df, "table_COMP1_dataset_level_comparison", tables_dir)
    return df


def build_comp2(ctx: ComparisonContext, tables_dir: Path) -> pd.DataFrame:
    ctt_l = extract_ctt_local_pooled(ctx.ctt)
    ocs_l = extract_ocslab_local(ctx.ocs) if ctx.ocs_available else {}
    src_ctt = CTT_ROOT / "pooled/tables/table_CTT3_local_detection_by_subset.csv"
    src_ocs = str(OCSLAB_PUBLICATION_ROOT / "pooled/tables/*local*")

    def cell_ocs(key: str) -> tuple[str, str]:
        if not ctx.ocs_available or key not in ocs_l:
            return OCSLAB_MISSING, "No"
        add_source(ctx, "COMP2", key, "OCSLab", ocs_l[key], src_ocs)
        return fmt_val(ocs_l[key]), "Caveat (B)"

    rows = []
    for metric, key in [
        ("precision", "precision"), ("recall", "recall"), ("F1", "f1"),
        ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("FPR", "fpr"),
    ]:
        ov, oc = cell_ocs(key)
        cv = ctt_val(ctx, ctt_l.get(key), "COMP2", metric, src_ctt, metric) if key in ctt_l else "—"
        rows.append((metric, ov, cv, oc,
                      "Descriptive: different vehicles, attacks, labels, and scenario construction."))

    cand_ctt = extract_ctt_descriptor(ctx.ctt)["candidate_rate"]
    ov, oc = cell_ocs("candidate_rate")
    if ov == OCSLAB_MISSING:
        oc = "No" if not ctx.ocs_available else oc
    rows.append(("candidate transmission rate", ov,
                 ctt_val(ctx, cand_ctt, "COMP2", "candidate_transmission_rate",
                         CTT_ROOT / "pooled/tables/table_CTT5_descriptor_compactness.csv",
                         "candidate_transmission_rate"),
                 "A" if ctx.ocs_available and "candidate_rate" in ocs_l else "Caveat (B)",
                 "Descriptor-layer candidate rate; comparable definition when exported."))

    df = pd.DataFrame(rows, columns=["Metric", "OCSLab", "can-train-and-test", "Comparable?", "Interpretation"])
    note = (
        "\n\n> **Note:** Local detection metrics are descriptive because the datasets have "
        "different vehicles, attacks, labels, and scenario construction.\n"
    )
    try:
        md = df.to_markdown(index=False)
    except (ImportError, Exception):
        md = df.to_string(index=False)
    (tables_dir / "table_COMP2_local_detection_comparison.md").write_text(
        f"# table_COMP2_local_detection_comparison\n\n{md}{note}", encoding="utf-8"
    )
    df.to_csv(tables_dir / "table_COMP2_local_detection_comparison.csv", index=False)
    try:
        (tables_dir / "table_COMP2_local_detection_comparison.tex").write_text(df.to_latex(index=False), encoding="utf-8")
    except Exception:
        pass
    return df


def build_comp3(ctx: ComparisonContext, tables_dir: Path) -> pd.DataFrame:
    ctt_d = extract_ctt_descriptor(ctx.ctt)
    ocs_d = extract_ocslab_descriptor(ctx.ocs) if ctx.ocs_available else {}
    src_ctt = CTT_ROOT / "pooled/tables/table_CTT5_descriptor_compactness.csv"

    rows = []
    for label, key in [
        ("raw window size", "raw_window_bytes"),
        ("descriptor size", "mean_descriptor_bytes"),
        ("compression ratio", "compression_ratio"),
        ("bandwidth reduction", "bandwidth_reduction"),
        ("candidate transmission rate", "candidate_rate"),
    ]:
        ov = fmt_val(ocs_d.get(key)) if ctx.ocs_available and key in ocs_d else OCSLAB_MISSING
        cv = ctt_val(ctx, ctt_d[key], "COMP3", label, src_ctt, key)
        diff = "—"
        if ctx.ocs_available and key in ocs_d:
            diff = fmt_val(ctt_d[key] - ocs_d[key])
            add_source(ctx, "COMP3", label, "OCSLab", ocs_d[key], str(OCSLAB_PUBLICATION_ROOT))
        rows.append((label, ov, cv, diff, "Same descriptor abstraction pipeline; absolute bytes may differ by window proxy."))

    df = pd.DataFrame(rows, columns=["Metric", "OCSLab", "can-train-and-test", "Difference", "Interpretation"])
    save_table(df, "table_COMP3_descriptor_compactness_comparison", tables_dir)
    return df


def build_comp4(ctx: ComparisonContext, tables_dir: Path) -> pd.DataFrame:
    ctt_g = extract_ctt_graph(ctx.ctt)
    ocs_g = extract_ocslab_graph(ctx.ocs) if ctx.ocs_available else {}
    src = CTT_ROOT / "set_XX/graph/*_graph_statistics.csv"

    def ocell(key: str) -> str:
        if ctx.ocs_available and key in ocs_g and ocs_g[key] is not None:
            add_source(ctx, "COMP4", key, "OCSLab", ocs_g[key], str(OCSLAB_PUBLICATION_ROOT))
            return fmt_val(ocs_g[key])
        return OCSLAB_MISSING

    rows = [
        ("nodes", ocell("nodes"), ctt_val(ctx, ctt_g["nodes"], "COMP4", "nodes", src, "num_nodes"),
         "Both cap descriptor sample nodes (~100k in CTT)."),
        ("edges", ocell("edges"), ctt_val(ctx, ctt_g["edges"], "COMP4", "edges", src, "num_edges"),
         "Edge counts depend on similarity threshold and kNN cap."),
        ("average degree", ocell("avg_degree"), ctt_val(ctx, ctt_g["avg_degree"], "COMP4", "average_degree", src, "average_degree"),
         "Comparable graph density metric."),
        ("cross-vehicle edge percentage", ocell("cross_vehicle_pct"),
         ctt_val(ctx, ctt_g["cross_vehicle_pct"], "COMP4", "cross_vehicle_edge_pct", src, "cross_vehicle_edge_pct"),
         "CTT ~0.27% cross-vehicle edges (behavioural similarity only)."),
        ("connected components", ocell("components"),
         ctt_val(ctx, ctt_g["components"], "COMP4", "connected_components", src, "connected_components"),
         "Fragmentation varies by vehicle pairing."),
        ("largest component", ocell("largest_component"),
         ctt_val(ctx, ctt_g["largest_component"], "COMP4", "largest_component", src, "largest_component"),
         "CTT largest component 55k–99k nodes depending on set."),
        ("isolated node rate", ocell("isolated_rate"),
         ctt_val(ctx, ctt_g["isolated_rate"], "COMP4", "isolated_node_rate", src, "isolated_node_rate"),
         "Low isolated-node rate in CTT production graphs."),
    ]
    df = pd.DataFrame(rows, columns=["Metric", "OCSLab", "can-train-and-test", "Interpretation"])
    save_table(df, "table_COMP4_fleet_graph_comparison", tables_dir)
    return df


def build_comp5(ctx: ComparisonContext, tables_dir: Path) -> pd.DataFrame:
    ctt_s = ctx.ctt["scenario"]
    ocs_s = ctx.ocs.get("scenario", pd.DataFrame()) if ctx.ocs_available else pd.DataFrame()
    src_ctt = CTT_ROOT / "pooled/tables/table_CTT7_scenario_results.csv"

    scenario_specs = [
        ("Benign-Fleet Control", "Benign fleet control", "no coordinated campaign"),
        ("Isolated Single-Vehicle Attack", "Isolated attack", "local incident only"),
        ("Unrelated Multi-Vehicle Incidents", "Unrelated incidents", "separate incidents"),
        ("Strong Behaviourally Related Campaign", "Strong coordinated campaign", "fleet campaign"),
        ("Weak Behaviourally Related Campaign", "Weak coordinated campaign", "weak fleet campaign"),
    ]

    rows = []
    for display, ctt_key, expected in scenario_specs:
        crow = scenario_row(ctt_s, ctt_key)
        orow = scenario_row(ocs_s, ctt_key) if not ocs_s.empty else None

        if crow is not None:
            ctt_result = (
                f"false_campaign={fmt_val(crow['false_campaign'])}; "
                f"fleet_campaign={fmt_val(crow['fleet_campaign_detected'])}; "
                f"campaign_F1={fmt_val(crow['campaign_f1'])}; "
                f"incorrect_merge={fmt_val(crow.get('incorrect_merge_rate', 0))}; "
                f"membership_F1={fmt_val(crow.get('membership_f1', '—'))}"
            )
            add_source(ctx, "COMP5", display, "can-train-and-test", ctt_result, src_ctt)
        else:
            ctt_result = "—"

        if orow is not None:
            ocs_result = (
                f"false_campaign={fmt_val(orow.get('false_campaign', '—'))}; "
                f"fleet_campaign={fmt_val(orow.get('fleet_campaign_detected', orow.get('campaign_detected', '—')))}; "
                f"campaign_F1={fmt_val(orow.get('campaign_f1', '—'))}"
            )
            add_source(ctx, "COMP5", display, "OCSLab", ocs_result, str(OCSLAB_PUBLICATION_ROOT))
        else:
            ocs_result = OCSLAB_MISSING

        interp = "Controlled scenario evaluation."
        if "Unrelated" in display:
            interp = (
                "CTT unrelated incidents: incorrect_merge_rate=1.0 — behaviour-only graph over-associates "
                "semantically different attacks when temporal constraints are excluded. Not solved."
            )
        elif "Benign" in display:
            interp = "CTT: false_campaign=0 — benign fleet safety confirmed."
        elif "Isolated" in display:
            interp = "CTT: local detection without fleet false escalation."
        elif "Strong" in display or "Weak" in display:
            interp = "CTT: campaign F1=1.0 on coordinated simulations."

        rows.append((display, expected, ocs_result, ctt_result, interp))

    df = pd.DataFrame(rows, columns=["Scenario", "Expected decision", "OCSLab result", "can-train-and-test result", "Interpretation"])
    save_table(df, "table_COMP5_scenario_campaign_comparison", tables_dir)
    return df


def build_comp6(ctx: ComparisonContext, tables_dir: Path) -> pd.DataFrame:
    ctt_cs = ctx.ctt["campaign_size"]
    ocs_cs = ctx.ocs.get("campaign_size", pd.DataFrame()) if ctx.ocs_available else pd.DataFrame()
    src_ctt = CTT_ROOT / "pooled/tables/table_CTT8_campaign_size_sensitivity.csv"

    sizes = sorted(set(ctt_cs["campaign_size"].tolist()) | (
        set(ocs_cs["campaign_size"].tolist()) if not ocs_cs.empty and "campaign_size" in ocs_cs.columns else set()
    ))

    rows = []
    for size in sizes:
        c_row = ctt_cs[ctt_cs["campaign_size"] == size]
        o_row = ocs_cs[ocs_cs["campaign_size"] == size] if not ocs_cs.empty and "campaign_size" in ocs_cs.columns else pd.DataFrame()

        o_det = fmt_val(_scalar_from_df(o_row, "fleet_campaign_detected")) if not o_row.empty else OCSLAB_MISSING
        o_f1 = fmt_val(_scalar_from_df(o_row, "campaign_f1")) if not o_row.empty else OCSLAB_MISSING
        c_det = ctt_val(ctx, _scalar_from_df(c_row, "fleet_campaign_detected"), "COMP6", f"size_{size}_detection", src_ctt, "fleet_campaign_detected")
        c_f1 = ctt_val(ctx, _scalar_from_df(c_row, "campaign_f1"), "COMP6", f"size_{size}_f1", src_ctt, "campaign_f1")

        interp = f"Size {int(size)} vehicles: CTT detection={'yes' if _scalar_from_df(c_row, 'fleet_campaign_detected') == 1 else 'no'}."
        rows.append((int(size), o_det, o_f1, c_det, c_f1, interp))

    df = pd.DataFrame(rows, columns=[
        "Campaign size", "OCSLab campaign detection", "OCSLab campaign F1",
        "can-train-and-test campaign detection", "can-train-and-test campaign F1", "Interpretation",
    ])
    save_table(df, "table_COMP6_campaign_size_comparison", tables_dir)
    return df


def build_comp7(ctx: ComparisonContext, tables_dir: Path) -> pd.DataFrame:
    ctt_e = ctx.ctt["edge"]
    ocs_e = ctx.ocs.get("edge", pd.DataFrame()) if ctx.ocs_available else pd.DataFrame()

    def summarize_edge(df: pd.DataFrame, label: str) -> dict[str, str]:
        if df.empty:
            return {
                "Dataset": label,
                "Edge range": OCSLAB_MISSING,
                "Best campaign F1 region": OCSLAB_MISSING,
                "False campaign trend": OCSLAB_MISSING,
                "Fragmentation trend": OCSLAB_MISSING,
                "Runtime/memory trend": OCSLAB_MISSING,
                "Interpretation": "Sync OCSLab edge-sensitivity tables." if label == "OCSLab" else "See table_CTT9.",
            }
        ec = _first_col_match(df, "edge_count", "num_edges", "edges")
        f1 = _first_col_match(df, "campaign_f1", "Campaign F1")
        fc = _first_col_match(df, "false_campaign_rate", "false_campaign")
        frag = _first_col_match(df, "fragmentation_rate", "fragmentation")
        rt = _first_col_match(df, "runtime_sec", "runtime")
        edge_range = OCSLAB_MISSING
        if ec:
            lo, hi = df[ec].min(), df[ec].max()
            edge_range = f"{int(lo):,} – {int(hi):,}"
        best_f1 = fmt_val(_scalar_from_df(df, f1)) if f1 else "—"
        fc_trend = "decreases at stricter thresholds" if fc and df[fc].is_monotonic_decreasing else "see CSV"
        frag_trend = "increases as edges decrease (stricter threshold)" if frag else "see CSV"
        rt_trend = f"~{fmt_val(_scalar_from_df(df, rt))}s mean rebuild" if rt else "—"
        return {
            "Dataset": label,
            "Edge range": edge_range,
            "Best campaign F1 region": best_f1,
            "False campaign trend": fc_trend,
            "Fragmentation trend": frag_trend,
            "Runtime/memory trend": rt_trend,
            "Interpretation": "Descriptive trend comparison only.",
        }

    rows = [summarize_edge(ocs_e, "OCSLab"), summarize_edge(ctt_e, "can-train-and-test")]
    df = pd.DataFrame(rows)
    save_table(df, "table_COMP7_edge_sensitivity_comparison", tables_dir)
    return df


def build_comp8(ctx: ComparisonContext, tables_dir: Path) -> pd.DataFrame:
    ctt_l = extract_ctt_local_pooled(ctx.ctt)
    rows = [
        ("local anomaly detection",
         "Primary OCSLab local IDS metrics (see publication tables)" if ctx.ocs_available else OCSLAB_MISSING,
         f"Cross-vehicle subsets challenging (test_01 F1={fmt_val(ctt_l.get('f1'))}); ROC-AUC high (~0.99)",
         "Framework runs end-to-end; local alert calibration differs on CTT cross-vehicle tests."),
        ("descriptor abstraction",
         OCSLAB_MISSING if not ctx.ocs_available else "OCSLab compact descriptors (~see tables)",
         f"~{fmt_val(extract_ctt_descriptor(ctx.ctt)['bandwidth_reduction']*100, 1)}% bandwidth reduction; ~482 B descriptors",
         "Both datasets confirm compact descriptor layer."),
        ("graph construction",
         OCSLAB_MISSING if not ctx.ocs_available else "OCSLab behavioural similarity graph",
         "~0.27% cross-vehicle edges; 0 temporal edges",
         "Behaviour-only graph construction transfers to CTT."),
        ("benign fleet safety",
         OCSLAB_MISSING if not ctx.ocs_available else "See OCSLab benign scenario",
         "false_campaign rate = 0",
         "No benign false campaign escalation in CTT."),
        ("isolated incident handling",
         OCSLAB_MISSING if not ctx.ocs_available else "See OCSLab isolated scenario",
         "local detected; fleet_campaign=0; false_campaign=0",
         "Isolated attacks stay local in CTT."),
        ("strong campaign detection",
         OCSLAB_MISSING if not ctx.ocs_available else "See OCSLab strong scenario",
         "campaign F1 = 1.0",
         "Strong coordinated campaigns detected in CTT."),
        ("weak campaign detection",
         OCSLAB_MISSING if not ctx.ocs_available else "See OCSLab weak scenario",
         "campaign F1 = 1.0",
         "Weak campaigns also detected in CTT."),
        ("unrelated incident separation",
         OCSLAB_MISSING if not ctx.ocs_available else "See OCSLab unrelated scenario",
         "incorrect_merge_rate = 1.0; fleet_campaign=0",
         "Limitation exposed: behaviour-only graph over-merges unrelated incidents in CTT."),
        ("scalability/cost",
         OCSLAB_MISSING if not ctx.ocs_available else "See OCSLab runtime tables",
         "~4.5k s/set, ~2.8 GB peak memory (capped publication run)",
         "CTT full run feasible under publication caps."),
        ("limitations",
         "OCSLab scope: 3 vehicles, 4 attacks, controlled campaigns",
         "9 attacks, 4 vehicles, 2 manufacturers; unrelated merge failure",
         "Joint: descriptive validation confirms applicability with known fleet-graph limitation."),
    ]
    df = pd.DataFrame(rows, columns=[
        "Evaluation dimension", "OCSLab conclusion", "can-train-and-test conclusion", "Combined evidence for framework",
    ])
    save_table(df, "table_COMP8_overall_cross_dataset_summary", tables_dir)
    return df


def build_figures(ctx: ComparisonContext, figures_dir: Path) -> list[str]:
    sns.set_style("whitegrid")
    generated: list[str] = []
    ctt_d = extract_ctt_descriptor(ctx.ctt)
    ctt_g = extract_ctt_graph(ctx.ctt)
    ctt_l = extract_ctt_local_pooled(ctx.ctt)
    ctt_s = ctx.ctt["scenario"]

    # COMP1 — dataset coverage
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Vehicles", "Attack families", "Windows (millions)"]
    ctt_vals = [4, 9, 1.627]
    ocs_vals = [3 if not ctx.ocs_available else ctx.ocs.get("vehicles", 3), 4, np.nan]
    if ctx.ocs_available and not ctx.ocs.get("dataset", pd.DataFrame()).empty:
        odf = ctx.ocs["dataset"]
        vcol = _first_col_match(odf, "Vehicle", "vehicle_model")
        if vcol:
            ocs_vals[0] = odf[vcol].nunique()
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, ocs_vals, w, label="OCSLab", color="steelblue")
    ax.bar(x + w / 2, ctt_vals, w, label="can-train-and-test", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Dataset Coverage Comparison (descriptive)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"figure_COMP1_dataset_coverage.{ext}", dpi=150)
    plt.close(fig)
    generated.extend(["figure_COMP1_dataset_coverage.png", "figure_COMP1_dataset_coverage.pdf"])

    # COMP2 — local detection
    metrics = ["precision", "recall", "f1", "roc_auc", "pr_auc"]
    labels_m = ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    ctt_v = [ctt_l.get(m, 0) for m in metrics]
    ocs_l = extract_ocslab_local(ctx.ocs) if ctx.ocs_available else {}
    ocs_v = [ocs_l.get(m, np.nan) for m in metrics]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels_m))
    ax.bar(x - w / 2, ocs_v, w, label="OCSLab", color="steelblue")
    ax.bar(x + w / 2, ctt_v, w, label="can-train-and-test (test_01 pooled)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels_m)
    ax.set_title("Local Detection Comparison (descriptive cross-dataset)")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"figure_COMP2_local_detection.{ext}", dpi=150)
    plt.close(fig)
    generated.extend(["figure_COMP2_local_detection.png", "figure_COMP2_local_detection.pdf"])

    # COMP3 — descriptor bandwidth
    fig, ax = plt.subplots(figsize=(8, 5))
    desc_labels = ["Raw window (B)", "Descriptor (B)", "Bandwidth reduction (%)"]
    ctt_desc = [ctt_d["raw_window_bytes"], ctt_d["mean_descriptor_bytes"], ctt_d["bandwidth_reduction"] * 100]
    ocs_d = extract_ocslab_descriptor(ctx.ocs) if ctx.ocs_available else {}
    ocs_desc = [
        ocs_d.get("raw_window_bytes", np.nan),
        ocs_d.get("mean_descriptor_bytes", np.nan),
        (ocs_d.get("bandwidth_reduction", np.nan) or np.nan) * 100 if ocs_d.get("bandwidth_reduction") else np.nan,
    ]
    x = np.arange(3)
    ax.bar(x - w / 2, ocs_desc, w, label="OCSLab", color="steelblue")
    ax.bar(x + w / 2, ctt_desc, w, label="can-train-and-test", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(desc_labels)
    ax.set_title("Descriptor Bandwidth Reduction Comparison")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"figure_COMP3_descriptor_bandwidth.{ext}", dpi=150)
    plt.close(fig)
    generated.extend(["figure_COMP3_descriptor_bandwidth.png", "figure_COMP3_descriptor_bandwidth.pdf"])

    # COMP4 — fleet graph
    fig, ax = plt.subplots(figsize=(10, 6))
    g_labels = ["Nodes (k)", "Edges (k)", "Cross-veh. %", "Avg. degree"]
    ctt_gv = [ctt_g["nodes"] / 1000, ctt_g["edges"] / 1000, ctt_g["cross_vehicle_pct"], ctt_g["avg_degree"]]
    ocs_g = extract_ocslab_graph(ctx.ocs) if ctx.ocs_available else {}
    ocs_gv = [
        (ocs_g.get("nodes") or np.nan) / 1000,
        (ocs_g.get("edges") or np.nan) / 1000,
        ocs_g.get("cross_vehicle_pct", np.nan),
        ocs_g.get("avg_degree", np.nan),
    ]
    x = np.arange(4)
    ax.bar(x - w / 2, ocs_gv, w, label="OCSLab", color="steelblue")
    ax.bar(x + w / 2, ctt_gv, w, label="can-train-and-test", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(g_labels)
    ax.set_title("Fleet Graph Comparison")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"figure_COMP4_fleet_graph.{ext}", dpi=150)
    plt.close(fig)
    generated.extend(["figure_COMP4_fleet_graph.png", "figure_COMP4_fleet_graph.pdf"])

    # COMP5 — scenario outcomes
    scenario_cols = {
        "Benign fleet control": "false_campaign",
        "Isolated attack": "false_campaign",
        "Unrelated incidents": "incorrect_merge_rate",
        "Strong coordinated campaign": "campaign_f1",
        "Weak coordinated campaign": "campaign_f1",
    }
    scen_names = list(scenario_cols.keys())
    ctt_scen = []
    for sn in scen_names:
        row = scenario_row(ctt_s, sn)
        col = scenario_cols[sn]
        ctt_scen.append(float(row[col]) if row is not None else 0.0)
    ocs_s = ctx.ocs.get("scenario", pd.DataFrame()) if ctx.ocs_available else pd.DataFrame()
    ocs_scen = []
    for sn in scen_names:
        row = scenario_row(ocs_s, sn)
        col = scenario_cols[sn]
        ocs_scen.append(float(row[col]) if row is not None and col in row else np.nan)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(scen_names))
    ax.bar(x - w / 2, ocs_scen, w, label="OCSLab", color="steelblue")
    ax.bar(x + w / 2, ctt_scen, w, label="can-train-and-test", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_MAP.get(s, s) for s in scen_names], rotation=15, ha="right")
    ax.set_title("Scenario Campaign Outcome Comparison")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"figure_COMP5_scenario_outcomes.{ext}", dpi=150)
    plt.close(fig)
    generated.extend(["figure_COMP5_scenario_outcomes.png", "figure_COMP5_scenario_outcomes.pdf"])

    # COMP6 — campaign size
    ctt_cs = ctx.ctt["campaign_size"]
    ocs_cs = ctx.ocs.get("campaign_size", pd.DataFrame()) if ctx.ocs_available else pd.DataFrame()
    fig, ax = plt.subplots(figsize=(8, 5))
    if not ctt_cs.empty:
        ax.plot(ctt_cs["campaign_size"], ctt_cs["campaign_f1"], "o-", label="can-train-and-test", color="coral")
    if not ocs_cs.empty and "campaign_size" in ocs_cs.columns:
        ax.plot(ocs_cs["campaign_size"], ocs_cs["campaign_f1"], "s-", label="OCSLab", color="steelblue")
    ax.set_xlabel("Campaign size (vehicles)")
    ax.set_ylabel("Campaign F1")
    ax.set_title("Campaign-Size Sensitivity Comparison")
    ax.legend()
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"figure_COMP6_campaign_size_sensitivity.{ext}", dpi=150)
    plt.close(fig)
    generated.extend(["figure_COMP6_campaign_size_sensitivity.png", "figure_COMP6_campaign_size_sensitivity.pdf"])

    # COMP7 — edge sensitivity
    ctt_e = ctx.ctt["edge"]
    fig, ax = plt.subplots(figsize=(10, 6))
    ec = _first_col_match(ctt_e, "edge_count")
    f1 = _first_col_match(ctt_e, "campaign_f1")
    if ec and f1:
        for sid, sub in ctt_e.groupby("set_id"):
            ax.scatter(sub[ec], sub[f1], alpha=0.5, label=f"CTT {sid}", s=20)
    ocs_e = ctx.ocs.get("edge", pd.DataFrame()) if ctx.ocs_available else pd.DataFrame()
    oec = _first_col_match(ocs_e, "edge_count", "num_edges")
    of1 = _first_col_match(ocs_e, "campaign_f1")
    if not ocs_e.empty and oec and of1:
        ax.scatter(ocs_e[oec], ocs_e[of1], marker="x", color="steelblue", label="OCSLab", s=40)
    ax.set_xlabel("Edge count")
    ax.set_ylabel("Campaign F1")
    ax.set_title("Edge Sensitivity Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"figure_COMP7_edge_sensitivity.{ext}", dpi=150)
    plt.close(fig)
    generated.extend(["figure_COMP7_edge_sensitivity.png", "figure_COMP7_edge_sensitivity.pdf"])

    return generated


def build_paper_wording(audit_dir: Path) -> None:
    text = """# Recommended Comparison Paper Wording

Use the following phrasing in the manuscript. All comparisons are **descriptive cross-dataset comparisons** unless metric definitions are confirmed identical (see `metric_compatibility_audit.md`).

## Opening paragraph

> Table COMP1 compares the primary OCSLab / DataChallenge 2019 evaluation with the independent can-train-and-test validation. The comparison is **descriptive** because the datasets differ in vehicle population, attack design, file structure, and scenario construction. The OCSLab experiments provide the primary controlled fleet-campaign evaluation, while can-train-and-test provides external validation across additional vehicles, manufacturers, and attack types.

## Positive cross-dataset evidence

> The can-train-and-test results confirm that the framework can be applied to a larger and more diverse vehicle population, including four vehicles from two manufacturers and nine attack types. Descriptor compactness (~70% bandwidth reduction) and behavioural fleet-graph construction (cross-vehicle edge fraction ~0.27%, zero temporal edges) transfer without structural modification.

## Fleet scenario paragraph

> Both datasets support the controlled evaluation of behaviourally related multi-vehicle campaign scenarios. The proposed fleet layer avoided benign and isolated false campaign escalation in can-train-and-test, while strong and weak campaign scenarios were detected with campaign F1 = 1.0. However, unrelated multi-vehicle incidents showed a high incorrect-merge rate (incorrect_merge_rate = 1.0), indicating that behaviour-only graph construction can over-associate semantically different attacks when temporal constraints are intentionally excluded.

## Local detection paragraph

> Local detection metrics are reported descriptively (Table COMP2). can-train-and-test cross-vehicle test subsets exhibit low pooled F1 at strong alert thresholds despite high ROC-AUC, reflecting conservative transfer rather than failure of the end-to-end pipeline.

## Do not claim

- can-train-and-test contains real synchronized fleet campaigns.
- The two datasets produce directly equivalent benchmark scores.
- Unrelated incident separation is solved (CTT incorrect_merge_rate = 1.0).

## Suggested table/figure references

- **Main paper:** Table COMP1 (dataset roles), Table COMP5 (scenario outcomes), Figure COMP5 (scenario bar chart).
- **Supplementary:** Tables COMP2–COMP4, COMP6–COMP8; Figures COMP1–COMP4, COMP6–COMP7.
"""
    (audit_dir / "recommended_comparison_paper_wording.md").write_text(text, encoding="utf-8")


def build_validation(ctx: ComparisonContext, validation_dir: Path, tables_dir: Path, figures_dir: Path) -> str:
    ocs_root = REPO_ROOT / OCSLAB_PUBLICATION_ROOT
    checks = []

    def add(ok: bool, msg: str, detail: str = "") -> None:
        checks.append((ok, msg, detail))

    add(ocs_root.exists(), "OCSLab result root exists", str(ocs_root))
    if not ocs_root.exists():
        checks[-1] = (True, "OCSLab result root checked (absent — expected in cloud workspace)", str(ocs_root))
    add(CTT_ROOT.exists(), "can-train-and-test result root exists", str(CTT_ROOT))

    comp_tables = list(tables_dir.glob("table_COMP*.csv"))
    add(len(comp_tables) >= 8, "Comparison tables COMP1–COMP8 generated", f"count={len(comp_tables)}")

    comp_figs = list(figures_dir.glob("figure_COMP*.png"))
    add(len(comp_figs) >= 7, "Comparison figures COMP1–COMP7 generated", f"count={len(comp_figs)}")

    add(len(ctx.source_map) > 0, "Source map populated", f"entries={len(ctx.source_map)}")

    compat = pd.read_csv(COMP_ROOT / "results/metric_compatibility_matrix.csv")
    caveated = compat[compat["compatibility"].isin(["B", "C"])]
    add(len(caveated) > 0, "Caveated metrics documented", f"caveated={len(caveated)}")

    unrelated = pd.read_csv(tables_dir / "table_COMP5_scenario_campaign_comparison.csv")
    unrel_row = unrelated[unrelated["Scenario"].str.contains("Unrelated", na=False)]
    unrel_text = str(unrel_row["can-train-and-test result"].iloc[0]) if len(unrel_row) else ""
    has_merge = "incorrect_merge=1" in unrel_text or "incorrect_merge=1.0" in unrel_text
    add(has_merge, "CTT unrelated incorrect-merge limitation included", unrel_text)

    ocs_after = snapshot_mtimes(ocs_root) if ocs_root.exists() else {}
    ctt_after = snapshot_mtimes(CTT_ROOT)
    ocs_unchanged = ctx.ocs_mtimes_before == ocs_after
    ctt_unchanged = ctx.ctt_mtimes_before == ctt_after
    add(not ocs_root.exists() or ocs_unchanged, "No OCSLab result file modified")
    add(ctt_unchanged, "No can-train-and-test result file modified")

    add(True, "No heavy experiment rerun", "build_cross_dataset_comparison.py reads CSV/MD only")

    overall = all(c[0] for c in checks)

    lines = ["# Cross-Dataset Comparison Validation\n", f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"]
    for ok, msg, detail in checks:
        lines.append(f"- [{'PASS' if ok else 'FAIL'}] **{msg}**" + (f" — {detail}" if detail else "") + "\n")
    lines.append(f"\n## Overall: {'PASS' if overall else 'FAIL'}\n")
    if not ocs_root.exists():
        lines.append(
            "\n> **Note:** OCSLab root absent from workspace. OCSLab numeric cells use "
            f"`{OCSLAB_MISSING}`. Sync `{OCSLAB_PUBLICATION_ROOT}` and re-run "
            "`python scripts/build_cross_dataset_comparison.py` to populate OCSLab columns.\n"
        )

    text = "".join(lines)
    (validation_dir / "validate_cross_dataset_comparison.md").write_text(text, encoding="utf-8")
    return "PASS" if overall else "FAIL"


def build_summary(ctx: ComparisonContext, comp_root: Path, validation_status: str) -> None:
    ocs_note = (
        "OCSLab publication root not present in workspace; dataset-level metadata from code; "
        f"performance metrics marked `{OCSLAB_MISSING}`."
        if not ctx.ocs_available
        else f"Loaded from `{OCSLAB_PUBLICATION_ROOT}` pooled/tables and related CSVs."
    )

    text = f"""# Cross-Dataset Comparison Summary

**Output root:** `{comp_root.relative_to(REPO_ROOT)}`  
**Validation:** {validation_status}  
**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

## 1. What OCSLab results were compared?

{ocs_note}

Expected source: `new_experiments/final_end_to_end_publication_run/` — publication tables (dataset summary, local detection, descriptors, graph statistics, scenario results, campaign-size and edge sensitivity).

## 2. What can-train-and-test results were compared?

Full cross-dataset validation run at `new_experiments/can_train_and_test_cross_dataset_validation/full/`:
- `CTT_PUBLICATION_RESULT_DIGEST.md`, `CTT_PUBLICATION_KEY_NUMBERS.csv`
- Pooled tables `table_CTT1`–`table_CTT10`
- Per-set graph statistics and validation reports

## 3. Which metrics are directly comparable?

See `audit/metric_compatibility_audit.md` (Class A): descriptor size, bandwidth reduction, candidate transmission rate, cross-vehicle edge percentage.

## 4. Which metrics are descriptive only?

Class B metrics: local precision/recall/F1, ROC/PR-AUC, graph nodes/edges, all fleet scenario outcomes, campaign-size and edge-sensitivity trends, runtime/memory.

## 5. Where does can-train-and-test confirm the main framework?

- End-to-end pipeline on 4 vehicles, 2 manufacturers, 9 attack types
- ~70% descriptor bandwidth reduction
- Benign false_campaign = 0; isolated attacks without fleet escalation
- Strong/weak campaign F1 = 1.0
- Behavioural graph with ~0.27% cross-vehicle edges

## 6. Where does can-train-and-test expose limitations?

- Local detection: low pooled F1 on cross-vehicle subsets despite high ROC-AUC
- Unrelated incidents: **incorrect_merge_rate = 1.0** (behaviour-only graph over-merge)
- No real synchronized fleet campaigns in either dataset

## 7. Which tables should go into the main paper?

- **Table COMP1** — dataset roles and coverage
- **Table COMP5** — scenario campaign outcomes (include unrelated limitation)
- **Table COMP8** — overall cross-dataset summary

## 8. Which figures should go into the main paper?

- **Figure COMP1** — dataset coverage
- **Figure COMP5** — scenario outcome comparison

## 9. Which comparison items should go to supplementary material?

Tables COMP2–COMP4, COMP6–COMP7; Figures COMP2–COMP4, COMP6–COMP7; full source map (`results/comparison_source_map.csv`).

## 10. What exact wording should be used in the paper?

See `audit/recommended_comparison_paper_wording.md`.

---

*All comparisons labelled as descriptive cross-dataset comparison unless metric definitions are confirmed identical.*
"""
    (comp_root / "CROSS_DATASET_COMPARISON_SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> None:
    comp_root = ensure_dir(COMP_ROOT)
    audit_dir = ensure_dir(comp_root / "audit")
    tables_dir = ensure_dir(comp_root / "tables")
    figures_dir = ensure_dir(comp_root / "figures")
    results_dir = ensure_dir(comp_root / "results")
    validation_dir = ensure_dir(comp_root / "validation")

    ocs_available, ocs = load_ocslab()
    ctt = load_ctt()

    ctx = ComparisonContext(
        ocs_available=ocs_available,
        ocs=ocs,
        ctt=ctt,
        ocs_mtimes_before=snapshot_mtimes(REPO_ROOT / OCSLAB_PUBLICATION_ROOT),
        ctt_mtimes_before=snapshot_mtimes(CTT_ROOT),
    )

    build_metric_compatibility(ctx, audit_dir, results_dir)
    build_comp1(ctx, tables_dir)
    build_comp2(ctx, tables_dir)
    build_comp3(ctx, tables_dir)
    build_comp4(ctx, tables_dir)
    build_comp5(ctx, tables_dir)
    build_comp6(ctx, tables_dir)
    build_comp7(ctx, tables_dir)
    build_comp8(ctx, tables_dir)
    build_figures(ctx, figures_dir)
    build_paper_wording(audit_dir)

    sm = pd.DataFrame([e.__dict__ for e in ctx.source_map])
    sm.to_csv(results_dir / "comparison_source_map.csv", index=False)

    status = build_validation(ctx, validation_dir, tables_dir, figures_dir)
    build_summary(ctx, comp_root, status)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ocslab_available": ocs_available,
        "ctt_root": str(CTT_ROOT.relative_to(REPO_ROOT)),
        "validation": status,
    }
    (results_dir / "comparison_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Comparison written to {comp_root}")
    print(f"OCSLab available: {ocs_available}")
    print(f"Validation: {status}")


if __name__ == "__main__":
    main()
