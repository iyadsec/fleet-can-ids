#!/usr/bin/env python3
"""Assemble baseline/ablation comparison tables from existing OCSLab campaign-clustering runs.

Does not rerun experiments. Sources are read from ``origin/cursor/campaign-clustering``.
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "new_experiments" / "baseline_ablation_comparison"
SOURCE_REF = "origin/cursor/campaign-clustering"

# Paper ablation methods (user naming) -> framework method keys / table_06 labels
METHODS: list[dict[str, str]] = [
    {
        "paper_id": "M1_Local_IF_Only",
        "method_key": "local_ids",
        "table06_label": "M1 Local IDS",
        "display": "M1 Local IF only",
        "notes": "Vehicle-level IF only; campaign F1 N/A (no fleet layer)",
    },
    {
        "paper_id": "M2_Descriptor_Clustering_Only",
        "method_key": "descriptor_clustering",
        "table06_label": "M2 Descriptor clustering",
        "display": "M2 Descriptor clustering",
        "notes": "DBSCAN on descriptors; no GNN",
    },
    {
        "paper_id": "M3_GraphSAGE_Fleet_Model",
        "method_key": "fcgnn",
        "table06_label": "M4 Proposed FCGNN (GraphSAGEFleetCorrelator)",
        "display": "M3 GraphSAGE fleet",
        "notes": "Behavioural graph + GraphSAGE + clustering + campaign logic",
    },
]

PRIMARY_CAMPAIGN_SIZE = 5
PRIMARY_COORD_STRENGTH = 1.0
S2_COORD_STRENGTH = 0.0  # independent multi-vehicle attacks
REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]

SOURCE_FILES = {
    "table_06": "new_experiments/publication_ready/tables/table_06_method_ablation.csv",
    "S2_summary": "new_experiments/results/S2_non_coordinated/summary_mean_std.csv",
    "S3_summary": "new_experiments/results/S3_strong_campaign/summary_mean_std.csv",
    "S4_summary": "new_experiments/results/S4_weak_campaign/summary_mean_std.csv",
    "S0_summary": "new_experiments/results/S0_benign_control/summary_mean_std.csv",
    "S3_runs": "new_experiments/results/S3_strong_campaign/run_level_metrics.csv",
    "S4_runs": "new_experiments/results/S4_weak_campaign/run_level_metrics.csv",
    "S2_runs": "new_experiments/results/S2_non_coordinated/run_level_metrics.csv",
}


def git_show_text(ref: str, repo_path: str) -> str:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{repo_path}"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise FileNotFoundError(f"Missing {ref}:{repo_path}\n{proc.stderr}")
    return proc.stdout


def parse_pm(value: str) -> tuple[float | None, float | None]:
    """Parse '0.322 $\\pm$ 0.412' or plain floats."""
    if value is None or str(value).strip() in ("", "nan", "N/A"):
        return None, None
    text = str(value).strip()
    m = re.match(
        r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(?:\$\\pm\$|±|\+/-)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
        text,
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(text), 0.0
    except ValueError:
        return None, None


def load_table06() -> pd.DataFrame:
    raw = git_show_text(SOURCE_REF, SOURCE_FILES["table_06"])
    return pd.read_csv(io.StringIO(raw))


def load_summary(scenario_key: str) -> pd.DataFrame:
    path = {
        "S2": SOURCE_FILES["S2_summary"],
        "S3": SOURCE_FILES["S3_summary"],
        "S4": SOURCE_FILES["S4_summary"],
        "S0": SOURCE_FILES["S0_summary"],
    }[scenario_key]
    return pd.read_csv(io.StringIO(git_show_text(SOURCE_REF, path)))


def slice_summary(
    df: pd.DataFrame,
    *,
    method_key: str,
    campaign_size: int | float,
    coordination_strength: float,
) -> pd.Series | None:
    mask = (
        (df["method"] == method_key)
        & (df["campaign_size"].astype(float) == float(campaign_size))
        & (df["coordination_strength"].astype(float) == float(coordination_strength))
    )
    rows = df.loc[mask]
    if rows.empty:
        return None
    return rows.iloc[0]


@dataclass
class MetricCell:
    mean: float | None = None
    std: float | None = None
    source: str = ""
    note: str = ""


@dataclass
class MethodMetrics:
    paper_id: str
    display: str
    notes: str
    local_vehicle_recall: MetricCell = field(default_factory=MetricCell)
    strong_campaign_f1: MetricCell = field(default_factory=MetricCell)
    weak_campaign_f1: MetricCell = field(default_factory=MetricCell)
    independent_merge: MetricCell = field(default_factory=MetricCell)
    false_campaign_s0: MetricCell = field(default_factory=MetricCell)
    runtime_total_sec: MetricCell = field(default_factory=MetricCell)


def build_metrics() -> list[MethodMetrics]:
    table06 = load_table06()
    s2 = load_summary("S2")
    s3 = load_summary("S3")
    s4 = load_summary("S4")
    s0 = load_summary("S0")

    rows: list[MethodMetrics] = []
    for spec in METHODS:
        mm = MethodMetrics(paper_id=spec["paper_id"], display=spec["display"], notes=spec["notes"])

        # Local detection (vehicle recall) — table_06 S3 row; metric computed in framework ablation aggregate
        s3_t06 = table06[(table06["Scenario"] == "S3") & (table06["Method"] == spec["table06_label"])]
        if not s3_t06.empty:
            mean, std = parse_pm(s3_t06.iloc[0]["Vehicle recall"])
            mm.local_vehicle_recall = MetricCell(mean, std, SOURCE_FILES["table_06"], "S3 vehicle recall")

        # Campaign F1 — publication table_06 (corrected evaluation, campaign_size=5, coord=1.0)
        for scenario, attr in (("S3", "strong_campaign_f1"), ("S4", "weak_campaign_f1")):
            trow = table06[(table06["Scenario"] == scenario) & (table06["Method"] == spec["table06_label"])]
            if not trow.empty:
                mean, std = parse_pm(trow.iloc[0]["Campaign F1"])
                setattr(
                    mm,
                    attr,
                    MetricCell(mean, std, SOURCE_FILES["table_06"], f"{scenario} campaign F1"),
                )

        # S2 incorrect campaign merging — summary_mean_std, independent attacks (coord=0)
        s2_row = slice_summary(
            s2,
            method_key=spec["method_key"],
            campaign_size=PRIMARY_CAMPAIGN_SIZE,
            coordination_strength=S2_COORD_STRENGTH,
        )
        if s2_row is not None:
            mm.independent_merge = MetricCell(
                float(s2_row["incorrect_campaign_merging_mean"]),
                float(s2_row["incorrect_campaign_merging_std"]),
                SOURCE_FILES["S2_summary"],
                "S2 incorrect_campaign_merging (campaign_size=5, coord=0)",
            )

        # Benign false campaign rate (S0) when fleet layer exists
        s0_row = slice_summary(
            s0,
            method_key=spec["method_key"],
            campaign_size=0,
            coordination_strength=0.0,
        )
        if s0_row is not None:
            mm.false_campaign_s0 = MetricCell(
                float(s0_row["false_campaign_alert_rate_mean"]),
                float(s0_row["false_campaign_alert_rate_std"]),
                SOURCE_FILES["S0_summary"],
                "S0 false_campaign_alert_rate",
            )

        # End-to-end runtime — average of S3/S4 primary config slices
        rt_vals: list[float] = []
        for sdf in (s3, s4):
            row = slice_summary(
                sdf,
                method_key=spec["method_key"],
                campaign_size=PRIMARY_CAMPAIGN_SIZE,
                coordination_strength=PRIMARY_COORD_STRENGTH,
            )
            if row is not None and pd.notna(row.get("runtime_total_sec_mean")):
                rt_vals.append(float(row["runtime_total_sec_mean"]))
        if rt_vals:
            mm.runtime_total_sec = MetricCell(
                sum(rt_vals) / len(rt_vals),
                None,
                f"{SOURCE_FILES['S3_summary']} + {SOURCE_FILES['S4_summary']}",
                "Mean runtime_total_sec (S3/S4, campaign_size=5, coord=1.0)",
            )

        rows.append(mm)
    return rows


def fmt_pm(cell: MetricCell, *, na_for_zero_campaign_m1: bool = False) -> str:
    if cell.mean is None:
        return "N/A"
    if na_for_zero_campaign_m1 and cell.mean == 0.0 and cell.std == 0.0:
        return "N/A"
    if cell.std is None or cell.std == 0.0:
        return f"{cell.mean:.3f}"
    return f"{cell.mean:.3f} $\\pm$ {cell.std:.3f}"


def metrics_to_dataframe(method_rows: list[MethodMetrics]) -> pd.DataFrame:
    records = []
    for mm in method_rows:
        is_m1 = mm.paper_id == "M1_Local_IF_Only"
        records.append(
            {
                "method_id": mm.paper_id,
                "method_display": mm.display,
                "local_vehicle_recall_mean": mm.local_vehicle_recall.mean,
                "local_vehicle_recall_std": mm.local_vehicle_recall.std,
                "strong_campaign_f1_mean": mm.strong_campaign_f1.mean,
                "strong_campaign_f1_std": mm.strong_campaign_f1.std,
                "weak_campaign_f1_mean": mm.weak_campaign_f1.mean,
                "weak_campaign_f1_std": mm.weak_campaign_f1.std,
                "independent_incorrect_merge_mean": mm.independent_merge.mean,
                "independent_incorrect_merge_std": mm.independent_merge.std,
                "false_campaign_rate_s0_mean": mm.false_campaign_s0.mean,
                "false_campaign_rate_s0_std": mm.false_campaign_s0.std,
                "runtime_total_sec_mean": mm.runtime_total_sec.mean,
                "notes": mm.notes,
                "strong_campaign_f1_display": fmt_pm(mm.strong_campaign_f1, na_for_zero_campaign_m1=is_m1),
                "weak_campaign_f1_display": fmt_pm(mm.weak_campaign_f1, na_for_zero_campaign_m1=is_m1),
                "independent_merge_display": fmt_pm(mm.independent_merge),
            }
        )
    return pd.DataFrame(records)


def write_paper_csv(df: pd.DataFrame, path: Path) -> None:
    paper = df[
        [
            "method_display",
            "independent_merge_display",
            "strong_campaign_f1_display",
            "weak_campaign_f1_display",
            "notes",
        ]
    ].copy()
    paper.columns = ["Method", "Independent merge", "Strong F1", "Weak F1", "Notes"]
    paper.to_csv(path, index=False)


def write_metrics_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline and ablation comparison (OCSLab; campaign size 5; seeds "
        + ", ".join(str(s) for s in REQUIRED_SEEDS)
        + r").}",
        r"\label{tab:baseline_ablation}",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        r"Method & Ind.\ merge & Strong F1 & Weak F1 & Notes \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        method = row["method_display"].replace("_", r"\_")
        merge = row["independent_merge_display"].replace("±", r"$\pm$")
        strong = row["strong_campaign_f1_display"].replace("±", r"$\pm$")
        weak = row["weak_campaign_f1_display"].replace("±", r"$\pm$")
        note = str(row["notes"]).replace("_", r"\_")
        if len(note) > 42:
            note = note[:39] + r"\ldots"
        lines.append(f"{method} & {merge} & {strong} & {weak} & {note} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_figure(df: pd.DataFrame, path: Path) -> None:
    labels = df["method_display"].tolist()
    strong = df["strong_campaign_f1_mean"].fillna(0.0).tolist()
    weak = df["weak_campaign_f1_mean"].fillna(0.0).tolist()
    strong_err = df["strong_campaign_f1_std"].fillna(0.0).tolist()
    weak_err = df["weak_campaign_f1_std"].fillna(0.0).tolist()

    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.bar(
        [i - width / 2 for i in x],
        strong,
        width,
        yerr=strong_err,
        capsize=3,
        label="Strong campaign F1",
        color="#2c6eab",
    )
    ax.bar(
        [i + width / 2 for i in x],
        weak,
        width,
        yerr=weak_err,
        capsize=3,
        label="Weak campaign F1",
        color="#e07a2f",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("Campaign F1")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Baseline / ablation: coordinated campaign F1")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, format="pdf")
    plt.close(fig)


def write_readme(df: pd.DataFrame, path: Path) -> None:
    text = f"""# Baseline and ablation comparison

## Status

**Results assembled from existing experiments** on branch `{SOURCE_REF}`.
No experiments were rerun for this bundle.

## Methods

| Paper ID | Framework key | Description |
|----------|---------------|-------------|
| M1_Local_IF_Only | `local_ids` | Vehicle-level Isolation Forest only |
| M2_Descriptor_Clustering_Only | `descriptor_clustering` | Descriptor DBSCAN (no GNN) |
| M3_GraphSAGE_Fleet_Model | `fcgnn` | Proposed GraphSAGE fleet correlator |

## Scenarios

| Scenario | Role | Primary slice |
|----------|------|---------------|
| S2_non_coordinated | Independent multi-vehicle attacks | `campaign_size=5`, `coordination_strength=0.0` |
| S3_strong_campaign | Strong coordinated campaign | Publication `table_06` (campaign size 5) |
| S4_weak_campaign | Weak coordinated campaign | Publication `table_06` (campaign size 5) |
| S0_benign_control | Benign / false campaign (supplementary) | `campaign_size=0` |

Seeds (paper): {", ".join(str(s) for s in REQUIRED_SEEDS)}.

## Source files

"""
    for key, rel in SOURCE_FILES.items():
        text += f"- `{rel}` ({key})\n"

    text += """
## Metric definitions (where calculated)

- **Local vehicle recall**: `Vehicle recall` in `table_06_method_ablation.csv` (S3); local IF detection only.
- **Strong / weak campaign F1**: `Campaign F1` in `table_06_method_ablation.csv` for S3/S4.
  M1 has no fleet layer → campaign F1 reported as N/A in the paper table (0 by definition).
- **Independent incorrect merge**: `incorrect_campaign_merging_mean` in
  `S2_non_coordinated/summary_mean_std.csv` (unrelated attacks must not merge).
- **False campaign (S0)**: `false_campaign_alert_rate_mean` in `S0_benign_control/summary_mean_std.csv`.
- **Runtime**: mean `runtime_total_sec_mean` from S3/S4 summaries (`campaign_size=5`, `coord=1.0`).

Ground-truth campaign labels come from controlled scenario assignment only; attack types are not model inputs.

## Outputs

- `results/baseline_ablation_metrics.csv` — full numeric metrics
- `tables/table_baseline_ablation.csv` — paper-ready CSV
- `tables/table_baseline_ablation.tex` — IEEE-friendly LaTeX
- `figures/figure_baseline_ablation_campaign_f1.pdf` — campaign F1 bar chart

## Reproduce

```bash
python scripts/build_baseline_ablation_comparison.py
```
"""
    path.write_text(text, encoding="utf-8")


def write_source_manifest(path: Path) -> None:
    rows = []
    for key, rel in SOURCE_FILES.items():
        rows.append({"artifact_key": key, "git_ref": SOURCE_REF, "repo_path": rel})
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    subprocess.run(["git", "fetch", "origin", "cursor/campaign-clustering"], cwd=REPO, check=False)

    for sub in ("results", "tables", "figures"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    method_rows = build_metrics()
    df = metrics_to_dataframe(method_rows)

    write_metrics_csv(df, OUT / "results" / "baseline_ablation_metrics.csv")
    write_paper_csv(df, OUT / "tables" / "table_baseline_ablation.csv")
    write_latex_table(df, OUT / "tables" / "table_baseline_ablation.tex")
    write_figure(df, OUT / "figures" / "figure_baseline_ablation_campaign_f1.pdf")
    write_readme(df, OUT / "README.md")
    write_source_manifest(OUT / "results" / "source_manifest.csv")

    print(f"Wrote baseline ablation bundle to {OUT}")
    print(df[["method_display", "independent_merge_display", "strong_campaign_f1_display", "weak_campaign_f1_display"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
