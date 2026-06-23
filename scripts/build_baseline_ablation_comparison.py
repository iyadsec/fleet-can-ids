#!/usr/bin/env python3
"""Assemble baseline/ablation comparison from existing OCSLab runs (no reruns).

M3 (proposed GraphSAGE fleet) uses the **balanced publication run** — the same
pipeline family as draft Tables III–IV (Section VII), but with the corrected
balanced split (authoritative).

M1/M2 use the **framework ablation** bundle (table_06 + S2 summaries), which
applies a different evaluation-correction layer and must not be mixed with M3
fleet F1 without noting the provenance split.
"""

from __future__ import annotations

import io
import re
import subprocess
import statistics as stats
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "new_experiments" / "baseline_ablation_comparison"
SOURCE_REF = "origin/cursor/campaign-clustering"

PRIMARY_CAMPAIGN_SIZE = 5
REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]

# Authoritative balanced publication (draft Table III unrelated merge; Table IV family)
BALANCED = {
    "P6_safety": "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P6_benign_isolated_unrelated_results.csv",
    "P7_strong": "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P7_strong_campaign_results.csv",
    "P8_weak": "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P8_weak_campaign_results.csv",
    "campaign_metrics": "new_experiments/final_end_to_end_publication_run_balanced/results/scenario_evaluation/campaign_metrics.csv",
    "split_audit": "new_experiments/final_end_to_end_publication_run_balanced/audit/original_vs_balanced_split.md",
}

# Framework ablation (M1/M2 only — corrected promotion rules, phase-2/3 scenarios)
ABLATION = {
    "table_06": "new_experiments/publication_ready/tables/table_06_method_ablation.csv",
    "S2_summary": "new_experiments/results/S2_non_coordinated/summary_mean_std.csv",
}

SOURCE_FILES = {**BALANCED, **ABLATION}


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


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return stats.mean(values), stats.pstdev(values)


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
    result_set: str
    local_vehicle_recall: MetricCell = field(default_factory=MetricCell)
    strong_campaign_f1: MetricCell = field(default_factory=MetricCell)
    weak_campaign_f1: MetricCell = field(default_factory=MetricCell)
    independent_merge: MetricCell = field(default_factory=MetricCell)
    runtime_total_sec: MetricCell = field(default_factory=MetricCell)


def load_m3_balanced_metrics() -> dict[str, MetricCell]:
    """M3 metrics from balanced publication run (authoritative Section VII pipeline)."""
    p6 = pd.read_csv(io.StringIO(git_show_text(SOURCE_REF, BALANCED["P6_safety"])))
    unrelated = p6[p6["test_condition"] == "Unrelated Multi-Vehicle Incidents"].iloc[0]
    merge = MetricCell(
        float(unrelated["incorrect_merging"]),
        0.0,
        BALANCED["P6_safety"],
        "Balanced run; unrelated multi-vehicle incidents (draft Table III family)",
    )

    cm = pd.read_csv(io.StringIO(git_show_text(SOURCE_REF, BALANCED["campaign_metrics"])))
    cs = str(PRIMARY_CAMPAIGN_SIZE)

    strong_vals = [
        float(r["campaign_f1"])
        for r in cm.to_dict("records")
        if r.get("experiment_group") == "Strong Coordinated Campaign"
        and str(r.get("campaign_size")) == cs
    ]
    weak_vals = [
        float(r["campaign_f1"])
        for r in cm.to_dict("records")
        if r.get("experiment_group") == "Weak Coordinated Campaign"
        and str(r.get("campaign_size")) == cs
    ]
    sm, ss = mean_std(strong_vals)
    wm, ws = mean_std(weak_vals)

    p7 = pd.read_csv(io.StringIO(git_show_text(SOURCE_REF, BALANCED["P7_strong"])))
    p7_cs5 = p7[p7["campaign_size"].astype(int) == PRIMARY_CAMPAIGN_SIZE].iloc[0]
    runtime = MetricCell(
        float(p7_cs5["end_to_end_latency"]),
        None,
        BALANCED["P7_strong"],
        "Balanced run end_to_end_latency (strong, campaign_size=5)",
    )

    return {
        "independent_merge": merge,
        "strong_campaign_f1": MetricCell(
            sm, ss, BALANCED["campaign_metrics"], f"Per-seed strong F1, campaign_size={cs}"
        ),
        "weak_campaign_f1": MetricCell(
            wm, ws, BALANCED["campaign_metrics"], f"Per-seed weak F1, campaign_size={cs}"
        ),
        "runtime_total_sec": runtime,
        "local_vehicle_recall": MetricCell(
            1.0, 0.0, BALANCED["P6_safety"], "Local detection on coordinated scenarios = 1.0"
        ),
    }


def load_m1_m2_ablation() -> dict[str, dict[str, MetricCell]]:
    """M1/M2 from framework ablation (table_06 + S2); not the balanced publication pipeline."""
    table06 = pd.read_csv(io.StringIO(git_show_text(SOURCE_REF, ABLATION["table_06"])))
    s2 = pd.read_csv(io.StringIO(git_show_text(SOURCE_REF, ABLATION["S2_summary"])))

    specs = {
        "M1_Local_IF_Only": ("local_ids", "M1 Local IDS"),
        "M2_Descriptor_Clustering_Only": ("descriptor_clustering", "M2 Descriptor clustering"),
    }
    out: dict[str, dict[str, MetricCell]] = {}
    for paper_id, (method_key, t06_label) in specs.items():
        cells: dict[str, MetricCell] = {}
        s3_row = table06[(table06["Scenario"] == "S3") & (table06["Method"] == t06_label)]
        if not s3_row.empty:
            m, s = parse_pm(s3_row.iloc[0]["Vehicle recall"])
            cells["local_vehicle_recall"] = MetricCell(m, s, ABLATION["table_06"], "S3 vehicle recall")

        for scen, key in (("S3", "strong_campaign_f1"), ("S4", "weak_campaign_f1")):
            row = table06[(table06["Scenario"] == scen) & (table06["Method"] == t06_label)]
            if not row.empty:
                m, s = parse_pm(row.iloc[0]["Campaign F1"])
                cells[key] = MetricCell(m, s, ABLATION["table_06"], f"{scen} campaign F1 (framework ablation)")

        s2_row = s2[
            (s2["method"] == method_key)
            & (s2["campaign_size"].astype(float) == float(PRIMARY_CAMPAIGN_SIZE))
            & (s2["coordination_strength"].astype(float) == 0.0)
        ]
        if not s2_row.empty:
            r = s2_row.iloc[0]
            cells["independent_merge"] = MetricCell(
                float(r["incorrect_campaign_merging_mean"]),
                float(r["incorrect_campaign_merging_std"]),
                ABLATION["S2_summary"],
                "Framework ablation S2 (not balanced unrelated-incidents table)",
            )
        out[paper_id] = cells
    return out


def build_metrics() -> list[MethodMetrics]:
    m3 = load_m3_balanced_metrics()
    ablation = load_m1_m2_ablation()

    methods = [
        MethodMetrics(
            "M1_Local_IF_Only",
            "M1 Local IF only",
            "Vehicle-level IF only; campaign F1 N/A (no fleet layer)",
            "framework_ablation",
        ),
        MethodMetrics(
            "M2_Descriptor_Clustering_Only",
            "M2 Descriptor clustering",
            "DBSCAN on descriptors; no GNN (framework ablation eval)",
            "framework_ablation",
        ),
        MethodMetrics(
            "M3_GraphSAGE_Fleet_Model",
            "M3 GraphSAGE fleet",
            "Balanced publication pipeline (authoritative primary dataset)",
            "balanced_publication",
        ),
    ]

    for mm in methods:
        src = m3 if mm.paper_id == "M3_GraphSAGE_Fleet_Model" else ablation.get(mm.paper_id, {})
        for attr in (
            "local_vehicle_recall",
            "strong_campaign_f1",
            "weak_campaign_f1",
            "independent_merge",
            "runtime_total_sec",
        ):
            if attr in src:
                setattr(mm, attr, src[attr])

    # M1 has no fleet merge path — report N/A semantics via 0 with note in display
    if methods[0].independent_merge.mean is None:
        methods[0].independent_merge = MetricCell(
            0.0, 0.0, "", "No fleet clustering (not applicable)"
        )

    return methods


def fmt_pm(cell: MetricCell, *, na_for_zero_campaign_m1: bool = False) -> str:
    if cell.mean is None:
        return "N/A"
    if na_for_zero_campaign_m1 and cell.mean == 0.0 and (cell.std or 0.0) == 0.0:
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
                "result_set": mm.result_set,
                "local_vehicle_recall_mean": mm.local_vehicle_recall.mean,
                "local_vehicle_recall_std": mm.local_vehicle_recall.std,
                "strong_campaign_f1_mean": mm.strong_campaign_f1.mean,
                "strong_campaign_f1_std": mm.strong_campaign_f1.std,
                "weak_campaign_f1_mean": mm.weak_campaign_f1.mean,
                "weak_campaign_f1_std": mm.weak_campaign_f1.std,
                "independent_incorrect_merge_mean": mm.independent_merge.mean,
                "independent_incorrect_merge_std": mm.independent_merge.std,
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


def write_latex_table(df: pd.DataFrame, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Baseline/ablation (OCSLab). M3 from balanced publication run; M1/M2 from framework ablation.}",
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
        if len(note) > 38:
            note = note[:35] + r"\ldots"
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
        [i - width / 2 for i in x], strong, width, yerr=strong_err, capsize=3,
        label="Strong campaign F1", color="#2c6eab",
    )
    ax.bar(
        [i + width / 2 for i in x], weak, width, yerr=weak_err, capsize=3,
        label="Weak campaign F1", color="#e07a2f",
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


def write_provenance(path: Path) -> None:
    text = """# Result provenance — resolving draft vs baseline-ablation conflicts

## Three OCSLab result sets (do not mix without labelling)

| Result set | Branch path | Strong F1 (cs=5) | Unrelated merge | Status |
|------------|-------------|------------------|-----------------|--------|
| **A. Draft Tables III–IV (original split)** | pre-balanced end-to-end run | **0.867** | **0.400** | Superseded — draft not yet updated |
| **B. Balanced publication (authoritative)** | `final_end_to_end_publication_run_balanced/` | **0.733** | **0.400** | **Use for Section VII primary results** |
| **C. Framework ablation (M1/M2/M3 compare)** | `publication_ready/table_06` + `results/S2_*` | **0.406** (fcgnn) | **0.000** (fcgnn) | Descriptive ablation only; different eval correction |

### Why draft Table IV (0.867) ≠ baseline-ablation (0.406 or 0.733)

1. **Draft 0.867** comes from the **original train/validation split** before the balanced Chevrolet-inclusive split.
   See `audit/original_vs_balanced_split.md`: original strong cs=5 F1 = 0.867 → balanced = 0.733.

2. **Baseline-ablation 0.406** (previous bundle) pulled `table_06_method_ablation.csv` from the **framework ablation**
   pipeline with evaluation-correction / promotion rules applied on phase-2/3 scenario artifacts — not the balanced
   end-to-end publication run.

3. **Corrected baseline-ablation M3** now uses the **balanced publication run** (0.733 strong F1, 0.400 unrelated merge),
   aligning with draft Table III (merge) and the authoritative Section VII numbers (F1).

### Why draft Table III (0.400) ≠ old baseline-ablation (0.000)

- Draft **0.400** = `table_P6` unrelated multi-vehicle incidents from the **balanced publication** pipeline.
- Old baseline **0.000** = `S2_non_coordinated/summary_mean_std.csv` under **framework ablation** (different scenario
  construction, metric definition, and fcgnn slice at campaign_size=5, coord=0).

### Final publication version

**Use B (balanced publication)** for all primary OCSLab fleet results in Section VII.
Update draft Table IV from 0.867 → 0.733 (and cs=10 from 0.933 → 1.000) when revising the manuscript.

**Use C (framework ablation)** only for the explicit M1 vs M2 vs M3 ablation subsection — with M3 fleet metrics
sourced from B when reporting headline coordinated-campaign performance.

Cross-dataset (can-train-and-test) results remain in `OVERLEAF_CROSS_DATASET_ARTIFACTS/` and `experimental-2026-06-19/03_cross_dataset_ctt/`.
"""
    path.write_text(text, encoding="utf-8")


def write_readme(path: Path) -> None:
    path.write_text(
        f"""# Baseline and ablation comparison

Assembled from `{SOURCE_REF}` — **no experiments rerun**.

## Important

Read `RESULT_PROVENANCE.md` before citing numbers. M3 uses the **balanced publication** run;
M1/M2 use the **framework ablation** bundle (table_06).

## Reproduce

```bash
python scripts/build_baseline_ablation_comparison.py
python scripts/consolidate_experimental_results.py
```
""",
        encoding="utf-8",
    )


def main() -> int:
    subprocess.run(["git", "fetch", "origin", "cursor/campaign-clustering"], cwd=REPO, check=False)
    for sub in ("results", "tables", "figures"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    df = metrics_to_dataframe(build_metrics())
    df.to_csv(OUT / "results" / "baseline_ablation_metrics.csv", index=False)
    write_paper_csv(df, OUT / "tables" / "table_baseline_ablation.csv")
    write_latex_table(df, OUT / "tables" / "table_baseline_ablation.tex")
    write_figure(df, OUT / "figures" / "figure_baseline_ablation_campaign_f1.pdf")
    write_readme(OUT / "README.md")
    write_provenance(OUT / "RESULT_PROVENANCE.md")
    pd.DataFrame(
        [{"artifact_key": k, "git_ref": SOURCE_REF, "repo_path": v} for k, v in SOURCE_FILES.items()]
    ).to_csv(OUT / "results" / "source_manifest.csv", index=False)

    print(df[["method_display", "result_set", "independent_merge_display", "strong_campaign_f1_display", "weak_campaign_f1_display"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
