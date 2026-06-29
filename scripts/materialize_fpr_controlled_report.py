#!/usr/bin/env python3
"""Write FPR-controlled validation report and threshold CSV from archived results.

When ``data/processed/window_features.csv`` is unavailable, this script documents
threshold policy, superseded high-FPR sources, and verified pooled metrics from
``results/vehicle_level_threshold_comparison.csv``. Per-vehicle FPR-controlled rows
require running ``scripts/run_vehicle_level_fpr_controlled.py`` with the OCSLab dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.vehicle_level_evaluation import _df_to_ieee_tex


def _load_comparison(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "vehicle_level_threshold_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def _load_thresholds(results_dir: Path) -> dict:
    path = results_dir / "vehicle_level_thresholds.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _pooled_metrics_row(comparison: pd.DataFrame) -> dict:
    row = comparison[comparison["threshold_method"].str.startswith("FPR<=5%")].iloc[0]
    return {
        "vehicle_model": "pooled",
        "roc_auc": float(row["test_roc_auc"]),
        "pr_auc": float(row["test_pr_auc"]),
        "precision": float(row["test_precision"]),
        "recall": float(row["test_recall"]),
        "f1": float(row["test_f1"]),
        "false_positive_rate": float(row["test_fpr"]),
        "threshold": float(row["threshold_value"]),
        "threshold_method": "FPR<=5%",
        "detection_latency_ms": float(row["detection_latency_ms"]),
        "test_windows": "",
        "status": "verified_from_threshold_comparison",
    }


def _write_threshold_csv(thresholds: dict, out: Path) -> None:
    rows = [
        {
            "vehicle_model": vehicle,
            "threshold_method": "FPR<=5%",
            "threshold_value": round(float(value), 6),
            "validation_fpr_cap": 0.05,
        }
        for vehicle, value in thresholds["per_vehicle_threshold_fpr_le_5pct"].items()
    ]
    rows.append(
        {
            "vehicle_model": "pooled",
            "threshold_method": thresholds["selected_threshold_method"],
            "threshold_value": round(float(thresholds["selected_pooled_threshold"]), 6),
            "validation_fpr_cap": 0.05,
        }
    )
    pd.DataFrame(rows).to_csv(out, index=False)


def _write_validation_report(
    *,
    comparison: pd.DataFrame,
    thresholds: dict,
    pooled_row: dict,
    p4_path: Path,
    out: Path,
) -> None:
    f1_row = comparison[comparison["threshold_method"] == "F1-optimal"].iloc[0]
    fpr_row = comparison[comparison["threshold_method"].str.startswith("FPR<=5%")].iloc[0]

    lines = [
        "# Vehicle-Level Isolation Forest Validation Report",
        "",
        "## Intended protocol (FPR-controlled)",
        "",
        "1. Train Isolation Forest on **benign-only** training windows (70/15/15 stratified split per vehicle).",
        "2. Score validation and test windows with normalised anomaly scores.",
        "3. On **validation only**, select threshold with FPR ≤ 5%; among feasible thresholds, pick **highest recall**.",
        "4. Apply per-vehicle thresholds to the **held-out test** partition.",
        "5. Report PR-AUC, precision, recall, F1, and FPR per vehicle and pooled.",
        "",
        "## Superseded high-FPR Table I sources",
        "",
        "| Candidate source | Threshold policy | Typical pooled FPR | Notes |",
        "|------------------|------------------|--------------------|-------|",
        f"| `results/vehicle_level_threshold_comparison.csv` (F1-optimal) | F1-optimal on validation | **{f1_row['test_fpr']:.1%}** | High F1, impractical FPR |",
        f"| `experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv` | Fixed strong thresholds (Chevrolet 0.85, Hyundai/Kia 0.7) | **~31%** | Fleet pipeline descriptor gate, not FPR-controlled |",
        "| Draft Table I (manuscript: Chevrolet 0.211, Hyundai 0.436, Kia 0.353, pooled 0.373) | Fixed / F1-oriented threshold (not FPR≤5%) | **~37%** | Inconsistent with deployment policy and Table X |",
        "",
        "### Threshold policy diagnosis",
        "",
        f"- **F1-optimal (validation):** test F1={f1_row['test_f1']:.3f}, test FPR={f1_row['test_fpr']:.3f}",
        f"- **FPR≤5% (corrected):** test F1={fpr_row['test_f1']:.3f}, test FPR={fpr_row['test_fpr']:.3f}",
        "",
        "The draft Table I high-FPR values align with **fixed strong-threshold fleet evaluation** "
        "(`table_P4_vehicle_level_results.csv` and `experiments/04_train_vehicle_ids.py` defaults), "
        "not the FPR-controlled protocol used for Table X.",
        "",
    ]

    if p4_path.exists():
        p4 = pd.read_csv(p4_path)
        lines.extend(
            [
                f"### Archived per-vehicle high-FPR table (`{p4_path}`)",
                "",
                "```",
                p4.to_string(index=False),
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Corrected FPR-controlled metrics (test)",
            "",
            "### Pooled (verified)",
            "",
            f"- PR-AUC: **{pooled_row['pr_auc']:.4f}**",
            f"- Precision: **{pooled_row['precision']:.4f}**",
            f"- Recall: **{pooled_row['recall']:.4f}**",
            f"- F1: **{pooled_row['f1']:.4f}**",
            f"- FPR: **{pooled_row['false_positive_rate']:.4f}**",
            f"- Detection latency (ms): **{pooled_row['detection_latency_ms']:.2f}**",
            "",
            "### Per-vehicle",
            "",
            "Per-vehicle FPR≤5% test metrics were **not previously exported** in curated paper artifacts. "
            "They are computed by `run_vehicle_level_evaluation()` when `data/processed/window_features.csv` "
            "is available. Re-run:",
            "",
            "```bash",
            "export OCSLAB_DATASET_DIR=/path/to/In-Vehicle\\ Network\\ Intrusion\\ Detection\\ Challenge",
            "python experiments/01_load_dataset.py --config configs/default.yaml",
            "python experiments/02_generate_windows.py --config configs/default.yaml",
            "python experiments/03_extract_features.py",
            "python scripts/run_vehicle_level_fpr_controlled.py --config configs/default.yaml",
            "```",
            "",
            "## Selected thresholds (validation FPR≤5%)",
            "",
        ]
    )
    for vehicle, value in thresholds["per_vehicle_threshold_fpr_le_5pct"].items():
        lines.append(f"- **{vehicle}:** {float(value):.6f}")
    lines.append(
        f"- **Pooled reference:** {float(thresholds['selected_pooled_threshold']):.6f} "
        f"({thresholds['selected_threshold_method']})"
    )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "**Replace Table I** with `tables/table_vehicle_level_ids_fpr_controlled.tex` "
            "and `results/vehicle_level_metrics_fpr_controlled.csv`.",
            "Use the pooled FPR≤5% row for the deployment-oriented summary (matches Table X).",
            "Do not report F1-optimal or fixed strong-threshold metrics as primary vehicle-level IDS results.",
            "",
        ]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize FPR-controlled report from archived results")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tables-dir", default="tables")
    args = parser.parse_args()

    results_dir = _ROOT / args.results_dir
    tables_dir = _ROOT / args.tables_dir
    p4_path = _ROOT / "experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv"

    comparison = _load_comparison(results_dir)
    thresholds = _load_thresholds(results_dir)
    pooled_row = _pooled_metrics_row(comparison)

    metrics_path = results_dir / "vehicle_level_metrics_fpr_controlled.csv"
    pd.DataFrame([pooled_row]).to_csv(metrics_path, index=False)

    thresholds_path = results_dir / "vehicle_level_thresholds_fpr_controlled.csv"
    _write_threshold_csv(thresholds, thresholds_path)

    report_path = results_dir / "vehicle_level_validation_report.md"
    _write_validation_report(
        comparison=comparison,
        thresholds=thresholds,
        pooled_row=pooled_row,
        p4_path=p4_path,
        out=report_path,
    )

    tex_df = pd.DataFrame(
        [
            {
                "Vehicle": "pooled",
                "PR-AUC": pooled_row["pr_auc"],
                "Precision": pooled_row["precision"],
                "Recall": pooled_row["recall"],
                "F1": pooled_row["f1"],
                "FPR": pooled_row["false_positive_rate"],
            }
        ]
    )
    tex_path = tables_dir / "table_vehicle_level_ids_fpr_controlled.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(
        _df_to_ieee_tex(
            tex_df,
            "Vehicle-level Isolation Forest (benign-only training) with validation FPR$\\leq$5\\% threshold selection.",
            "tab:vehicle-ids-fpr-controlled",
        ),
        encoding="utf-8",
    )

    print("\n=== Corrected LaTeX table (Table I) — pooled row verified ===\n")
    print(tex_path.read_text(encoding="utf-8"))
    print(f"\nWrote: {metrics_path}")
    print(f"Wrote: {thresholds_path}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {tex_path}")
    print(
        "\nNote: per-vehicle rows require OCSLab features. "
        "Run scripts/run_vehicle_level_fpr_controlled.py after preprocessing."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
