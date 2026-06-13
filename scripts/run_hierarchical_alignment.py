#!/usr/bin/env python3
"""Align evaluation with hierarchical local IDS + fleet correlation design."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.hierarchical_alignment.collect import collect_all_runs
from src.experiments.hierarchical_alignment.metrics import build_capability_comparison
from src.experiments.hierarchical_alignment.outputs import export_figures, export_tables
from src.experiments.hierarchical_alignment.statistics import run_hierarchical_statistics
from src.utils.paths import resolve_project_root

OUT_ROOT = Path("new_experiments/final_validated_runs/hierarchical_alignment")


def _ensure_dirs(root: Path) -> None:
    for sub in ("audit", "results", "tables", "figures", "logs", "validation", "supplementary"):
        (root / sub).mkdir(parents=True, exist_ok=True)


def _write_audit(root: Path) -> None:
    audit = root / "audit"
    (audit / "output_schema_definition.md").write_text(
        """# Hierarchical output schema

Each event record contains independent vehicle-local and fleet-correlation fields.

| Field | Source | Meaning | Evaluation use |
|-------|--------|---------|----------------|
| `event_id` | Scenario manifest | Unique event identifier | Join key |
| `scenario_vehicle_id` | Scenario manifest | Vehicle instance in scenario | Offline metadata |
| `vehicle_token` | Scenario manifest | Anonymised vehicle token | Offline metadata |
| `local_anomaly_score` | Isolation Forest | Continuous anomaly score | ROC-AUC, PR-AUC |
| `local_evidence_level` | Isolation Forest bands | benign / weak / strong | Interpretation |
| `local_event_alert` | Isolation Forest (`local_alert`) | Strong local alert only | Event precision/recall/F1 |
| `fleet_cluster_id` | DBSCAN on similarity or embedding | Behavioural cluster label | Campaign structure |
| `fleet_campaign_member` | Fleet qualification gate | Member of coordinated campaign cluster | Campaign metrics |
| `fleet_campaign_confidence` | GNN score + cohesion | Campaign-support confidence (not event label) | Supplementary |
| `fleet_decision` | Fleet correlation layer | no fleet evidence / isolated / coordinated | Campaign evaluation |
| `ground_truth_malicious` | Scenario GT | Malicious window label | Event evaluation |
| `ground_truth_campaign_member` | Scenario GT | True campaign membership | Campaign evaluation |

## Invariants

1. `local_event_alert` is never derived from `fleet_campaign_member`, `cluster_id`, or `fleet_decision`.
2. C1 (local-only IDS) leaves fleet fields empty / N/A.
3. C2 and C3 preserve identical local fields; only fleet outputs differ.
4. Vehicle model, attack type, and source trace remain offline evaluation metadata only.
""",
        encoding="utf-8",
    )
    (audit / "result_writer_audit.md").write_text(
        """# Result writer audit

## Finding: local event decisions were conflated with fleet membership

`src/experiments/experiment_pipeline.py` function `_decisions_to_predictions` sets:

```python
predicted_malicious = (local_alert == 1) | (weak_signal == 1) | (final_decision == coordinated_attack)
```

This OR-logic promoted every coordinated cluster member to a malicious **event** prediction, even when the Isolation Forest did not fire a strong local alert. That made graph-based methods appear weaker at event detection despite their intended role being **campaign correlation**.

## Hierarchical alignment correction

- **Event metrics** use `local_event_alert` (= `local_alert` only) from Isolation Forest.
- **Campaign metrics** use `fleet_campaign_member` and `fleet_decision` from C2/C3 only.
- Raw run artifacts are reused; aligned outputs are written under `hierarchical_alignment/` without overwriting prior results.

## Configurations

| Code | Method | Fleet layer |
|------|--------|-------------|
| C1 | `local_ids` | None |
| C2 | `descriptor_clustering` | Similarity-only DBSCAN |
| C3 | `fcgnn` | GraphSAGE-based fleet correlation |

No model architecture, graph construction, clustering algorithm, scenario composition, or seeds were changed.
""",
        encoding="utf-8",
    )


def _write_summary(
    root: Path,
    *,
    local_df: pd.DataFrame,
    fleet_df: pd.DataFrame,
    weak_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    n_events: int,
) -> None:
    c2 = fleet_df[fleet_df["framework_config"] == "C2"]
    c3 = fleet_df[fleet_df["framework_config"] == "C3"]
    s3_c2 = c2[c2["scenario_id"] == "S3"]["campaign_f1"].mean()
    s3_c3 = c3[c3["scenario_id"] == "S3"]["campaign_f1"].mean()
    s4_c2 = c2[c2["scenario_id"] == "S4"]["campaign_f1"].mean()
    s4_c3 = c3[c3["scenario_id"] == "S4"]["campaign_f1"].mean()
    sig = stats_df[stats_df.get("significant", False)] if not stats_df.empty else pd.DataFrame()

    text = f"""# Final hierarchical alignment summary

## 1. Which component performs event detection?

The **vehicle-level Isolation Forest** (benign-only training). Event-level precision, recall, F1, FPR, ROC-AUC, and PR-AUC are computed from `local_event_alert` / `local_anomaly_score` only.

## 2. Which component performs campaign correlation?

**C2 — similarity-only fleet correlation** (descriptor DBSCAN) and **C3 — GraphSAGE-based fleet correlation** (same graph, message passing + embedding clustering). C1 does not perform campaign correlation.

## 3. Were local decisions previously overwritten?

**Yes.** Original `predicted_malicious` OR-ed fleet coordinated membership and weak signals into the event label, inflating FPR and conflating tasks.

## 4. What was corrected?

Separated hierarchical fields, recomputed event metrics from Isolation Forest outputs, restricted campaign metrics to C2 vs C3, and regenerated tables/figures under `hierarchical_alignment/` without retraining models.

## 5. Are event metrics now reported only for Isolation Forest?

**Yes.** All configurations share the same local Isolation Forest evidence; event metrics are identical across C1/C2/C3 on the same scenario records.

## 6. Does GraphSAGE improve campaign F1 over similarity-only correlation?

S3 mean campaign F1 — C2: {s3_c2:.4f}, C3: {s3_c3:.4f}. S4 mean campaign F1 — C2: {s4_c2:.4f}, C3: {s4_c3:.4f}. See `results/statistical_tests.csv` for paired C3 vs C2 tests ({len(sig)} significant after Holm correction).

## 7. How are weak signals interpreted?

Individually weak or inconclusive local signals (`local_evidence_level=weak`, `local_event_alert=false`) can still support fleet campaign membership when correlated across vehicles. See `results/weak_campaign_support.csv`.

## 8. Which results belong in the main paper?

- Table H2: local IDS performance
- Tables H3–H5: fleet campaign correlation (C2 vs C3)
- Table H6: capability comparison (local-only vs complete framework)
- Table H7 / statistical tests: C3 vs C2 campaign metrics
- Figures H1–H5

## 9. Which old figures or tables must be excluded?

- Any figure comparing Local IDS event F1 vs GraphSAGE event F1
- Tables treating `predicted_malicious` (OR-logic) as event classification for graph methods
- Claims that GraphSAGE replaces or outperforms the local IDS at event detection
- C1 vs C3 campaign F1 comparisons

## 10. What limitations remain?

- Weak-band local evidence remains inconclusive at event level by design
- Campaign metrics depend on DBSCAN qualification gates and fixed 200-node scenarios
- Latency includes end-to-end pipeline runtime from stored run logs
- Benign-on-attacked descriptors may score in weak bands without being fleet campaign members

---

Aligned events: {n_events:,} rows across reused validated runs.
"""
    (root / "FINAL_HIERARCHICAL_ALIGNMENT_SUMMARY.md").write_text(text, encoding="utf-8")


def main() -> int:
    project_root = resolve_project_root()
    out_root = project_root / OUT_ROOT
    _ensure_dirs(out_root)
    log_lines: list[str] = []

    try:
        _write_audit(out_root)
        log_lines.append("Wrote audit documents")

        aligned, local_df, fleet_df, weak_df, err_df = collect_all_runs()
        if aligned.empty:
            raise RuntimeError("No runs collected — check Phase 2/3 result paths")

        results = out_root / "results"
        aligned.to_csv(results / "hierarchical_event_predictions.csv", index=False)
        local_df.to_csv(results / "local_event_metrics.csv", index=False)
        fleet_df.to_csv(results / "fleet_campaign_metrics.csv", index=False)
        weak_df.to_csv(results / "weak_campaign_support.csv", index=False)
        err_df.to_csv(results / "campaign_membership_errors.csv", index=False)
        build_capability_comparison().to_csv(results / "capability_comparison.csv", index=False)

        stats_df = run_hierarchical_statistics(fleet_df)
        stats_df.to_csv(results / "statistical_tests.csv", index=False)
        log_lines.append(f"Collected {len(aligned)} aligned events, {len(fleet_df)} fleet run rows")

        tables = export_tables(
            local_df,
            fleet_df,
            weak_df,
            stats_df,
            pd.read_csv(results / "capability_comparison.csv"),
            out_root / "tables",
            events_df=aligned,
        )
        figures = export_figures(fleet_df, weak_df, out_root / "figures")
        log_lines.append(f"Tables: {tables}")
        log_lines.append(f"Figures: {figures}")

        _write_summary(
            out_root,
            local_df=local_df,
            fleet_df=fleet_df,
            weak_df=weak_df,
            stats_df=stats_df,
            n_events=len(aligned),
        )
        log_lines.append("Wrote FINAL_HIERARCHICAL_ALIGNMENT_SUMMARY.md")
    except Exception as exc:
        log_lines.append(f"FAIL: {exc}\n{traceback.format_exc()}")
        (out_root / "logs" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log").write_text(
            "\n".join(log_lines), encoding="utf-8"
        )
        raise

    (out_root / "logs" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log").write_text(
        "\n".join(log_lines), encoding="utf-8"
    )
    print("\n".join(log_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
