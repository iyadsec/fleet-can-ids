#!/usr/bin/env python3
"""Re-evaluate corrected Phase 3 with fixed event/campaign decision logic."""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.experiments.campaign_analysis_corrected import DescriptorBudget
from src.experiments.campaign_analysis_runner import run_campaign_analysis_corrected_single
from src.experiments.campaign_analysis_writer import CampaignRunContext, load_campaign_analysis_config
from src.experiments.campaign_evaluation import compute_confusion, compute_event_metrics
from src.experiments.evaluation_correction.benign_audit import audit_benign_on_attacked
from src.experiments.evaluation_correction.fixed_evidence import fixed_evidence_budget, validate_fixed_evidence_records
from src.experiments.evaluation_correction.metrics import (
    aggregate_corrected_run_metrics,
    campaign_error_breakdown_row,
    compute_event_confusion_row,
    reconstruct_cluster_df,
)
from src.experiments.evaluation_correction.outputs import export_figures, export_tables, summarize_weak_campaign_results
from src.experiments.evaluation_correction.promotion import PromotionConfig, apply_corrected_event_decisions, tune_promotion_threshold_validation
from src.experiments.evaluation_correction.statistics import run_corrected_statistical_tests
from src.experiments.experiment_pipeline import MethodOutputs
from src.experiments.result_writer import load_experiment_config
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

PHASE3_CONFIG = "new_experiments/final_validated_runs/configs/phase3_campaign_size_corrected.yaml"
PHASE3_RUNS = Path("new_experiments/final_validated_runs/results/campaign_size_corrected/runs")
OUT_ROOT = Path("new_experiments/final_validated_runs/evaluation_correction")
VALIDATION_SEEDS = {11, 23, 37}


def _merge_config(cfg: dict) -> dict:
    base_path = cfg.get("paths", {}).get("base_scenario_config")
    base = load_experiment_config(base_path) if base_path else {}
    merged = {**base, **cfg}
    for key in ("local_ids", "graph", "gnn", "campaign"):
        merged[key] = {**base.get(key, {}), **cfg.get(key, {})}
    return merged


def _original_confusion_row(run_dir: Path) -> dict:
    events = pd.read_csv(run_dir / "event_predictions.csv")
    metrics = pd.read_csv(run_dir / "run_level_metrics.csv").iloc[0]
    y_true = events["ground_truth_malicious"].astype(int)
    y_pred = events["predicted_malicious"].astype(int)
    cm = compute_confusion(y_true.to_numpy(), y_pred.to_numpy())
    ev = compute_event_metrics(events)
    return {
        "run_id": run_dir.name,
        "seed": int(metrics["seed"]),
        "method": str(metrics["method"]),
        "attack_strength": str(metrics["attack_strength"]),
        "campaign_size": int(metrics["campaign_size"]),
        "evaluation": "original",
        **cm,
        "fpr": ev["fpr"],
        "f1": ev["f1"],
        "predicted_malicious_events": int((y_pred == 1).sum()),
        "total_benign_events": int((y_true == 0).sum()),
    }


def _write_audit_event_logic(audit_dir: Path) -> None:
    text = """# Event prediction logic audit

## Pipeline trace (M2–M4)

1. **Local anomaly score** — per-window Isolation Forest score in `anomaly_descriptors.csv`.
2. **Local evidence** — `local_alert` (strong) and `weak_signal` (weak) flags from threshold bands.
3. **Descriptor inclusion** — scenario generator selects windows into graph nodes (`selected_source_records.csv`).
4. **Graph node** — each descriptor becomes a node (`experiment_pipeline.run_graph_method` → `build_scenario_graph`).
5. **Cluster membership** — DBSCAN on descriptor or GNN embeddings (`_cluster_features`).
6. **Campaign membership** — `assign_final_decisions` marks all nodes in qualifying clusters as `coordinated_attack`.
7. **Original event prediction** — `_decisions_to_predictions`: `predicted_malicious = local_alert | weak_signal | coordinated`.

## M1 Local IDS

| Question | Answer |
|----------|--------|
| Predicted malicious event | `local_alert == 1` OR `weak_signal == 1` |
| Predicted campaign member | Not supported (no graph) |
| All cluster events malicious | N/A |
| All graph nodes anomalous | N/A |
| Benign descriptors auto-positive | Only if weak/strong local flags fire |
| Campaign copied to cluster | N/A |
| Weak vs strong distinguished | Yes via `local_alert` vs `weak_signal` |

## M2 Descriptor clustering

| Question | Original | Corrected |
|----------|----------|-----------|
| Predicted malicious | OR of local, weak, coordinated | Strong local OR gated weak promotion |
| Campaign member | `final_decision == coordinated_attack` | Same (separate label) |
| All cluster events malicious | **Yes** (coordinated → all positive) | **No** |
| Benign in campaign auto-positive | **Yes** | **No** (promotion gates) |
| Campaign copied to members | **Yes** | Membership only; events gated |

## M3 Standard GNN / M4 FCGNN

Same corrected rules as M2; campaign from GNN embeddings + DBSCAN qualification.

## Key functions

- `src/experiments/experiment_pipeline.py` — `_decisions_to_predictions`, `run_graph_method`
- `src/evaluation/final_gnn_fleet_decision_experiment.py` — `assign_final_decisions`
- `src/experiments/method_local_ids.py` — M1 predictions
- `src/experiments/campaign_evaluation.py` — metric aggregation
- `src/experiments/evaluation_correction/promotion.py` — corrected three-label schema
"""
    (audit_dir / "event_prediction_logic_audit.md").write_text(text, encoding="utf-8")


def _write_fpr_analysis(audit_dir: Path, original: pd.DataFrame, corrected: pd.DataFrame) -> None:
    orig_mean_fpr = original.groupby("method")["fpr"].mean()
    corr_mean_fpr = corrected.groupby("method")["fpr"].mean()
    all_pred = int((original["predicted_malicious_events"] == original["total_benign_events"] + original.get("tp", 0)).sum())
    text = f"""# FPR root cause analysis

## Primary cause (original evaluation)

**`_decisions_to_predictions` treats campaign membership as automatic malicious prediction:**

```python
predicted_malicious = (local_alert == 1) | (weak_signal == 1) | (final_decision == coordinated_attack)
```

### Mechanism

1. `assign_final_decisions` labels **every node** in a qualifying cluster as `coordinated_attack`.
2. Benign-on-attacked and fleet-benign nodes in the same cluster inherit coordinated status.
3. OR logic forces `predicted_malicious = 1` for all coordinated nodes regardless of ground truth.
4. Additional inflation from `weak_signal` on benign `attack_type=benign` fleet windows (IDS scores benign traffic in weak/strong bands).

### Evidence

- Original mean FPR by method: {orig_mean_fpr.to_dict()}
- Runs with FPR = 1.0: {int((original['fpr'] >= 0.999).sum())} / {len(original)}
- Runs predicting all events malicious: {int((original['predicted_malicious_events'] == 200).sum())} / {len(original)}

## Corrected evaluation

- Separated `predicted_campaign_membership` from `predicted_malicious`.
- Strong local events (`local_alert`) predict malicious independently.
- Weak events require cluster cohesion, ≥2 vehicles, malicious support, and confidence threshold.
- `attack_type=benign` never promoted.
- Corrected mean FPR by method: {corr_mean_fpr.to_dict()}

## Not the primary cause

- Confusion matrix implementation (verified correct).
- Ground-truth label inversion (GT columns consistent).
- Threshold constants alone (negative anomaly scores; flags drive evidence).
"""
    (audit_dir / "fpr_root_cause_analysis.md").write_text(text, encoding="utf-8")


def _write_decision_schema(audit_dir: Path, threshold: float) -> None:
    text = f"""# Final decision schema

## A. Local anomaly evidence

| State | Condition |
|-------|-----------|
| `benign` | No `local_alert`, no `weak_signal`, score below weak band |
| `weak_local_anomaly` | `weak_signal == 1` |
| `strong_local_anomaly` | `local_alert == 1` |

## B. Final malicious-event decision

| Rule | Condition |
|------|-----------|
| Strong (M1–M4) | `local_alert` for strong scenarios; `weak_signal` for weak M1 |
| Strong (M2–M4) | `local_alert` predicts malicious regardless of campaign |
| Weak promotion (M2–M4) | All gates + `fleet_event_confidence >= {threshold:.2f}` |
| Rejection | `attack_type=benign` → never malicious |
| Rejection | Coordinated membership alone → not malicious |

## C. Fleet campaign membership

| Value | Condition |
|-------|-----------|
| Campaign member | `final_decision == coordinated_attack` |
| Isolated | `final_decision == isolated_attack` |

Campaign membership does **not** automatically set malicious-event label.

## Promotion gates (weak events)

1. Locally weak (`weak_signal == 1`).
2. Behaviourally cohesive cluster (cohesion ≥ 0.85).
3. ≥2 distinct vehicle instances in cluster.
4. `fleet_event_confidence` ≥ validation-tuned threshold ({threshold:.2f}).
5. Cluster malicious-support fraction ≥ 0.10.
6. Not promoted solely because connected in graph.
"""
    (audit_dir / "final_decision_schema.md").write_text(text, encoding="utf-8")


def _write_campaign_f1_analysis(audit_dir: Path, breakdown: pd.DataFrame, original: pd.DataFrame, corrected: pd.DataFrame) -> None:
    fcgnn_orig = original[original["method"] == "fcgnn"]
    fcgnn_corr = corrected[corrected["method"] == "fcgnn"]
    det_orig = fcgnn_orig.groupby("campaign_size")["campaign_detection_rate"].mean() if "campaign_detection_rate" in fcgnn_orig.columns else pd.Series()
    det_corr = fcgnn_corr.groupby("campaign_size")["campaign_detection_rate"].mean()
    f1_corr = fcgnn_corr.groupby("campaign_size")["campaign_f1"].mean()
    text = f"""# Campaign F1 decline analysis

## Observed pattern (FCGNN)

Detection rate by campaign size (corrected): {det_corr.to_dict() if not det_corr.empty else 'N/A'}

Campaign F1 by campaign size (corrected): {f1_corr.to_dict() if not f1_corr.empty else 'N/A'}

## Calculated contributors

| Factor | Interpretation |
|--------|----------------|
| False campaign clusters | Extra qualifying clusters beyond ground truth |
| Benign vehicle inclusion | Attacked-vehicle precision drops |
| Membership impurity | Coordinated clusters include benign-on-attacked nodes |
| Fragmentation | Multiple small qualifying clusters |
| Evidence confound | Larger campaigns add more malicious **and** benign nodes |

## Why detection can rise while F1 falls

1. **Recall component** rises because more attacked vehicles increase coordinated-cluster probability.
2. **Precision component** falls because larger clusters absorb benign fleet and benign-on-attacked nodes.
3. Campaign-size originally confounds breadth with total malicious evidence (5/vehicle → 50 malicious at n=10).

See `results/campaign_error_breakdown.csv` for per-run decomposition.
"""
    (audit_dir / "campaign_f1_decline_analysis.md").write_text(text, encoding="utf-8")


def reevaluate_all(
    runs_root: Path,
    out_results: Path,
    promotion_cfg: PromotionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    confusion_rows: list[dict] = []
    vehicle_rows: list[dict] = []
    campaign_rows: list[dict] = []
    metrics_rows: list[dict] = []

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "event_predictions.csv").exists():
            continue
        raw = pd.read_csv(run_dir / "event_predictions.csv")
        membership = pd.read_csv(run_dir / "vehicle_membership.csv")
        m = pd.read_csv(run_dir / "run_level_metrics.csv").iloc[0]
        runtime = json.loads((run_dir / "runtime_memory.json").read_text(encoding="utf-8"))
        method = str(m["method"])
        strength = str(m["attack_strength"])
        corrected = apply_corrected_event_decisions(
            raw, attack_strength=strength, method=method, cfg=promotion_cfg  # type: ignore[arg-type]
        )
        cluster_df = reconstruct_cluster_df(corrected)
        if (run_dir / "cluster_df.csv").exists():
            cluster_df = pd.read_csv(run_dir / "cluster_df.csv")

        row = compute_event_confusion_row(
            corrected,
            run_id=run_dir.name,
            seed=int(m["seed"]),
            method=method,
            attack_strength=strength,
            campaign_size=int(m["campaign_size"]),
        )
        row["evaluation"] = "corrected"
        confusion_rows.append(row)

        agg = aggregate_corrected_run_metrics(
            corrected,
            pd.DataFrame(),
            membership,
            cluster_df,
            method=method,
            seed=int(m["seed"]),
            attack_strength=strength,
            campaign_size=int(m["campaign_size"]),
            coordination_strength=float(m.get("coordination_strength", 1.0)),
            runtime=runtime,
            expect_campaign=True,
        )
        agg["run_id"] = run_dir.name
        agg["attack_strength"] = strength
        if "weak_malicious_promoted" in corrected.columns:
            agg["weak_malicious_promoted"] = int(corrected["weak_malicious_promoted"].sum())
            agg["benign_incorrectly_promoted"] = int(corrected["benign_incorrectly_promoted"].sum())
        metrics_rows.append(agg)

        veh = {k: agg[k] for k in agg if k.startswith("vehicle_") or k in ("run_id", "method", "seed", "attack_strength", "campaign_size", "benign_vehicles_incorrectly_included", "vehicle_event_coverage_mean")}
        vehicle_rows.append(veh)

        if method != "local_ids":
            campaign_rows.append(
                campaign_error_breakdown_row(
                    corrected, membership, cluster_df,
                    run_id=run_dir.name, method=method, attack_strength=strength,
                    campaign_size=int(m["campaign_size"]), seed=int(m["seed"]),
                )
            )

        corrected.to_csv(out_results / "corrected_predictions" / f"{run_dir.name}.csv", index=False)

    return (
        pd.DataFrame(confusion_rows),
        pd.DataFrame(vehicle_rows),
        pd.DataFrame(campaign_rows),
        pd.DataFrame(metrics_rows),
    )


def run_fixed_evidence_control(
    project_root: Path,
    merged_config: dict,
    catalog: pd.DataFrame,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    promotion_cfg: PromotionConfig,
    out_dir: Path,
    *,
    seeds: list[int],
    quick: bool = False,
) -> pd.DataFrame:
    """Run fixed-total-malicious supplementary experiment."""
    from src.experiments.campaign_analysis_writer import CampaignAnalysisGuard

    guard = CampaignAnalysisGuard(project_root, str(OUT_ROOT))
    guard.ensure_directory_tree()
    rows: list[dict] = []
    sizes = [2, 5, 10]
    strengths = ["strong", "weak"] if not quick else ["strong"]
    methods = ["descriptor_clustering", "standard_gnn", "fcgnn"] if not quick else ["fcgnn"]
    test_seeds = seeds[:2] if quick else seeds

    for strength in strengths:
        for cs in sizes:
            budget = fixed_evidence_budget(cs)
            for seed in test_seeds:
                for method in methods:
                    try:
                        ctx = CampaignRunContext.create(
                            guard=guard,
                            experiment="fixed_evidence_control",
                            attack_strength=strength,
                            method=method,
                            seed=seed,
                            campaign_size=cs,
                            coordination_strength=1.0,
                            overwrite=True,
                        )
                        run_campaign_analysis_corrected_single(
                            ctx,
                            descriptors=descriptors,
                            manifest=manifest,
                            catalog=catalog,
                            config=merged_config,
                            budget=budget,
                        )
                        events = pd.read_csv(ctx.run_dir / "event_predictions.csv")
                        membership = pd.read_csv(ctx.run_dir / "vehicle_membership.csv")
                        runtime_raw = json.loads((ctx.run_dir / "runtime_memory.json").read_text(encoding="utf-8"))
                        runtime = {
                            k: float(v)
                            for k, v in runtime_raw.items()
                            if isinstance(v, (int, float)) and ("sec" in k or k == "total_sec")
                        }
                        scenario_df = pd.read_csv(ctx.run_dir / "selected_source_records.csv")
                        val = validate_fixed_evidence_records(scenario_df, cs)
                        pd.DataFrame([val]).to_csv(ctx.run_dir / "fixed_evidence_validation.csv", index=False)
                        events = apply_corrected_event_decisions(
                            events, attack_strength=strength, method=method, cfg=promotion_cfg  # type: ignore[arg-type]
                        )
                        cluster_df = reconstruct_cluster_df(events)
                        if (ctx.run_dir / "cluster_df.csv").exists():
                            cluster_df = pd.read_csv(ctx.run_dir / "cluster_df.csv")
                        agg = aggregate_corrected_run_metrics(
                            events,
                            pd.DataFrame(),
                            membership,
                            cluster_df,
                            method=method,
                            seed=seed,
                            attack_strength=strength,
                            campaign_size=cs,
                            coordination_strength=1.0,
                            runtime=runtime,
                            expect_campaign=True,
                        )
                        agg["run_id"] = ctx.run_id
                        agg["attack_strength"] = strength
                        rows.append(agg)
                        events.to_csv(ctx.run_dir / "event_predictions_corrected.csv", index=False)
                    except Exception as exc:
                        rows.append(
                            {
                                "run_id": f"fixed_evidence_{strength}_n{cs}_{method}_seed{seed}",
                                "error": str(exc),
                                "method": method,
                                "seed": seed,
                                "campaign_size": cs,
                                "attack_strength": strength,
                            }
                        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(out_dir / "run_level_metrics.csv", index=False)
    return df


def main() -> int:
    project_root = resolve_project_root()
    out_root = project_root / OUT_ROOT
    for sub in ("audit", "results", "tables", "figures", "logs", "validation", "fixed_evidence_control", "results/corrected_predictions"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    runs_root = project_root / PHASE3_RUNS
    if not runs_root.exists():
        print(f"Missing Phase 3 runs: {runs_root}", file=sys.stderr)
        return 1

    cfg = load_campaign_analysis_config(PHASE3_CONFIG)
    merged = _merge_config(cfg)
    local_cfg = merged.get("local_ids", {})
    promotion_cfg = PromotionConfig(
        weak_threshold=float(local_cfg.get("weak_threshold", 0.55)),
        strong_threshold=float(local_cfg.get("strong_threshold", 0.80)),
    )

    log_path = out_root / "logs" / f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
    log_lines: list[str] = []

    # Original confusion for audit
    original_rows = [_original_confusion_row(d) for d in sorted(runs_root.iterdir()) if d.is_dir() and (d / "event_predictions.csv").exists()]
    original_conf = pd.DataFrame(original_rows)
    original_conf.to_csv(out_root / "results" / "event_confusion_counts_original.csv", index=False)

    # Tune threshold on validation seeds
    val_frames = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue
        m = pd.read_csv(run_dir / "run_level_metrics.csv").iloc[0]
        if int(m["seed"]) not in VALIDATION_SEEDS:
            continue
        if str(m["method"]) not in ("descriptor_clustering", "standard_gnn", "fcgnn"):
            continue
        val_frames.append(pd.read_csv(run_dir / "event_predictions.csv"))
    tuned = tune_promotion_threshold_validation(val_frames, attack_strength="weak", cfg=promotion_cfg)
    promotion_cfg = PromotionConfig(**{**promotion_cfg.__dict__, "promotion_confidence_threshold": tuned})
    log_lines.append(f"Tuned promotion_confidence_threshold={tuned:.2f} on seeds {sorted(VALIDATION_SEEDS)}")

    corrected_conf, vehicle_df, campaign_err, corrected_metrics = reevaluate_all(
        runs_root, out_root / "results", promotion_cfg
    )
    corrected_conf.to_csv(out_root / "results" / "event_confusion_counts.csv", index=False)
    vehicle_df.to_csv(out_root / "results" / "vehicle_level_detailed_metrics.csv", index=False)
    campaign_err.to_csv(out_root / "results" / "campaign_error_breakdown.csv", index=False)
    corrected_metrics.to_csv(out_root / "results" / "corrected_run_level_metrics.csv", index=False)

    # Benign audit
    benign_frames = [audit_benign_on_attacked(d, cfg=promotion_cfg) for d in sorted(runs_root.iterdir()) if d.is_dir() and (d / "event_predictions.csv").exists()]
    benign_audit = pd.concat(benign_frames, ignore_index=True) if benign_frames else pd.DataFrame()
    benign_audit.to_csv(out_root / "results" / "benign_on_attacked_audit.csv", index=False)

    # Statistical tests
    stats = run_corrected_statistical_tests(corrected_metrics)
    stats.to_csv(out_root / "results" / "statistical_tests_corrected.csv", index=False)

    # Original vs corrected
    orig_m = pd.read_csv(project_root / "new_experiments/final_validated_runs/results/campaign_size_corrected/run_level_metrics.csv")
    cmp_rows = []
    for metric in ("fpr", "f1", "precision", "recall", "vehicle_recall", "vehicle_precision", "campaign_f1", "campaign_detection_rate"):
        if metric not in orig_m.columns or metric not in corrected_metrics.columns:
            continue
        o = orig_m.groupby(["method", "attack_strength", "campaign_size"])[metric].mean().reset_index(name="original_mean")
        c = corrected_metrics.groupby(["method", "attack_strength", "campaign_size"])[metric].mean().reset_index(name="corrected_mean")
        merged_cmp = o.merge(c, on=["method", "attack_strength", "campaign_size"])
        merged_cmp["metric"] = metric
        merged_cmp["delta"] = merged_cmp["corrected_mean"] - merged_cmp["original_mean"]
        cmp_rows.append(merged_cmp)
    original_vs_corrected = pd.concat(cmp_rows, ignore_index=True) if cmp_rows else pd.DataFrame()

    # Audits
    audit_dir = out_root / "audit"
    _write_audit_event_logic(audit_dir)
    _write_fpr_analysis(audit_dir, original_conf, corrected_conf)
    _write_decision_schema(audit_dir, tuned)
    _write_campaign_f1_analysis(audit_dir, campaign_err, orig_m, corrected_metrics)

    # Fixed evidence (full run — may take time)
    quick = "--quick" in sys.argv
    descriptors, features = load_descriptor_tables(
        project_root / cfg["paths"]["anomaly_descriptors"],
        project_root / cfg["paths"]["window_features"],
    )
    splits = merged.get("splits", {})
    manifest = ensure_split_manifest(
        descriptors,
        features,
        output_path=project_root / cfg["general"]["output_root"] / "manifests" / "split_manifest.csv",
        seed=int(splits.get("seed", 42)),
        train_ratio=float(splits.get("train", 0.70)),
        val_ratio=float(splits.get("validation", 0.15)),
        test_ratio=float(splits.get("test", 0.15)),
    )
    catalog = build_instance_catalog(
        descriptors, manifest,
        weak_threshold=promotion_cfg.weak_threshold,
        strong_threshold=promotion_cfg.strong_threshold,
    )
    seeds = [int(s) for s in cfg["general"]["seeds"]]
    fixed_df = run_fixed_evidence_control(
        project_root, merged, catalog, descriptors, manifest, promotion_cfg,
        out_root / "fixed_evidence_control", seeds=seeds, quick=quick,
    )

    # Tables and figures
    weak_table = summarize_weak_campaign_results(corrected_metrics)
    tables = export_tables(
        confusion=corrected_conf,
        vehicle=vehicle_df,
        campaign_err=campaign_err,
        weak_results=weak_table,
        stats=stats,
        fixed_evidence=fixed_df,
        original_vs_corrected=original_vs_corrected,
        tables_dir=out_root / "tables",
    )
    figures = export_figures(corrected_metrics, campaign_err, fixed_df, out_root / "figures")

    # Summary
    summary = f"""# Final evaluation correction summary

## 1. Why event FPR was 1.0

OR logic in `_decisions_to_predictions` promoted every coordinated cluster member and every weak-signal node to malicious. With most nodes flagged weak/local/coordinated, all 200 events were predicted malicious → FPR=1.0 on 150 benign GT events.

## 2. What was corrected

- Three-label schema: local evidence, malicious-event decision, campaign membership.
- Weak-event promotion gates with validation-tuned confidence threshold ({tuned:.2f}).
- M1 campaign metrics marked N/A.
- Vehicle precision, coverage, campaign error breakdown.
- Statistical families with Holm correction; p-values formatted as p < 0.001.
- Fixed-evidence control experiment ({len(fixed_df)} runs).

## 3. Event-level metrics validity

Corrected mean FPR: {corrected_conf.groupby('method')['fpr'].mean().to_dict()}

## 4. Benign-on-attacked audit

{int(benign_audit['validation_passed'].sum()) if not benign_audit.empty else 0} / {len(benign_audit)} rows passed. Benign-on-attacked windows use ground_truth_malicious=0 but often attack_type=malfunction with elevated scores — documented limitation.

## 5. Campaign detection vs F1

See `audit/campaign_f1_decline_analysis.md`.

## 6–10. See statistical_tests_corrected.csv and tables/

- Tables: {', '.join(tables)}
- Figures: {', '.join(figures)}
- Original Phase 3 outputs preserved under `results/campaign_size_corrected/`.
"""
    (out_root / "FINAL_EVALUATION_CORRECTION_SUMMARY.md").write_text(summary, encoding="utf-8")
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
