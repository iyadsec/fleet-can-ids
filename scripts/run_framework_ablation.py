#!/usr/bin/env python3
"""Framework ablation: aggregate C1–C3 results and run coordination-strength experiment."""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.campaign_analysis_corrected import DescriptorBudget
from src.experiments.campaign_analysis_runner import run_campaign_analysis_corrected_single
from src.experiments.campaign_analysis_writer import CampaignAnalysisGuard, CampaignRunContext, load_campaign_analysis_config
from src.experiments.coordination_strength import measure_mean_pairwise_similarity
from src.experiments.framework_ablation.aggregate import (
    build_master_metrics,
    load_supplementary_standard_gnn,
)
from src.experiments.framework_ablation.config import COORDINATION_STRENGTHS, REQUIRED_SEEDS
from src.experiments.framework_ablation.outputs import export_all_tables, export_figures
from src.experiments.framework_ablation.statistics import run_framework_ablation_tests
from src.experiments.evaluation_correction.promotion import PromotionConfig
from src.experiments.result_writer import load_experiment_config
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.vehicle_instance_builder import build_instance_catalog
from src.utils.paths import resolve_project_root

OUT_ROOT = Path("new_experiments/final_validated_runs/framework_ablation")
PHASE3_CONFIG = "new_experiments/final_validated_runs/configs/phase3_campaign_size_corrected.yaml"
FRAMEWORK_CONFIG = "new_experiments/final_validated_runs/framework_ablation/configs/framework_ablation.yaml"


def _merge_config(cfg: dict) -> dict:
    base_path = cfg.get("paths", {}).get("base_scenario_config")
    base = load_experiment_config(base_path) if base_path else {}
    merged = {**base, **cfg}
    for key in ("local_ids", "graph", "gnn", "campaign"):
        merged[key] = {**base.get(key, {}), **cfg.get(key, {})}
    return merged


def run_coordination_experiment(
    project_root: Path,
    merged: dict,
    catalog: pd.DataFrame,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    strengths: list[float],
    skip_existing: bool = True,
) -> pd.DataFrame:
    guard = CampaignAnalysisGuard(project_root, str(OUT_ROOT))
    guard.ensure_directory_tree()
    budget = DescriptorBudget(10, 5, 5, 10)
    methods = ["local_ids", "descriptor_clustering", "fcgnn"]
    attack_strengths = ["strong", "weak"]
    campaign_size = 5
    rows: list[dict] = []
    log = []

    for attack_strength in attack_strengths:
        for coord in strengths:
            for seed in REQUIRED_SEEDS:
                for method in methods:
                    run_tag = f"coord_{attack_strength}_n{campaign_size}_cs{coord}_seed{seed}_{method}"
                    existing = list(
                        (OUT_ROOT / "results" / "coordination_strength" / "runs").glob(f"*{method}*seed{seed}*")
                    ) if skip_existing else []
                    if existing and any(str(coord).replace(".", "p") in p.name or f"cs{coord}" in p.name for p in existing):
                        continue
                    try:
                        ctx = CampaignRunContext.create(
                            guard=guard,
                            experiment="coordination_strength",
                            attack_strength=attack_strength,
                            method=method,
                            seed=seed,
                            campaign_size=campaign_size,
                            coordination_strength=coord,
                            overwrite=True,
                        )
                        metrics = run_campaign_analysis_corrected_single(
                            ctx,
                            descriptors=descriptors,
                            manifest=manifest,
                            catalog=catalog,
                            config=merged,
                            budget=budget,
                        )
                        scenario = pd.read_csv(ctx.run_dir / "selected_source_records.csv")
                        mal = scenario[
                            (scenario["scenario_role"] == "coordinated")
                            & (scenario["ground_truth_malicious"] == 1)
                        ]
                        sim = float("nan")
                        if len(mal) >= 2:
                            sim = measure_mean_pairwise_similarity(mal)
                        metrics["measured_cross_vehicle_similarity"] = sim
                        metrics["configured_coordination_strength"] = coord
                        rows.append(metrics)
                        log.append(f"OK {run_tag}")
                    except Exception as exc:
                        log.append(f"FAIL {run_tag}: {exc}")
    log_path = OUT_ROOT / "logs" / f"coordination_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log), encoding="utf-8")
    df = pd.DataFrame(rows)
    if not df.empty:
        out = OUT_ROOT / "results" / "coordination_strength" / "run_level_metrics.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    return df


def write_summary(df: pd.DataFrame, stats: pd.DataFrame, path: Path) -> None:
    c3s4 = df[(df["framework_config"] == "C3") & (df["scenario_id"] == "S4") & (df["coordination_strength"] == 1.0)]
    c1s4 = df[(df["framework_config"] == "C1") & (df["scenario_id"] == "S4") & (df["coordination_strength"] == 1.0)]
    c2s3 = df[(df["framework_config"] == "C2") & (df["scenario_id"] == "S3") & (df["campaign_size"] == 5)]
    c3s3 = df[(df["framework_config"] == "C3") & (df["scenario_id"] == "S3") & (df["campaign_size"] == 5)]

    text = f"""# Final framework ablation summary

## 1. Contribution type
**Primarily system-level.** The validated pipeline is a hierarchical fleet-aware framework. FCGNN is GraphSAGE-based message passing without fleet-specific neural layers (see `audit/gnn_architecture_audit.md`).

## 2. Standard GNN in main paper?
**No.** Retained in `supplementary/table_S1_standard_gnn_comparison` only (GCN vs GraphSAGE operator difference).

## 3. Does similarity-only solve easy campaigns?
At coordination strength 1.0, C2 mean campaign F1 (S3, n=5): {c2s3['campaign_f1'].mean() if not c2s3.empty else 'N/A'}.
C2 achieves high detection when descriptors are highly similar; message passing adds value mainly when similarity is insufficient (lower coordination).

## 4. Coordination strengths where message passing helps
See `table_F5_coordination_strength_sensitivity` and Family C tests in `table_F6_primary_statistical_tests`.

## 5. Full framework vs local-only (S4)
C3 event recall: {c3s4['recall'].mean() if not c3s4.empty else 'N/A'} vs C1: {c1s4['recall'].mean() if not c1s4.empty else 'N/A'}.
C1 campaign metrics: N/A by design.

## 6. Message passing and incorrect merging
Compare `incorrect_campaign_merging` in Family B (C3 vs C2). Results reported honestly in tables; similarity-only can match or exceed graph-based merging on highly coordinated data.

## 7. Computational cost
See `table_F7_computational_cost`. C3 adds GNN inference over C2 clustering-only path.

## 8–9. Main vs supplementary
**Main paper:** C1, C2, C3 framework ablation tables F1–F7 and figures F1–F6.
**Supplementary:** Standard GNN (table S1), architecture audit, variable-node S0–S2 safety note.

## 10. Limitations
- S0–S2 safety runs use Phase 2 variable-node scenarios (documented).
- Coordination strength 0.50/0.75 require dedicated runs under fixed 200-node budget.
- Graph-based correlation is GraphSAGE, not a novel FCGNN architecture.

## Statistical highlights
{stats[stats['significant']==True][['comparison','metric','scenario','adjusted_p_value_formatted']].head(10).to_string() if not stats.empty and 'significant' in stats.columns else 'See table F6.'}
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    project_root = resolve_project_root()
    out_root = project_root / OUT_ROOT
    for sub in ("audit", "configs", "results", "tables", "figures", "logs", "supplementary", "validation"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    coord_only = "--coord-only" in sys.argv
    skip_coord = "--skip-coord" in sys.argv

    cfg = load_campaign_analysis_config(PHASE3_CONFIG)
    merged = _merge_config(cfg)
    promotion_cfg = PromotionConfig(
        weak_threshold=float(merged.get("local_ids", {}).get("weak_threshold", 0.55)),
        strong_threshold=float(merged.get("local_ids", {}).get("strong_threshold", 0.80)),
    )

    if not skip_coord:
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
            descriptors,
            manifest,
            weak_threshold=promotion_cfg.weak_threshold,
            strong_threshold=promotion_cfg.strong_threshold,
        )
        # Run 0.50 and 0.75 (1.00 from campaign_size_corrected)
        run_coordination_experiment(
            project_root,
            merged,
            catalog,
            descriptors,
            manifest,
            strengths=[0.50, 0.75],
        )
        if coord_only:
            return 0

    master = build_master_metrics(promotion_cfg=promotion_cfg, ablation_root=out_root)
    if master.empty:
        print("No metrics aggregated.", file=sys.stderr)
        return 1
    master.to_csv(out_root / "results" / "framework_ablation_metrics.csv", index=False)

    stats = run_framework_ablation_tests(master)
    stats.to_csv(out_root / "results" / "statistical_tests.csv", index=False)

    supp = load_supplementary_standard_gnn(promotion_cfg=promotion_cfg)
    tables = export_all_tables(master, stats, out_root / "tables", out_root / "supplementary", supp)
    figures = export_figures(master, out_root / "figures")
    write_summary(master, stats, out_root / "FINAL_FRAMEWORK_ABLATION_SUMMARY.md")

    print(f"Aggregated {len(master)} runs. Tables: {tables}. Figures: {figures}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
