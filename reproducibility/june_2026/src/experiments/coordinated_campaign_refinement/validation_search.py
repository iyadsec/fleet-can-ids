"""Validation-only scenario search for connectivity and DBSCAN."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.campaign_analysis_corrected import REQUIRED_SEEDS, generate_corrected_campaign_scenario
from src.experiments.coordinated_campaign_refinement.refinement_pipeline import RefinementDesign, run_refinement_fcgnn
from src.experiments.model_diversity_final_tuned.false_campaign_metrics import compute_false_campaign_breakdown
from src.experiments.evaluation_correction.metrics import reconstruct_cluster_df
from src.experiments.vehicle_instance_builder import build_instance_catalog, _segment_rows

VAL_SEEDS = [131, 137, 149]


def _validation_catalog(catalog: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    val_files = set(manifest[manifest["split"] == "validation"]["source_file"].astype(str))
    return catalog[catalog["source_file"].astype(str).isin(val_files)].copy()


def build_validation_scenarios(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
    budget,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Strong cs 2/5/10 on validation split only."""
    val_cat = _validation_catalog(catalog, manifest)
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for cs in (2, 5, 10):
        for seed in VAL_SEEDS[:1]:
            key = f"val_strong_cs{cs}_seed{seed}"
            try:
                scen, _, mem, _, _ = generate_corrected_campaign_scenario(
                    attack_strength="strong",
                    seed=seed + cs * 1000,
                    descriptors=descriptors,
                    manifest=manifest,
                    catalog=val_cat,
                    config=config,
                    campaign_size=cs,
                    coordination_strength=1.0,
                    budget=budget,
                )
                scenarios[key] = (scen, mem)
            except ValueError:
                continue
    return scenarios


def _score_metrics(breakdown: dict[str, float], cand_stats: dict) -> float:
    return (
        float(breakdown.get("campaign_f1", 0))
        + float(breakdown.get("campaign_membership_precision", 0))
        - float(breakdown.get("fragmentation", 0)) * 0.5
        - float(cand_stats.get("benign_incorrectly_promoted", 0)) * 0.02
    )


def connectivity_search(
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: dict[str, Any],
    out_path: Any,
) -> pd.DataFrame:
    taus = [0.85, 0.90, 0.95]
    k_cross_opts = [3, 5, 10]
    k_same_opts = [0, 2, 5]
    rows = []
    for tau, k_cross, k_same in product(taus, k_cross_opts, k_same_opts):
        design = RefinementDesign(
            design_id="D2_search",
            candidate_only=True,
            similarity_threshold=tau,
            top_k_same_vehicle=k_same,
            top_k_cross_vehicle=k_cross,
        )
        scores = []
        for key, (scen, mem) in scenarios.items():
            try:
                outputs, cand_stats, _, _ = run_refinement_fcgnn(scen, mem, config, seed=131, design=design)
                br = compute_false_campaign_breakdown(
                    outputs.event_predictions, mem, outputs.cluster_df, expect_campaign=True
                )
                gs = outputs.graph_stats.iloc[0]
                if gs.get("cross_vehicle_edges", 0) == 0:
                    continue
                if gs.get("isolated_node_percentage", 0) > 50:
                    continue
                scores.append(_score_metrics(br, cand_stats))
            except Exception:
                continue
        if scores:
            rows.append({
                "similarity_threshold": tau,
                "max_cross_vehicle_neighbors": k_cross,
                "max_same_vehicle_neighbors": k_same,
                "mean_validation_score": float(np.mean(scores)),
                "n_scenarios": len(scores),
                "eligible": True,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mean_validation_score", ascending=False)
    df.to_csv(out_path, index=False)
    return df


def dbscan_search(
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    config: dict[str, Any],
    design: RefinementDesign,
    out_path: Any,
) -> pd.DataFrame:
    eps_opts = [0.5, 0.8, 1.0, 1.2]
    min_s_opts = [2, 3, 4, 5]
    rows = []
    for eps, ms in product(eps_opts, min_s_opts):
        d = RefinementDesign(**{**design.__dict__, "dbscan_eps": eps, "dbscan_min_samples": ms})
        scores = []
        for _, (scen, mem) in scenarios.items():
            try:
                outputs, cand_stats, _, _ = run_refinement_fcgnn(scen, mem, config, seed=131, design=d)
                br = compute_false_campaign_breakdown(
                    outputs.event_predictions, mem, outputs.cluster_df, expect_campaign=True
                )
                scores.append(_score_metrics(br, cand_stats))
            except Exception:
                continue
        if scores:
            rows.append({
                "dbscan_eps": eps,
                "dbscan_min_samples": ms,
                "mean_validation_score": float(np.mean(scores)),
                "n_scenarios": len(scores),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mean_validation_score", ascending=False)
    df.to_csv(out_path, index=False)
    return df


def select_frozen_params(connectivity_df: pd.DataFrame, dbscan_df: pd.DataFrame) -> RefinementDesign:
    c = connectivity_df.iloc[0] if not connectivity_df.empty else pd.Series({
        "similarity_threshold": 0.90, "max_cross_vehicle_neighbors": 5, "max_same_vehicle_neighbors": 2,
    })
    d = dbscan_df.iloc[0] if not dbscan_df.empty else pd.Series({"dbscan_eps": 0.8, "dbscan_min_samples": 3})
    return RefinementDesign(
        design_id="D2",
        candidate_only=True,
        similarity_threshold=float(c["similarity_threshold"]),
        top_k_same_vehicle=int(c["max_same_vehicle_neighbors"]),
        top_k_cross_vehicle=int(c["max_cross_vehicle_neighbors"]),
        dbscan_eps=float(d["dbscan_eps"]),
        dbscan_min_samples=int(d["dbscan_min_samples"]),
        consolidate_fragments=True,
    )
