"""Aggregate framework ablation metrics from validated run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.evaluation_correction.metrics import (
    aggregate_corrected_run_metrics,
    reconstruct_cluster_df,
)
from src.experiments.evaluation_correction.promotion import PromotionConfig, apply_corrected_event_decisions
from src.experiments.framework_ablation.config import (
    METHOD_TO_FRAMEWORK,
    SCENARIO_MAP,
    SUPPLEMENTARY_METHOD,
)

OUT_ROOT = Path("new_experiments/final_validated_runs/framework_ablation")
CORRECTED_METRICS = Path(
    "new_experiments/final_validated_runs/evaluation_correction/results/corrected_run_level_metrics.csv"
)
PHASE3_RUNS = Path("new_experiments/final_validated_runs/results/campaign_size_corrected/runs")
PHASE2_ROOT = Path("new_experiments/final_validated_runs/results")


def _scenario_id(row: pd.Series) -> str:
    if "scenario_key" in row and str(row["scenario_key"]).startswith("S"):
        return SCENARIO_MAP.get(str(row["scenario_key"]), str(row["scenario_key"])[:2])
    if "attack_strength" in row:
        return SCENARIO_MAP.get(str(row["attack_strength"]), "S?")
    return "S?"


def _add_framework_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["framework_config"] = out["method"].map(METHOD_TO_FRAMEWORK)
    out.loc[out["method"] == SUPPLEMENTARY_METHOD, "framework_config"] = "S_supplementary"
    out["scenario_id"] = out.apply(_scenario_id, axis=1)
    out["membership_purity"] = out.get("campaign_precision", np.nan)
    out["fragmentation"] = out.get("n_detected_campaign_clusters", 0) - out.get(
        "n_ground_truth_campaigns", 0
    ).clip(lower=0)
    return out


def load_corrected_campaign_size_metrics() -> pd.DataFrame:
    """Primary S3/S4 data from evaluation-corrected Phase 3 (200 nodes, coord=1.0)."""
    if not CORRECTED_METRICS.exists():
        return pd.DataFrame()
    df = pd.read_csv(CORRECTED_METRICS)
    df = df[df["method"].isin(list(METHOD_TO_FRAMEWORK.keys()))].copy()
    df["scenario_key"] = df["attack_strength"].map(
        {"strong": "S3_strong_campaign", "weak": "S4_weak_campaign"}
    )
    df["data_source"] = "campaign_size_corrected_corrected_eval"
    return _add_framework_columns(df)


def reevaluate_run_dir(
    run_dir: Path,
    *,
    promotion_cfg: PromotionConfig,
) -> dict[str, Any] | None:
    if not (run_dir / "event_predictions.csv").exists():
        return None
    mem_path = run_dir / "vehicle_membership.csv"
    if not mem_path.exists():
        mem_path = run_dir / "scenario_membership.csv"
    if not mem_path.exists():
        return None
    raw = pd.read_csv(run_dir / "event_predictions.csv")
    membership = pd.read_csv(mem_path)
    m = pd.read_csv(run_dir / "run_level_metrics.csv").iloc[0]
    method = str(m["method"])
    scenario_key = str(m.get("scenario_key", ""))
    attack_strength = "strong" if "S3" in scenario_key else "weak" if "S4" in scenario_key else str(
        m.get("attack_strength", "strong")
    )
    if method not in METHOD_TO_FRAMEWORK and method != SUPPLEMENTARY_METHOD:
        return None
    corrected = apply_corrected_event_decisions(
        raw,
        attack_strength=attack_strength,  # type: ignore[arg-type]
        method=method,
        cfg=promotion_cfg,
    )
    cluster_df = reconstruct_cluster_df(corrected)
    if (run_dir / "cluster_df.csv").exists():
        cluster_df = pd.read_csv(run_dir / "cluster_df.csv")
    runtime: dict[str, float] = {}
    if (run_dir / "runtime_memory.json").exists():
        rt = json.loads((run_dir / "runtime_memory.json").read_text(encoding="utf-8"))
        runtime = {k: float(v) for k, v in rt.items() if isinstance(v, (int, float)) and "sec" in k}
    expect_campaign = scenario_key in ("S3_strong_campaign", "S4_weak_campaign")
    agg = aggregate_corrected_run_metrics(
        corrected,
        pd.DataFrame(),
        membership,
        cluster_df,
        method=method,
        seed=int(m["seed"]),
        attack_strength=attack_strength,
        campaign_size=int(m.get("campaign_size", 0)),
        coordination_strength=float(m.get("coordination_strength", 1.0)),
        runtime=runtime,
        expect_campaign=expect_campaign,
    )
    agg["run_id"] = run_dir.name
    agg["scenario_key"] = scenario_key
    agg["graph_unique_undirected_edges"] = m.get("graph_unique_undirected_edges", np.nan)
    agg["measured_cross_vehicle_similarity"] = m.get(
        "cross_model_attack_similarity", m.get("campaign_similarity", np.nan)
    )
    if "weak_malicious_promoted" in corrected.columns:
        agg["weak_malicious_promoted"] = int(corrected["weak_malicious_promoted"].sum())
        agg["benign_incorrectly_promoted"] = int(corrected["benign_incorrectly_promoted"].sum())
    return agg


def load_safety_scenario_metrics(
    scenarios: tuple[str, ...] = ("S0_benign_control", "S1_isolated", "S2_non_coordinated"),
    *,
    promotion_cfg: PromotionConfig,
) -> pd.DataFrame:
    """Re-evaluate S0–S2 with corrected promotion (variable-node Phase 2 scenarios)."""
    rows: list[dict] = []
    for sc in scenarios:
        runs_dir = PHASE2_ROOT / sc / "runs"
        if not runs_dir.exists():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            row = reevaluate_run_dir(run_dir, promotion_cfg=promotion_cfg)
            if row and row["method"] in METHOD_TO_FRAMEWORK:
                row["data_source"] = f"phase2_{sc}_corrected_eval"
                row["node_budget_note"] = "variable_node_phase2"
                rows.append(row)
    return _add_framework_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()


def load_coordination_ablation_runs(
    ablation_root: Path,
    *,
    promotion_cfg: PromotionConfig,
) -> pd.DataFrame:
    runs_dir = ablation_root / "results" / "coordination_strength" / "runs"
    if not runs_dir.exists():
        return pd.DataFrame()
    rows: list[dict] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        row = reevaluate_run_dir(run_dir, promotion_cfg=promotion_cfg)
        if row:
            row["data_source"] = "framework_ablation_coordination"
            row["node_budget_note"] = "fixed_200_nodes"
            rows.append(row)
    return _add_framework_columns(pd.DataFrame(rows)) if rows else pd.DataFrame()


def load_supplementary_standard_gnn(
    *,
    promotion_cfg: PromotionConfig,
) -> pd.DataFrame:
    """Standard GNN from corrected Phase 3 for supplementary S1 table."""
    rows: list[dict] = []
    if not PHASE3_RUNS.exists():
        return pd.DataFrame()
    for run_dir in sorted(PHASE3_RUNS.iterdir()):
        if "standard_gnn" not in run_dir.name:
            continue
        row = reevaluate_run_dir(run_dir, promotion_cfg=promotion_cfg)
        if row:
            row["data_source"] = "campaign_size_corrected_standard_gnn_supplementary"
            rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_master_metrics(
    *,
    promotion_cfg: PromotionConfig | None = None,
    ablation_root: Path | None = None,
) -> pd.DataFrame:
    promotion_cfg = promotion_cfg or PromotionConfig()
    ablation_root = ablation_root or OUT_ROOT
    frames: list[pd.DataFrame] = []
    primary = load_corrected_campaign_size_metrics()
    if not primary.empty:
        frames.append(primary)
    safety = load_safety_scenario_metrics(promotion_cfg=promotion_cfg)
    if not safety.empty:
        frames.append(safety)
    coord = load_coordination_ablation_runs(ablation_root, promotion_cfg=promotion_cfg)
    if not coord.empty:
        # Prefer coord runs at cs=5 for overlapping coord strengths; keep primary for cs 2/10
        coord_cs5 = coord[coord["campaign_size"] == 5]
        if not coord_cs5.empty:
            if not primary.empty:
                primary_no_overlap = primary[
                    ~(
                        (primary["campaign_size"] == 5)
                        & (primary["coordination_strength"].isin(coord_cs5["coordination_strength"].unique()))
                    )
                ]
                frames = [primary_no_overlap, safety] if not safety.empty else [primary_no_overlap]
            frames.append(coord_cs5)
        else:
            frames.append(coord)
    if not frames:
        return pd.DataFrame()
    master = pd.concat(frames, ignore_index=True)
    master = master.drop_duplicates(
        subset=["framework_config", "scenario_id", "seed", "campaign_size", "coordination_strength"],
        keep="last",
    )
    return master
