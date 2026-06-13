"""Validation-only campaign gate grid search with constrained objectives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.experiments.model_diversity_final_tuned.scenario_cache import ScenarioCache, evaluate_gate_on_cache
from src.experiments.model_diversity_final_tuned.tuned_gate import TunedGateConfig, gate_parameter_grid


def load_gate_objective(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _aggregate_scenario_metrics(
    caches: list[ScenarioCache],
    gate: TunedGateConfig,
) -> dict[str, float]:
    per_run = [evaluate_gate_on_cache(cache, gate) for cache in caches]
    by_scenario: dict[str, list[dict[str, float]]] = {}
    for cache, m in zip(caches, per_run):
        by_scenario.setdefault(cache.scenario, []).append(m)

    def mean_metric(scenario: str, key: str, default: float = 0.0) -> float:
        vals = [r[key] for r in by_scenario.get(scenario, []) if key in r]
        return float(np.mean(vals)) if vals else default

    v3_f1 = mean_metric("V3", "campaign_f1")
    v4_f1 = mean_metric("V4", "campaign_f1")
    camp_runs = [m for c, m in zip(caches, per_run) if c.scenario in ("V3", "V4")]
    return {
        "V0_false_campaign_rate": mean_metric("V0", "false_campaign_alert_rate"),
        "V1_campaign_rate": mean_metric("V1", "false_campaign_alert_rate"),
        "V2_incorrect_merging_rate": mean_metric("V2", "incorrect_merging"),
        "V3_campaign_precision": mean_metric("V3", "campaign_precision"),
        "V3_campaign_recall": mean_metric("V3", "campaign_recall"),
        "V3_campaign_f1": v3_f1,
        "V4_campaign_precision": mean_metric("V4", "campaign_precision"),
        "V4_campaign_recall": mean_metric("V4", "campaign_recall"),
        "V4_campaign_f1": v4_f1,
        "mean_benign_vehicles_included": float(np.mean([m["benign_vehicles_included"] for m in per_run])),
        "membership_precision": float(np.mean([m["campaign_membership_precision"] for m in camp_runs])) if camp_runs else 0.0,
        "membership_recall": float(np.mean([m["campaign_membership_recall"] for m in camp_runs])) if camp_runs else 0.0,
        "validation_campaign_f1": float(np.mean([v3_f1, v4_f1])),
    }


def _check_constraints(metrics: dict[str, float], objective: dict[str, Any]) -> bool:
    c = objective.get("constraints", {})
    if metrics["V0_false_campaign_rate"] > c.get("V0_false_campaign_alert_rate_max", 0.05):
        return False
    if metrics["V1_campaign_rate"] > c.get("V1_campaign_alert_rate_max", 0.0):
        return False
    if metrics["V2_incorrect_merging_rate"] > c.get("V2_incorrect_merging_rate_max", 0.05):
        return False
    if metrics["mean_benign_vehicles_included"] > c.get("mean_benign_vehicles_included_max", 1.0):
        return False
    if metrics["membership_precision"] < c.get("membership_precision_min", 0.80):
        return False
    return True


def _pareto_frontier(rows: pd.DataFrame) -> pd.Series:
    """Mark Pareto-optimal candidates minimizing false campaign rate, maximizing F1."""
    fcr = rows["V0_false_campaign_rate"].to_numpy(dtype=float)
    f1 = rows["validation_campaign_f1"].to_numpy(dtype=float)
    n = len(rows)
    optimal = np.zeros(n, dtype=bool)
    order = np.argsort(fcr, kind="mergesort")
    best_f1 = -np.inf
    for idx in order:
        if f1[idx] >= best_f1:
            optimal[idx] = True
            best_f1 = f1[idx]
    return pd.Series(optimal, index=rows.index)


def run_gate_search(
    caches: list[ScenarioCache],
    objective_path: Path,
    output_csv: Path,
) -> tuple[TunedGateConfig, pd.DataFrame, dict[str, Any]]:
    objective = load_gate_objective(objective_path)
    cohesion_values = objective.get("derived_thresholds", {}).get("min_cluster_cohesion", [0.10, 0.15, 0.20])
    confidence_values = objective.get("derived_thresholds", {}).get("min_membership_confidence", [0.45, 0.55, 0.65])

    # Pre-evaluate per cache once for cohesion/confidence distribution hints
    if objective.get("derive_from_validation", True):
        cohesions = []
        confidences = []
        for cache in caches:
            if not cache.cluster_df.empty:
                cohesions.extend(cache.cluster_df["behavioral_cohesion"].astype(float).tolist())
            confidences.extend(cache.campaign_scores.astype(float).tolist())
        if cohesions:
            qs = np.quantile(cohesions, [0.25, 0.50, 0.75])
            cohesion_values = sorted({float(round(q, 2)) for q in qs})
        if confidences:
            qs = np.quantile(confidences, [0.25, 0.50, 0.75])
            confidence_values = sorted({float(round(q, 2)) for q in qs})

    grid = gate_parameter_grid(cohesion_values, confidence_values)
    rows: list[dict[str, Any]] = []
    best_cfg: TunedGateConfig | None = None
    best_score = -1.0
    best_fcr = 999.0
    feasible_found = False

    for i, gate in enumerate(grid):
        if i and i % 500 == 0:
            print(f"Gate search progress: {i}/{len(grid)}", flush=True)
        metrics = _aggregate_scenario_metrics(caches, gate)
        feasible = _check_constraints(metrics, objective)
        if feasible:
            feasible_found = True
        score = metrics["validation_campaign_f1"]
        row = {
            "candidate_id": f"gate_{i:04d}",
            "configuration": json.dumps(gate.to_dict()),
            "parameter_values": json.dumps(gate.to_dict()),
            **metrics,
            "constraint_pass": feasible,
            "pareto_optimal": False,
            "selected": False,
        }
        rows.append(row)
        if feasible and score > best_score:
            best_score, best_cfg = score, gate
        if not feasible_found and metrics["V0_false_campaign_rate"] < best_fcr:
            best_fcr = metrics["V0_false_campaign_rate"]
            if best_cfg is None or not feasible_found:
                best_cfg = gate
        elif not feasible_found and metrics["V0_false_campaign_rate"] == best_fcr and score > best_score:
            best_score, best_cfg = score, gate

    df = pd.DataFrame(rows)
    df["pareto_optimal"] = _pareto_frontier(df)
    if best_cfg is not None:
        sel_mask = df["configuration"] == json.dumps(best_cfg.to_dict())
        df.loc[sel_mask, "selected"] = True
    df.to_csv(output_csv, index=False)

    meta = {
        "n_candidates": len(grid),
        "feasible_found": feasible_found,
        "selected_config": best_cfg.to_dict() if best_cfg else {},
        "selected_hash": best_cfg.config_hash() if best_cfg else "",
        "test_data_used": False,
    }
    return best_cfg or TunedGateConfig(), df, meta
