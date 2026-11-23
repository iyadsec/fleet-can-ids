"""Joint shared-parameter search on mixed V0–V4 validation (staged grid)."""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from src.experiments.coordinated_campaign_refinement.refinement_pipeline import run_refinement_fcgnn
from src.experiments.final_shared_configuration.shared_config import SharedFleetConfiguration
from src.experiments.model_diversity_final_tuned.false_campaign_metrics import compute_false_campaign_breakdown


def _run_val(scenario_df, membership, config, seed, shared: SharedFleetConfiguration):
    return run_refinement_fcgnn(scenario_df, membership, config, seed, shared.to_refinement_design())


def _metrics(outputs, membership, expect_campaign: bool) -> dict[str, float]:
    br = compute_false_campaign_breakdown(
        outputs.event_predictions, membership, outputs.cluster_df, expect_campaign=expect_campaign
    )
    mp = float(br.get("campaign_membership_precision", 0))
    mr = float(br.get("campaign_membership_recall", 0))
    return {
        "campaign_f1": float(br.get("campaign_f1", 0)),
        "membership_f1": 2 * mp * mr / max(mp + mr, 1e-9),
        "campaign_recall": float(br.get("campaign_recall", 0)),
        "fragmentation": float(br.get("fragmentation", 0)),
        "incorrect_merging": float(br.get("incorrect_merging", 0)),
        "benign_vehicles_included": float(br.get("benign_vehicles_included", 0)),
        "false_campaign_clusters": float(br.get("false_campaign_cluster_count", 0)),
        "n_qualifying": float(br.get("n_detected_campaign_clusters", 0)),
    }


def _eval_config(
    shared: SharedFleetConfiguration,
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]],
    config: dict[str, Any],
) -> dict[str, float]:
    v0_fca, v1_coord, v2_merge = [], [], []
    v3_f1, v4_f1, v3_mf1, v4_mf1 = [], [], [], []
    benign_inc = []

    for run_id, (ann, mem, _) in scenarios.items():
        parts = run_id.split("_")
        scenario = parts[1] if len(parts) > 1 else "V0"
        seed = int(run_id.split("seed")[-1])
        expect = scenario in ("V3", "V4")
        try:
            outputs, _, _, _ = _run_val(ann, mem, config, seed, shared)
            m = _metrics(outputs, mem, expect)
        except Exception:
            return {"selection_score": -1e9, "valid": 0.0}
        if scenario == "V0":
            v0_fca.append(1.0 if m["n_qualifying"] > 0 else 0.0)
        elif scenario == "V1":
            v1_coord.append(1.0 if m["n_qualifying"] > 0 else 0.0)
        elif scenario == "V2":
            v2_merge.append(m["incorrect_merging"])
        elif scenario == "V3":
            v3_f1.append(m["campaign_f1"])
            v3_mf1.append(m["membership_f1"])
            benign_inc.append(m["benign_vehicles_included"])
        elif scenario == "V4":
            v4_f1.append(m["campaign_f1"])
            v4_mf1.append(m["membership_f1"])
            benign_inc.append(m["benign_vehicles_included"])

    if not v3_f1 or not v4_f1:
        return {"selection_score": -1e9, "valid": 0.0}

    obj = float(np.mean(v3_f1 + v4_f1 + v3_mf1 + v4_mf1))
    v0 = float(np.mean(v0_fca)) if v0_fca else 0.0
    v1 = float(np.mean(v1_coord)) if v1_coord else 0.0
    v2 = float(np.mean(v2_merge)) if v2_merge else 0.0
    ben = float(np.mean(benign_inc)) if benign_inc else 0.0
    feasible = v0 <= 0.05 and v1 == 0.0 and v2 <= 0.10 and ben <= 1.0
    score = obj if feasible else obj - 2.0 * v0 - v2 - 0.5 * ben
    return {
        "objective_score": obj,
        "selection_score": score,
        "v0_false_campaign_rate": v0,
        "v1_coordinated_rate": v1,
        "v2_incorrect_merging": v2,
        "mean_benign_included": ben,
        "mean_v3_campaign_f1": float(np.mean(v3_f1)),
        "mean_v4_campaign_f1": float(np.mean(v4_f1)),
        "constraints_met": float(feasible),
        "valid": 1.0,
    }


def joint_parameter_search(
    scenarios: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, SharedFleetConfiguration, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_cfg = SharedFleetConfiguration()
    best_score = -1e9
    best_diag: dict[str, Any] = {}
    constraints_met = False

    conn_grid = list(product([0.85, 0.90, 0.95], [2], [5, 10]))[:6]
    dbscan_grid = list(product([0.5, 0.8, 1.0], [2, 3, 4]))[:6]
    tail_grid = list(product([0.85], [2], [1, 2], [0.5]))[:2]

    for tau, k_same, k_cross in conn_grid:
        base = SharedFleetConfiguration(
            similarity_threshold=tau,
            max_same_vehicle_neighbors=k_same,
            max_cross_vehicle_neighbors=k_cross,
        )
        diag = _eval_config(base, scenarios, config)
        if diag.get("valid", 0) < 1:
            continue
        row = {**base.to_dict(), **diag, "search_stage": "connectivity"}
        rows.append(row)
        if diag["selection_score"] > best_score:
            best_score = diag["selection_score"]
            best_cfg = base
            best_diag = row
            constraints_met = bool(diag["constraints_met"])

    for eps, ms in dbscan_grid:
        cfg = SharedFleetConfiguration(
            **{**best_cfg.to_dict(), "dbscan_eps": eps, "dbscan_min_samples": ms}
        )
        diag = _eval_config(cfg, scenarios, config)
        if diag.get("valid", 0) < 1:
            continue
        row = {**cfg.to_dict(), **diag, "search_stage": "dbscan"}
        rows.append(row)
        if diag["selection_score"] > best_score:
            best_score = diag["selection_score"]
            best_cfg = cfg
            best_diag = row
            constraints_met = bool(diag["constraints_met"])

    for frag, mv, mcs, coh in tail_grid:
        cfg = SharedFleetConfiguration(
            **{
                **best_cfg.to_dict(),
                "fragment_centroid_threshold": frag,
                "minimum_distinct_vehicles": mv,
                "minimum_cross_vehicle_support": mcs,
                "minimum_campaign_cohesion": coh,
            }
        )
        diag = _eval_config(cfg, scenarios, config)
        if diag.get("valid", 0) < 1:
            continue
        row = {**cfg.to_dict(), **diag, "search_stage": "campaign_rules"}
        rows.append(row)
        if diag["selection_score"] > best_score:
            best_score = diag["selection_score"]
            best_cfg = cfg
            best_diag = row
            constraints_met = bool(diag["constraints_met"])

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("selection_score", ascending=False)
    return df, best_cfg, {
        "constraints_met": constraints_met,
        "best_diag": best_diag,
        "candidates_tested": len(rows),
    }
