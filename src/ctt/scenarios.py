"""Controlled fleet campaign scenario construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ctt.constants import ALL_VEHICLES, OUTPUT_ROOT, SCENARIO_SEEDS, SET_VEHICLE_POLICY
from src.ctt.utils import ensure_dir


SCENARIO_CONFIGS = {
    "benign_fleet_control": {
        "expected": "no_campaign",
        "attack_filter": ["benign"],
        "multi_vehicle": True,
        "same_family": False,
    },
    "isolated_attack": {
        "expected": "isolated",
        "attack_filter": None,  # one vehicle attacked
        "multi_vehicle": False,
        "same_family": False,
    },
    "unrelated_incidents": {
        "expected": "separate",
        "attack_filter": None,
        "multi_vehicle": True,
        "same_family": False,
        "different_families": True,
    },
    "strong_campaign": {
        "expected": "campaign",
        "attack_filter": None,
        "multi_vehicle": True,
        "same_family": True,
    },
    "weak_campaign": {
        "expected": "weak_campaign",
        "attack_filter": None,
        "multi_vehicle": True,
        "same_family": True,
        "weak_only": True,
    },
}

# Map set to vehicle for cross-fleet scenarios
SET_TO_VEHICLE = {s: SET_VEHICLE_POLICY[s]["known"] for s in SET_VEHICLE_POLICY}


def select_scenario_windows(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    """Select windows for a fleet scenario."""
    rng = np.random.default_rng(seed)
    merged = features.merge(predictions, on=["window_id", "vehicle_id", "dataset_set", "subset_name"], how="inner")
    config = SCENARIO_CONFIGS[scenario]

    # Use test data from each set for cross-vehicle fleet scenarios
    test_data = merged[merged["subset_name"].str.startswith("test_")]

    if scenario == "benign_fleet_control":
        selected = []
        for dataset_set, vid in SET_TO_VEHICLE.items():
            benign = test_data[(test_data["dataset_set"] == dataset_set) & (test_data["label"] == 0)]
            if not benign.empty:
                selected.append(benign.sample(n=min(50, len(benign)), random_state=seed))
        return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    if scenario == "isolated_attack":
        # Attack one vehicle only (set_01 / Impala), others benign
        selected = []
        attack_set = "set_01"
        atk = test_data[(test_data["dataset_set"] == attack_set) & (test_data["label"] == 1)]
        if not atk.empty:
            selected.append(atk.sample(n=min(30, len(atk)), random_state=seed))
        for dataset_set, vid in SET_TO_VEHICLE.items():
            if dataset_set == attack_set:
                continue
            benign = test_data[(test_data["dataset_set"] == dataset_set) & (test_data["label"] == 0)]
            if not benign.empty:
                selected.append(benign.sample(n=min(30, len(benign)), random_state=seed))
        return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    if scenario == "unrelated_incidents":
        families = ["dos", "fuzzing", "rpm_spoofing", "speed_spoofing"]
        selected = []
        for i, (dataset_set, vid) in enumerate(SET_TO_VEHICLE.items()):
            fam = families[i % len(families)]
            atk = test_data[
                (test_data["dataset_set"] == dataset_set)
                & (test_data["attack_type"] == fam)
                & (test_data["label"] == 1)
            ]
            if atk.empty:
                atk = test_data[(test_data["dataset_set"] == dataset_set) & (test_data["label"] == 1)]
            if not atk.empty:
                selected.append(atk.sample(n=min(20, len(atk)), random_state=seed + i))
        return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    if scenario in ("strong_campaign", "weak_campaign"):
        # Same attack family across vehicles
        target_family = "dos"
        selected = []
        for dataset_set in SET_TO_VEHICLE:
            atk = test_data[
                (test_data["dataset_set"] == dataset_set)
                & (test_data["attack_type"] == target_family)
                & (test_data["label"] == 1)
            ]
            if atk.empty:
                atk = test_data[(test_data["dataset_set"] == dataset_set) & (test_data["label"] == 1)]
            if not atk.empty:
                n = min(15 if scenario == "weak_campaign" else 25, len(atk))
                selected.append(atk.sample(n=n, random_state=seed))
        return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    return pd.DataFrame()


def run_scenario_evaluation(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    desc_df: pd.DataFrame,
    output_root: Path = OUTPUT_ROOT,
) -> pd.DataFrame:
    """Run all fleet scenarios across seeds."""
    from src.ctt.fleet_campaign import (
        build_pyg_data,
        dbscan_campaign_decision,
        evaluate_campaign,
        get_embeddings,
        train_graphsage,
    )
    from src.ctt.fleet_graph import build_behavioural_graph
    import json
    from src.ctt.features import LOCAL_FEATURE_COLUMNS

    DESCRIPTOR_FEATURE_COLS = [c for c in LOCAL_FEATURE_COLUMNS if not c.startswith("deviation")]

    results_dir = ensure_dir(output_root / "results" / "scenario_evaluation")
    all_results = []

    for scenario in SCENARIO_CONFIGS:
        scenario_dir = ensure_dir(output_root / "scenarios" / scenario)
        run_rows = []

        for seed in SCENARIO_SEEDS:
            windows = select_scenario_windows(features, predictions, scenario, seed)
            if windows.empty:
                continue
            windows.to_csv(scenario_dir / f"seed_{seed}_windows.csv", index=False)

            # Build scenario descriptors inline
            scen_pred = windows[windows["weak_prediction"] == 1].copy()
            if scen_pred.empty:
                run_rows.append({"scenario": scenario, "seed": seed, "campaign_detected": 0})
                continue

            desc_rows = []
            for _, row in scen_pred.iterrows():
                feat_vec = [float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0 for c in DESCRIPTOR_FEATURE_COLS]
                eid = f"EVT-{row['vehicle_id'][:3].upper()}-{int(row['window_id']):08d}"
                desc_rows.append({
                    "event_id": eid,
                    "descriptor_vector": json.dumps(feat_vec, separators=(",", ":")),
                    "vehicle_id": row["vehicle_id"],
                    "manufacturer": row.get("manufacturer", ""),
                    "attack_type": row["attack_type"],
                    "label": int(row["label"]),
                    "anomaly_score": float(row["anomaly_score"]),
                })
            scen_desc = pd.DataFrame(desc_rows)

            node_df, edge_df, _ = build_behavioural_graph(scen_desc)
            data = build_pyg_data(scen_desc, edge_df)
            model = train_graphsage(data, epochs=30)
            emb = get_embeddings(model, data)
            cluster_df = dbscan_campaign_decision(
                emb, data.event_ids, data.vehicle_ids, data.attack_types, data.labels
            )

            gt_vehicles = None
            if scenario in ("strong_campaign", "weak_campaign"):
                gt_vehicles = set(scen_desc[scen_desc["label"] == 1]["vehicle_id"].unique())

            metrics = evaluate_campaign(cluster_df, scenario, gt_vehicles)
            metrics["scenario"] = scenario
            metrics["seed"] = seed
            run_rows.append(metrics)

        if run_rows:
            scen_df = pd.DataFrame(run_rows)
            scen_df.to_csv(results_dir / f"{scenario}.csv", index=False)
            all_results.extend(run_rows)

    all_df = pd.DataFrame(all_results)
    if not all_df.empty:
        all_df.to_csv(results_dir / "run_level_metrics.csv", index=False)
    return all_df
