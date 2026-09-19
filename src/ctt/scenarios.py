"""Controlled fleet campaign scenario construction (CTT dataset-specific).

Separation of concerns:
  A. scenario construction / ground truth  (this module — may use labels)
  B. model inference                       (publication_fleet_core — no labels)
  C. post-hoc evaluation                   (fleet_campaign.evaluate_campaign)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.ctt.constants import ALL_VEHICLES, OUTPUT_ROOT, SCENARIO_SEEDS, SET_VEHICLE_POLICY
from src.ctt.features import LOCAL_FEATURE_COLUMNS
from src.ctt.utils import ensure_dir

if TYPE_CHECKING:
    from src.ctt.progress_logger import ProgressLogger

DESCRIPTOR_FEATURE_COLS = [c for c in LOCAL_FEATURE_COLUMNS if not c.startswith("deviation")]

SCENARIO_CONFIGS = {
    "benign_fleet_control": {
        "expected": "no_campaign",
        "attack_filter": ["benign"],
        "multi_vehicle": True,
        "same_family": False,
    },
    "isolated_attack": {
        "expected": "isolated",
        "attack_filter": None,
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

SET_TO_VEHICLE = {s: SET_VEHICLE_POLICY[s]["known"] for s in SET_VEHICLE_POLICY}


def select_scenario_windows(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    """Select windows for a fleet scenario (ground-truth aware — evaluation construction only)."""
    merge_keys = ["window_id", "vehicle_id", "dataset_set", "subset_name"]
    pred_cols = [c for c in predictions.columns if c not in features.columns or c in merge_keys]
    merged = features.merge(predictions[pred_cols], on=merge_keys, how="inner")
    config = SCENARIO_CONFIGS[scenario]
    del config  # retained for documentation / future filters

    test_data = merged[merged["subset_name"].astype(str).str.startswith("test_")]

    if scenario == "benign_fleet_control":
        selected = []
        for dataset_set, _vid in SET_TO_VEHICLE.items():
            benign = test_data[(test_data["dataset_set"] == dataset_set) & (test_data["label"] == 0)]
            if not benign.empty:
                selected.append(benign.sample(n=min(50, len(benign)), random_state=seed))
        return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    if scenario == "isolated_attack":
        selected = []
        attack_set = "set_01"
        atk = test_data[(test_data["dataset_set"] == attack_set) & (test_data["label"] == 1)]
        if not atk.empty:
            selected.append(atk.sample(n=min(30, len(atk)), random_state=seed))
        for dataset_set, _vid in SET_TO_VEHICLE.items():
            if dataset_set == attack_set:
                continue
            benign = test_data[(test_data["dataset_set"] == dataset_set) & (test_data["label"] == 0)]
            if not benign.empty:
                selected.append(benign.sample(n=min(30, len(benign)), random_state=seed))
        return pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()

    if scenario == "unrelated_incidents":
        families = ["dos", "fuzzing", "rpm_spoofing", "speed_spoofing"]
        selected = []
        for i, (dataset_set, _vid) in enumerate(SET_TO_VEHICLE.items()):
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


def _windows_to_scenario_descriptors(windows: pd.DataFrame) -> pd.DataFrame:
    """Build named-column descriptors for the shared 9-D GNN path (no JSON-only vectors)."""
    scen_pred = windows[windows["weak_prediction"] == 1].copy()
    if scen_pred.empty:
        return pd.DataFrame()

    rows = []
    for _, row in scen_pred.iterrows():
        eid = f"EVT-{str(row['vehicle_id'])[:3].upper()}-{int(row['window_id']):08d}"
        rec: dict = {
            "event_id": eid,
            "vehicle_id": row["vehicle_id"],
            "vehicle_token": str(row["vehicle_id"]),
            "vehicle_model": str(row["vehicle_id"]),
            "manufacturer": row.get("manufacturer", ""),
            "attack_type": row["attack_type"],
            "label": int(row["label"]),
            "anomaly_score": float(row["anomaly_score"]),
            "weak_prediction": int(row["weak_prediction"]),
            "strong_prediction": int(row.get("strong_prediction", 0)),
            "dataset_set": row.get("dataset_set", ""),
            "subset_name": row.get("subset_name", ""),
        }
        for c in DESCRIPTOR_FEATURE_COLS:
            if c in row.index and pd.notna(row[c]):
                rec[c] = float(row[c])
            else:
                rec[c] = 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def run_scenario_evaluation(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    desc_df: pd.DataFrame,
    output_root: Path = OUTPUT_ROOT,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    progress: ProgressLogger | None = None,
) -> pd.DataFrame:
    """Run all fleet scenarios across seeds using the shared publication fleet path."""
    from src.ctt.fleet_campaign import evaluate_campaign, run_fleet_campaign_inference
    from src.ctt.fleet_graph import fit_or_load_ctt_scaler
    from src.ctt.progress_logger import ProgressLogger

    del desc_df  # global descriptor pool unused; scenarios build descriptors from windows

    scenario_list = scenarios or list(SCENARIO_CONFIGS.keys())
    seed_list = seeds if seeds is not None else SCENARIO_SEEDS

    results_dir = ensure_dir(output_root / "results" / "scenario_evaluation")
    scaler_cache = output_root / "scalers" / "ctt_fleet_benign_scaler.json"
    # Fit scaler on available feature rows that look like train benign when present.
    scaler_source = features.copy()
    if "weak_prediction" not in scaler_source.columns and not predictions.empty:
        merge_keys = [c for c in ("window_id", "vehicle_id", "dataset_set", "subset_name") if c in features.columns]
        scaler_source = features.merge(
            predictions[
                [
                    c
                    for c in predictions.columns
                    if c in merge_keys
                    or c
                    in {
                        "anomaly_score",
                        "weak_prediction",
                        "strong_prediction",
                        "attack_type",
                        "label",
                        "subset_name",
                    }
                ]
            ],
            on=merge_keys,
            how="left",
            suffixes=("", "_pred"),
        )
    scaler = fit_or_load_ctt_scaler(scaler_source, cache_path=scaler_cache)

    all_results: list[dict] = []

    for scenario in scenario_list:
        if scenario not in SCENARIO_CONFIGS:
            continue
        scenario_dir = ensure_dir(output_root / "scenarios" / scenario)
        run_rows: list[dict] = []

        for seed in seed_list:
            windows = select_scenario_windows(features, predictions, scenario, seed)
            if windows.empty:
                continue
            windows.to_csv(scenario_dir / f"seed_{seed}_windows.csv", index=False)

            scen_desc = _windows_to_scenario_descriptors(windows)
            if scen_desc.empty:
                run_rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "campaign_detected": 0,
                        "campaign_f1": 0.0,
                    }
                )
                continue

            # Drop evaluation-only columns from a model-input view (kept on scen_desc for metrics).
            model_cols = [
                c
                for c in scen_desc.columns
                if c
                not in {
                    # Keep attack_type/label on the frame for post-hoc eval fields only;
                    # publication_fleet_core gate does not read them for qualification.
                }
            ]
            model_df = scen_desc[model_cols].copy()

            artifacts = run_fleet_campaign_inference(model_df, scaler=scaler, seed=seed)
            # Persist graph stats per run
            pd.DataFrame([artifacts.graph_stats]).to_csv(
                scenario_dir / f"seed_{seed}_graph_stats.csv", index=False
            )
            artifacts.node_decisions.to_csv(
                scenario_dir / f"seed_{seed}_node_decisions.csv", index=False
            )

            gt_vehicles = None
            if scenario in ("strong_campaign", "weak_campaign"):
                gt_vehicles = set(
                    scen_desc.loc[scen_desc["label"].astype(int) == 1, "vehicle_id"].astype(str).unique()
                )

            metrics = evaluate_campaign(
                artifacts, scenario, ground_truth_campaign_vehicles=gt_vehicles
            )
            metrics["scenario"] = scenario
            metrics["seed"] = seed
            metrics["scaler_id"] = artifacts.scaler_id
            metrics["config_hash"] = artifacts.config.config_hash()
            metrics["train_epochs"] = artifacts.train_metrics.get("epochs")
            metrics["train_lr"] = artifacts.train_metrics.get("learning_rate")
            metrics["train_weight_decay"] = artifacts.train_metrics.get("weight_decay")
            metrics["train_lambda"] = artifacts.train_metrics.get("campaign_loss_weight")
            run_rows.append(metrics)
            if progress:
                progress.scenario_completed(scenario, seed)

        if run_rows:
            scen_df = pd.DataFrame(run_rows)
            scen_df.to_csv(results_dir / f"{scenario}.csv", index=False)
            all_results.extend(run_rows)

    all_df = pd.DataFrame(all_results)
    if not all_df.empty:
        all_df.to_csv(results_dir / "run_level_metrics.csv", index=False)
    return all_df
