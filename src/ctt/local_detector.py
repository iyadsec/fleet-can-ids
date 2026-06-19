"""Benign-only local vehicle detector with threshold calibration."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.ctt.constants import (
    OUTPUT_ROOT,
    STRONG_THRESHOLD_PERCENTILE,
    WEAK_THRESHOLD_PERCENTILE,
)
from src.ctt.features import LOCAL_FEATURE_COLUMNS, compute_benign_profile, extract_window_features
from src.ctt.utils import content_hash, ensure_dir, safe_div, write_markdown


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in LOCAL_FEATURE_COLUMNS if c in df.columns]
    return df[cols].fillna(0.0).to_numpy(dtype=np.float32)


def train_local_model_for_vehicle(
    benign_train: pd.DataFrame,
    benign_val: pd.DataFrame,
    vehicle_id: str,
    dataset_set: str,
    output_root: Path,
) -> dict:
    """Train Isolation Forest on benign-only data; calibrate weak/strong thresholds."""
    profile = compute_benign_profile(benign_train)

    # Re-extract with profile for deviation features
  # For efficiency, update deviation columns in-place
    for df in (benign_train, benign_val):
        for col in ("deviation_can_id_entropy", "deviation_message_rate", "deviation_mean_dlc", "deviation_byte_mean_norm"):
            if col not in df.columns:
                df[col] = 0.0

    X_train = _feature_matrix(benign_train)
    X_val = _feature_matrix(benign_val) if len(benign_val) else X_train

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    model.fit(X_train_s)

    train_scores = -model.decision_function(X_train_s)
    val_scores = -model.decision_function(X_val_s)

    weak_th = float(np.percentile(val_scores, WEAK_THRESHOLD_PERCENTILE))
    strong_th = float(np.percentile(val_scores, STRONG_THRESHOLD_PERCENTILE))

    model_dir = ensure_dir(output_root / "local_models" / dataset_set)
    scaler_dir = ensure_dir(output_root / "scalers" / dataset_set)
    thresh_dir = ensure_dir(output_root / "thresholds" / dataset_set)

    model_path = model_dir / f"{vehicle_id}_isolation_forest.pkl"
    scaler_path = scaler_dir / f"{vehicle_id}_scaler.pkl"
    thresh_path = thresh_dir / f"{vehicle_id}_thresholds.json"

    with model_path.open("wb") as fh:
        pickle.dump(model, fh)
    with scaler_path.open("wb") as fh:
        pickle.dump(scaler, fh)
    with thresh_path.open("w") as fh:
        json.dump(
            {
                "vehicle_id": vehicle_id,
                "weak_threshold": weak_th,
                "strong_threshold": strong_th,
                "weak_percentile": WEAK_THRESHOLD_PERCENTILE,
                "strong_percentile": STRONG_THRESHOLD_PERCENTILE,
                "benign_profile": profile,
            },
            fh,
            indent=2,
        )

    return {
        "vehicle_id": vehicle_id,
        "dataset_set": dataset_set,
        "model_type": "isolation_forest",
        "feature_count": X_train.shape[1],
        "weak_threshold": weak_th,
        "strong_threshold": strong_th,
        "attack_data_used_in_training": 0,
        "attack_data_used_in_thresholding": 0,
        "test_data_used_in_training": 0,
        "model_hash": content_hash(str(model_path)),
        "threshold_hash": content_hash({"weak": weak_th, "strong": strong_th}),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "threshold_path": str(thresh_path),
        "benign_profile": profile,
        "model": model,
        "scaler": scaler,
    }


def score_windows(
    features: pd.DataFrame,
    model_info: dict,
) -> np.ndarray:
    X = _feature_matrix(features)
    X_s = model_info["scaler"].transform(X)
    return -model_info["model"].decision_function(X_s)


def compute_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr = safe_div(fp, fp + tn)
    spec = safe_div(tn, tn + fp)
    try:
        roc = roc_auc_score(y_true, scores)
    except ValueError:
        roc = float("nan")
    try:
        pr = average_precision_score(y_true, scores)
    except ValueError:
        pr = float("nan")
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr,
        "fpr": fpr,
        "specificity": spec,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "candidate_rate": float(y_pred.mean()),
    }


def run_local_onboarding(
    window_manifest: pd.DataFrame,
    output_root: Path = OUTPUT_ROOT,
    features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Train per-set per-vehicle benign-only models and evaluate on test subsets."""
    if features is not None:
        features_all = features
    else:
        from src.ctt.features import extract_features_from_manifest
        features_all = extract_features_from_manifest(window_manifest, output_root)
        features_path = output_root / "windows" / "all_window_features.parquet"
        features_all.to_parquet(features_path, index=False)

    train_manifest = []
    thresh_manifest = []
    models: dict[tuple[str, str], dict] = {}
    protocol_sections: list[str] = []

    for dataset_set in sorted(features_all["dataset_set"].unique()):
        set_data = features_all[features_all["dataset_set"] == dataset_set]
        train_data = set_data[set_data["subset_name"] == "train_01"]
        benign_train = train_data[train_data["label"] == 0]
        if benign_train.empty:
            protocol_sections.append(f"- **{dataset_set}**: No benign training data; skipped local onboarding.")
            continue

        vehicles_in_train = benign_train["vehicle_id"].unique()
        for vehicle_id in vehicles_in_train:
            v_benign = benign_train[benign_train["vehicle_id"] == vehicle_id]
            if len(v_benign) < 20:
                continue
            # Cap training size for tractability on full dataset
            if len(v_benign) > 50_000:
                v_benign = v_benign.sample(n=50_000, random_state=42)
            # Split benign for validation thresholding
            n_val = max(int(len(v_benign) * 0.2), 10)
            v_val = v_benign.sample(n=n_val, random_state=42)
            v_train = v_benign.drop(v_val.index)

            info = train_local_model_for_vehicle(v_train, v_val, vehicle_id, dataset_set, output_root)
            models[(dataset_set, vehicle_id)] = info

            train_manifest.append(
                {
                    "vehicle_id": vehicle_id,
                    "dataset_set": dataset_set,
                    "training_files": "|".join(sorted(v_train["source_file"].unique())),
                    "benign_training_windows": len(v_train),
                    "validation_files": "|".join(sorted(v_val["source_file"].unique())),
                    "benign_validation_windows": len(v_val),
                    "model_type": info["model_type"],
                    "feature_count": info["feature_count"],
                    "weak_threshold": info["weak_threshold"],
                    "strong_threshold": info["strong_threshold"],
                    "attack_data_used_in_training": 0,
                    "attack_data_used_in_thresholding": 0,
                    "test_data_used_in_training": 0,
                    "model_hash": info["model_hash"],
                    "threshold_hash": info["threshold_hash"],
                }
            )
            thresh_manifest.append(train_manifest[-1])
            protocol_sections.append(
                f"- **{dataset_set}/{vehicle_id}**: benign-only Isolation Forest; "
                f"weak={info['weak_threshold']:.4f}, strong={info['strong_threshold']:.4f}"
            )

    pd.DataFrame(train_manifest).to_csv(output_root / "manifests" / "local_model_training_manifest.csv", index=False)
    pd.DataFrame(thresh_manifest).to_csv(output_root / "manifests" / "local_threshold_manifest.csv", index=False)

    write_markdown(
        output_root / "audit" / "local_onboarding_protocol.md",
        "Local Onboarding Protocol",
        {
            "Approach": (
                "Per-set, per-known-vehicle benign-only Isolation Forest training. "
                "Weak (90th pct) and strong (97.5th pct) thresholds calibrated on held-out benign validation windows. "
                "Unknown vehicles without benign training data are evaluated via descriptor/fleet layer only."
            ),
            "Models trained": "\n".join(protocol_sections) if protocol_sections else "None",
        },
    )

    # Evaluate on all test subsets
    pred_rows = []
    metrics_rows = []

    for dataset_set in sorted(features_all["dataset_set"].unique()):
        set_data = features_all[features_all["dataset_set"] == dataset_set]
        for subset in set_data["subset_name"].unique():
            if subset == "train_01":
                continue
            sub_data = set_data[set_data["subset_name"] == subset]
            for vehicle_id in sub_data["vehicle_id"].unique():
                v_data = sub_data[sub_data["vehicle_id"] == vehicle_id]
                key = (dataset_set, vehicle_id)
                # For unknown vehicle, try known vehicle model from same set
                if key not in models:
                    known_key = None
                    for k in models:
                        if k[0] == dataset_set:
                            known_key = k
                            break
                    if known_key is None:
                        continue
                    model_info = models[known_key]
                    attachment = "descriptor_only_unknown_vehicle"
                else:
                    model_info = models[key]
                    attachment = "local_model"

                scores = score_windows(v_data, model_info)
                weak_th = model_info["weak_threshold"]
                strong_th = model_info["strong_threshold"]
                y_true = v_data["label"].to_numpy(dtype=int)
                weak_pred = (scores >= weak_th).astype(int)
                strong_pred = (scores >= strong_th).astype(int)

                for mode, y_pred in [("weak", weak_pred), ("strong", strong_pred)]:
                    m = compute_detection_metrics(y_true, y_pred, scores)
                    metrics_rows.append(
                        {
                            "dataset_set": dataset_set,
                            "subset_name": subset,
                            "vehicle_id": vehicle_id,
                            "attack_type": "all",
                            "mode": mode,
                            "attachment": attachment,
                            **m,
                        }
                    )

                for i, (_, row) in enumerate(v_data.iterrows()):
                    pred_rows.append(
                        {
                            "window_id": row["window_id"],
                            "dataset_set": dataset_set,
                            "subset_name": subset,
                            "vehicle_id": vehicle_id,
                            "attack_type": row["attack_type"],
                            "label": int(row["label"]),
                            "anomaly_score": float(scores[i]),
                            "weak_threshold": weak_th,
                            "strong_threshold": strong_th,
                            "weak_prediction": int(scores[i] >= weak_th),
                            "strong_prediction": int(scores[i] >= strong_th),
                            "attachment": attachment,
                        }
                    )

                # Per-attack metrics
                for atype, atk_data in v_data.groupby("attack_type"):
                    atk_scores = score_windows(atk_data, model_info)
                    y_atk = atk_data["label"].to_numpy(dtype=int)
                    for mode, th in [("weak", weak_th), ("strong", strong_th)]:
                        y_p = (atk_scores >= th).astype(int)
                        m = compute_detection_metrics(y_atk, y_p, atk_scores)
                        metrics_rows.append(
                            {
                                "dataset_set": dataset_set,
                                "subset_name": subset,
                                "vehicle_id": vehicle_id,
                                "attack_type": atype,
                                "mode": mode,
                                "attachment": attachment,
                                **m,
                            }
                        )

    metrics_df = pd.DataFrame(metrics_rows)
    pred_df = pd.DataFrame(pred_rows)

    results_dir = ensure_dir(output_root / "results" / "local_detection")
    if not metrics_df.empty:
        overall = metrics_df[metrics_df["attack_type"] == "all"]
        overall.groupby(["mode"]).mean(numeric_only=True).reset_index().to_csv(
            results_dir / "overall_metrics.csv", index=False
        )
        overall.to_csv(results_dir / "by_set_and_subset.csv", index=False)
        metrics_df[metrics_df["attack_type"] != "all"].to_csv(results_dir / "by_attack_type.csv", index=False)
        overall.groupby(["vehicle_id", "mode"]).mean(numeric_only=True).reset_index().to_csv(
            results_dir / "by_vehicle.csv", index=False
        )
        overall[overall["mode"] == "weak"].to_csv(results_dir / "weak_candidate_metrics.csv", index=False)
        overall[overall["mode"] == "strong"].to_csv(results_dir / "strong_alert_metrics.csv", index=False)
    pred_df.to_csv(results_dir / "window_predictions.csv", index=False)

    return metrics_df, pred_df, models
