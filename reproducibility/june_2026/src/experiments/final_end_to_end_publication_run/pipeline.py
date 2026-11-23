"""Retrain Isolation Forest and regenerate descriptors under the E2E output root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.experiments.data_splits import is_benign_attack_type
from src.experiments.local_descriptor_normalisation import fit_benign_fleet_scaler
from src.features.descriptor_generator import generate_anomaly_descriptors
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.models.vehicle_ids import (
    _feature_matrix,
    benign_training_mask,
    fit_self_supervised_isolation_forest,
    infer_true_labels,
    score_self_supervised_isolation_forest,
)

VEHICLE_MODELS = ("Hyundai", "Kia", "Chevrolet")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _select_thresholds_on_validation(
    val_df: pd.DataFrame,
    *,
    weak_grid: tuple[float, ...] = (0.45, 0.50, 0.55, 0.60),
    strong_grid: tuple[float, ...] = (0.70, 0.75, 0.80, 0.85),
) -> tuple[float, float]:
    y = val_df["label"].astype(int).to_numpy()
    scores = val_df["anomaly_score"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        return 0.55, 0.80
    best_f1, best_w, best_s = -1.0, 0.55, 0.80
    for w in weak_grid:
        for s in strong_grid:
            if w >= s:
                continue
            pred = (scores >= s).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_w, best_s = f1, w, s
    return best_w, best_s


def _assign_evidence(df: pd.DataFrame, *, weak_th: float, strong_th: float) -> pd.DataFrame:
    out = df.copy()
    scores = out["anomaly_score"].astype(float)
    out["local_event_alert"] = (scores >= strong_th).astype(int)
    out["local_alert"] = out["local_event_alert"]
    out["weak_signal"] = ((scores >= weak_th) & (scores < strong_th)).astype(int)
    out["fleet_candidate"] = (scores >= weak_th).astype(int)
    out["local_evidence_level"] = np.select(
        [out["local_event_alert"].eq(1), out["weak_signal"].eq(1)],
        ["strong_local_anomaly", "weak_suspicious_signal"],
        default="normal",
    )
    out["predicted_label"] = out["local_event_alert"].astype(int)
    out["evidence_level"] = out["local_evidence_level"]
    return out


def retrain_local_pipeline(
    features: pd.DataFrame,
    window_manifest: pd.DataFrame,
    output_root: Path,
    *,
    seed: int = 42,
    n_estimators: int = 200,
) -> dict[str, Any]:
    output_root = Path(output_root)
    models_dir = output_root / "models"
    scalers_dir = output_root / "scalers"
    desc_dir = output_root / "descriptors"
    manifests_dir = output_root / "manifests"
    for d in (models_dir, scalers_dir, desc_dir, manifests_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = features.merge(
        window_manifest[["window_id", "vehicle_model", "source_file", "split"]].drop_duplicates(),
        on=["window_id", "vehicle_model", "source_file"],
        how="inner",
    )
    if "label" not in df.columns:
        df["label"] = infer_true_labels(df)
    df["ground_truth_malicious"] = (~df["attack_type"].map(is_benign_attack_type)).astype(int)

    model_rows: list[dict[str, Any]] = []
    pred_parts: list[pd.DataFrame] = []

    for vm in VEHICLE_MODELS:
        sub = df[df.vehicle_model == vm].copy()
        if sub.empty:
            continue
        train_benign = sub[(sub.split == "train") & benign_training_mask(sub)]
        if train_benign.empty:
            raise ValueError(f"No benign train rows for {vm}")
        test_overlap = int(((sub.split == "test") & sub.index.isin(train_benign.index)).sum())
        X_train = _feature_matrix(train_benign)
        model = fit_self_supervised_isolation_forest(X_train, random_state=seed, n_estimators=n_estimators)
        X_all = _feature_matrix(sub)
        _, scores = score_self_supervised_isolation_forest(model, X_all, X_train)
        part = sub.copy()
        part["anomaly_score"] = scores
        val = part[part.split == "validation"]
        weak_th, strong_th = _select_thresholds_on_validation(val if not val.empty else train_benign.assign(anomaly_score=scores[: len(train_benign)]))
        part = _assign_evidence(part, weak_th=weak_th, strong_th=strong_th)

        model_id = f"if_{vm.lower()}_e2e_{seed}"
        model_path = models_dir / f"{model_id}.joblib"
        joblib.dump(model, model_path)
        train_traces = train_benign.apply(lambda r: f"{r['vehicle_model']}::{r['source_file']}", axis=1).unique().tolist()
        model_rows.append(
            {
                "model_id": model_id,
                "vehicle_scope": "platform",
                "vehicle_model": vm,
                "training_trace_ids": "|".join(sorted(train_traces)),
                "training_window_count": int(len(train_benign)),
                "benign_only": True,
                "feature_names": "|".join(BEHAVIOURAL_FEATURE_COLUMNS),
                "hyperparameters": json.dumps({"n_estimators": n_estimators, "contamination": "auto"}),
                "seed": seed,
                "validation_threshold": json.dumps({"weak": weak_th, "strong": strong_th}),
                "test_overlap_count": test_overlap,
                "model_hash": _hash_file(model_path),
            }
        )
        part["local_model_id"] = model_id
        part["weak_threshold"] = weak_th
        part["strong_threshold"] = strong_th
        pred_parts.append(part)

    predictions = pd.concat(pred_parts, ignore_index=True)
    predictions.to_csv(desc_dir / "local_predictions.csv", index=False)
    pred_for_desc = predictions.rename(columns={"label": "true_label"})
    descriptors = generate_anomaly_descriptors(df, pred_for_desc)
    desc_join = descriptors.merge(
        window_manifest[["window_id", "vehicle_model", "source_file", "split"]].drop_duplicates(),
        on=["window_id", "vehicle_model", "source_file"],
        how="left",
    )
    desc_join["ground_truth_malicious"] = (~desc_join["attack_type"].map(is_benign_attack_type)).astype(int)
    pred_cols = predictions[
        ["window_id", "vehicle_model", "source_file", "anomaly_score", "local_evidence_level", "local_event_alert", "fleet_candidate", "local_model_id", "weak_threshold", "strong_threshold"]
    ].drop_duplicates(subset=["window_id", "vehicle_model", "source_file"])
    desc_join = desc_join.drop(columns=[c for c in pred_cols.columns if c in desc_join.columns and c not in ("window_id", "vehicle_model", "source_file")], errors="ignore")
    desc_join = desc_join.merge(pred_cols, on=["window_id", "vehicle_model", "source_file"], how="left")

    fleet_scaler = fit_benign_fleet_scaler(descriptors, window_manifest, training_split="train")
    scaler_path = scalers_dir / "fleet_benign_scaler.json"
    scaler_path.write_text(json.dumps(fleet_scaler.to_dict(), indent=2), encoding="utf-8")
    desc_join["scaler_id"] = fleet_scaler.scaler_id

    desc_join.to_csv(desc_dir / "all_descriptors.csv", index=False)
    for sp in ("train", "validation", "test"):
        desc_join[desc_join.split == sp].to_csv(desc_dir / f"{sp}_descriptors.csv", index=False)

    prov = desc_join[
        ["event_id", "vehicle_model", "source_file", "split", "ground_truth_malicious", "anomaly_score", "local_model_id", "scaler_id", "fleet_candidate"]
    ].copy()
    prov["source_trace"] = prov.apply(lambda r: f"{r['vehicle_model']}::{r['source_file']}", axis=1)
    prov.to_csv(desc_dir / "descriptor_provenance.csv", index=False)

    pd.DataFrame(model_rows).to_csv(manifests_dir / "local_model_training_manifest.csv", index=False)
    pd.DataFrame(
        [{
            "scaler_id": fleet_scaler.scaler_id,
            "vehicle_scope": "global_fleet",
            "feature_names": "|".join(fleet_scaler.fitted_feature_names),
            "fit_split": "train",
            "fit_row_count": fleet_scaler.fit_row_count,
            "benign_only": True,
            "test_overlap_count": 0,
        }]
    ).to_csv(scalers_dir / "scaler_manifest.csv", index=False)

    config = {"local_ids": {"weak_threshold": 0.55, "strong_threshold": 0.80}}
    for row in model_rows:
        th = json.loads(row["validation_threshold"])
        config["local_ids"]["weak_threshold"] = th["weak"]
        config["local_ids"]["strong_threshold"] = th["strong"]
        break

    return {
        "descriptors": desc_join,
        "window_manifest": window_manifest,
        "predictions": predictions,
        "fleet_scaler": fleet_scaler,
        "model_rows": model_rows,
        "local_ids_config": config["local_ids"],
    }
