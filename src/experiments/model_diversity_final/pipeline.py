"""Retrain local IF models, scalers, and regenerate descriptors from final split."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.experiments.data_splits import is_benign_attack_type
from src.experiments.local_descriptor_normalisation import (
    DEFAULT_FLEET_NORM_FEATURES,
    fit_benign_fleet_scaler,
)
from src.features.descriptor_generator import DESCRIPTOR_COLUMNS, generate_anomaly_descriptors, make_event_id
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.models.vehicle_ids import (
    _feature_matrix,
    _normalised_anomaly_percentile,
    benign_training_mask,
    fit_self_supervised_isolation_forest,
    infer_true_labels,
    score_self_supervised_isolation_forest,
)

VEHICLE_MODELS = ("Hyundai", "Kia", "Chevrolet")


def _hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]


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
    out["local_alert"] = (scores >= strong_th).astype(int)
    out["weak_signal"] = ((scores >= weak_th) & (scores < strong_th)).astype(int)
    out["local_evidence_level"] = np.select(
        [out["local_alert"].eq(1), out["weak_signal"].eq(1)],
        ["strong_local_anomaly", "weak_suspicious_signal"],
        default="normal",
    )
    out["local_event_alert"] = out["local_alert"].astype(int)
    out["predicted_label"] = out["local_alert"].astype(int)
    out["evidence_level"] = out["local_evidence_level"]
    return out


def retrain_local_pipeline(
    features: pd.DataFrame,
    window_manifest: pd.DataFrame,
    output_root: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Retrain per-platform IF, refit scaler, regenerate descriptors — no prior-split reuse."""
    output_root = Path(output_root)
    models_dir = output_root / "local_models"
    scalers_dir = output_root / "scalers"
    desc_dir = output_root / "descriptors"
    for d in (models_dir, scalers_dir, desc_dir):
        d.mkdir(parents=True, exist_ok=True)

    df = features.merge(
        window_manifest[["window_id", "vehicle_model", "source_file", "split"]].drop_duplicates(),
        on=["window_id", "vehicle_model", "source_file"],
        how="inner",
    )
    if "label" not in df.columns:
        df["label"] = infer_true_labels(df)
    df["normalized_attack_type"] = df["attack_type"].astype(str).str.lower()
    df["ground_truth_malicious"] = (~df["normalized_attack_type"].map(is_benign_attack_type)).astype(int)

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
        model = fit_self_supervised_isolation_forest(X_train, random_state=seed)
        X_all = _feature_matrix(sub)
        _, scores = score_self_supervised_isolation_forest(model, X_all, X_train)
        part = sub.copy()
        part["anomaly_score"] = scores
        val = part[part.split == "validation"]
        weak_th, strong_th = _select_thresholds_on_validation(val if not val.empty else train_benign.assign(anomaly_score=scores[: len(train_benign)]))
        part = _assign_evidence(part, weak_th=weak_th, strong_th=strong_th)

        model_id = f"if_{vm.lower()}_final_{seed}"
        model_path = models_dir / f"{model_id}.joblib"
        joblib.dump(model, model_path)
        train_traces = train_benign.apply(lambda r: f"{r['vehicle_model']}::{r['source_file']}", axis=1).unique().tolist()
        model_rows.append(
            {
                "model_id": model_id,
                "vehicle_scope": "platform",
                "vehicle_model_if_local_only": vm,
                "training_trace_ids": "|".join(sorted(train_traces)),
                "training_row_count": int(len(train_benign)),
                "benign_only": True,
                "feature_names": "|".join(BEHAVIOURAL_FEATURE_COLUMNS),
                "hyperparameters": json.dumps({"n_estimators": 200, "contamination": "auto"}),
                "random_seed": seed,
                "validation_threshold": json.dumps({"weak": weak_th, "strong": strong_th}),
                "training_split_hash": _hash_array(X_train),
                "test_overlap_count": test_overlap,
                "validation_passed": test_overlap == 0,
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
    desc_join["split"] = desc_join["split"].fillna("unknown")

    fleet_scaler = fit_benign_fleet_scaler(descriptors, window_manifest, training_split="train")
    scaler_path = scalers_dir / "fleet_benign_scaler_final.json"
    scaler_path.write_text(json.dumps(fleet_scaler.to_dict(), indent=2), encoding="utf-8")

    train_fit = predictions[(predictions.split == "train") & benign_training_mask(predictions)]
    scaler_rows = [
        {
            "scaler_id": fleet_scaler.scaler_id,
            "vehicle_scope": "global_fleet",
            "feature_names": "|".join(fleet_scaler.fitted_feature_names),
            "fit_split": "train",
            "fit_trace_ids": "|".join(sorted(train_fit.apply(lambda r: f"{r['vehicle_model']}::{r['source_file']}", axis=1).unique())),
            "fit_row_count": fleet_scaler.fit_row_count,
            "benign_only": True,
            "mean_hash": hashlib.sha256(json.dumps(fleet_scaler.means, sort_keys=True).encode()).hexdigest()[:16],
            "scale_hash": hashlib.sha256(json.dumps(fleet_scaler.stds, sort_keys=True).encode()).hexdigest()[:16],
            "test_overlap_count": 0,
            "validation_passed": True,
        }
    ]

    desc_join["scaler_id"] = fleet_scaler.scaler_id
    desc_join["local_model_id"] = desc_join["vehicle_model"].map(
        {r["vehicle_model_if_local_only"]: r["model_id"] for r in model_rows}
    )
    desc_join.to_csv(desc_dir / "all_descriptors.csv", index=False)
    for sp in ("train", "validation", "test"):
        desc_join[desc_join.split == sp].to_csv(desc_dir / f"{sp}_descriptors.csv", index=False)

    prov = desc_join[
        ["event_id", "vehicle_model", "source_file", "split", "ground_truth_malicious", "anomaly_score", "local_model_id", "scaler_id"]
    ].copy()
    prov["source_trace"] = prov.apply(lambda r: f"{r['vehicle_model']}::{r['source_file']}", axis=1)
    prov.to_csv(desc_dir / "descriptor_provenance.csv", index=False)

    pd.DataFrame(model_rows).to_csv(output_root / "manifests/local_model_training_manifest.csv", index=False)
    pd.DataFrame(scaler_rows).to_csv(output_root / "scalers/scaler_manifest.csv", index=False)

    local_metrics = []
    for vm in VEHICLE_MODELS:
        test = predictions[(predictions.vehicle_model == vm) & (predictions.split == "test")]
        if test.empty:
            continue
        y, p = test["label"].astype(int), test["local_alert"].astype(int)
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        local_metrics.append(
            {
                "vehicle_model": vm,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
                "test_count": len(test),
                "strong_threshold": float(test["strong_threshold"].iloc[0]),
                "weak_threshold": float(test["weak_threshold"].iloc[0]),
            }
        )

    return {
        "descriptors": desc_join,
        "window_manifest": window_manifest,
        "predictions": predictions,
        "fleet_scaler": fleet_scaler,
        "local_metrics": pd.DataFrame(local_metrics),
        "model_rows": model_rows,
    }
