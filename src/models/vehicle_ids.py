"""Vehicle-level intrusion detection models (per OEM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import compute_confusion_matrix, compute_metrics
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.utils.logging import get_logger

logger = get_logger(__name__)

TASK_BINARY = "binary_classification"
TASK_ANOMALY = "anomaly_detection"
SELF_SUPERVISED_IDS_MODEL = "isolation_forest"

VEHICLE_MODELS = ("Hyundai", "Kia", "Chevrolet")
NORMAL_ATTACK_TYPES = {"normal", "attack_free", "benign", "none", "no_attack"}


@dataclass
class VehicleSplit:
    """Train/test matrices for one vehicle."""

    vehicle_model: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_benign: np.ndarray


@dataclass
class ModelResult:
    """Evaluation outcome for one vehicle × model."""

    vehicle_model: str
    model: str
    task: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    tn: int
    fp: int
    fn: int
    tp: int
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_model": self.vehicle_model,
            "model": self.model,
            "task": self.task,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
            "confusion_matrix": str(self.confusion_matrix),
        }


def load_feature_dataset(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "vehicle_model" not in df.columns:
        raise ValueError("Feature dataset must include a 'vehicle_model' column.")
    if "label" not in df.columns:
        df["label"] = infer_true_labels(df)
    df = df.dropna(subset=["label", "vehicle_model"])
    df["label"] = df["label"].astype(int)
    return df


def infer_true_labels(df: pd.DataFrame) -> pd.Series:
    """Infer labels for evaluation only; labels are not used for IDS training."""
    if "label" in df.columns:
        return pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if "raw_label" in df.columns:
        return df["raw_label"].astype(str).str.upper().ne("R").astype(int)
    if "attack_type" in df.columns:
        return ~df["attack_type"].astype(str).str.lower().isin(NORMAL_ATTACK_TYPES)
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index)


def benign_training_mask(df: pd.DataFrame) -> pd.Series:
    """
    Select benign windows for self-supervised IDS training.

    The proposed vehicle-level IDS is self-supervised and trained only on benign
    CAN windows. Attack labels are used only for evaluation, not training.
    """
    mask = pd.Series(False, index=df.index)
    if "label" in df.columns:
        mask |= pd.to_numeric(df["label"], errors="coerce").fillna(1).astype(int).eq(0)
    if "attack_type" in df.columns:
        mask |= df["attack_type"].astype(str).str.lower().isin(NORMAL_ATTACK_TYPES)
    if "raw_label" in df.columns:
        mask |= df["raw_label"].astype(str).str.upper().eq("R")
    return mask


def prepare_vehicle_split(
    df: pd.DataFrame,
    vehicle_model: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> VehicleSplit:
    """Stratified train/test split for a single vehicle."""
    subset = df[df["vehicle_model"] == vehicle_model].copy()
    if subset.empty:
        raise ValueError(f"No rows for vehicle_model={vehicle_model!r}")

    X = subset[BEHAVIOURAL_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    y = subset["label"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_train_benign = X_train[y_train == 0]
    if X_train_benign.size == 0:
        raise ValueError(f"No benign training samples for {vehicle_model}")

    return VehicleSplit(
        vehicle_model=vehicle_model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_benign=X_train_benign,
    )


def _result_from_predictions(
    vehicle_model: str,
    model_name: str,
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None,
) -> ModelResult:
    metrics = compute_metrics(y_true, y_pred, y_score=y_score)
    cm = compute_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    return ModelResult(
        vehicle_model=vehicle_model,
        model=model_name,
        task=task,
        accuracy=metrics.get("accuracy", float("nan")),
        precision=metrics.get("precision", float("nan")),
        recall=metrics.get("recall", float("nan")),
        f1=metrics.get("f1", float("nan")),
        roc_auc=metrics.get("roc_auc", float("nan")),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        confusion_matrix=cm.tolist(),
    )


def _train_test_indices(
    y: np.ndarray,
    *,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=y
    )
    return train_idx, test_idx


def predict_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_all: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe.predict(X_all), pipe.predict_proba(X_all)[:, 1]


def train_logistic_regression(split: VehicleSplit, *, random_state: int = 42) -> ModelResult:
    """Supervised binary classifier with scaling."""
    y_pred, y_score = predict_logistic_regression(
        split.X_train, split.y_train, split.X_test, random_state=random_state
    )
    return _result_from_predictions(
        split.vehicle_model, "logistic_regression", TASK_BINARY, split.y_test, y_pred, y_score
    )


def _anomaly_threshold_from_benign(train_scores: np.ndarray, percentile: float = 95.0) -> float:
    return float(np.percentile(train_scores, percentile))


def _normalised_anomaly_percentile(
    raw_scores: np.ndarray,
    benign_raw_scores: np.ndarray,
) -> np.ndarray:
    """
    Convert Isolation Forest raw anomaly scores into [0, 1].

    0 means more normal-like than benign training windows; 1 means more anomalous
    than the benign reference distribution. The reference distribution uses only
    benign windows, preserving the self-supervised IDS methodology.
    """
    reference = np.sort(np.asarray(benign_raw_scores, dtype=np.float64))
    if reference.size == 0:
        return np.zeros_like(raw_scores, dtype=np.float64)
    ranks = np.searchsorted(reference, raw_scores, side="right")
    return np.clip(ranks / reference.size, 0.0, 1.0)


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[BEHAVIOURAL_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)


def fit_self_supervised_isolation_forest(
    X_benign: np.ndarray,
    *,
    random_state: int = 42,
    n_estimators: int = 200,
) -> IsolationForest:
    """Fit the proposed benign-only vehicle-level IDS."""
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_benign)
    return model


def score_self_supervised_isolation_forest(
    model: IsolationForest,
    X_all: np.ndarray,
    X_benign_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return local anomaly labels placeholder and normalised anomaly scores."""
    benign_raw_scores = -model.decision_function(X_benign_reference)
    raw_scores = -model.decision_function(X_all)
    anomaly_scores = _normalised_anomaly_percentile(raw_scores, benign_raw_scores)
    return np.zeros(len(X_all), dtype=int), anomaly_scores.astype(float)


def predict_isolation_forest(
    X_train_benign: np.ndarray,
    X_all: np.ndarray,
    *,
    random_state: int = 42,
    threshold_percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    clf = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_benign)
    train_scores = -clf.decision_function(X_train_benign)
    threshold = _anomaly_threshold_from_benign(train_scores, threshold_percentile)
    test_scores = -clf.decision_function(X_all)
    return (test_scores > threshold).astype(int), test_scores


def train_isolation_forest(
    split: VehicleSplit,
    *,
    random_state: int = 42,
    threshold_percentile: float = 95.0,
) -> ModelResult:
    """Unsupervised anomaly detector trained on benign traffic only."""
    y_pred, test_scores = predict_isolation_forest(
        split.X_train_benign,
        split.X_test,
        random_state=random_state,
        threshold_percentile=threshold_percentile,
    )
    return _result_from_predictions(
        split.vehicle_model,
        "isolation_forest",
        TASK_ANOMALY,
        split.y_test,
        y_pred,
        test_scores,
    )


def predict_autoencoder(
    X_train_benign: np.ndarray,
    X_all: np.ndarray,
    *,
    random_state: int = 42,
    threshold_percentile: float = 95.0,
    epochs: int = 30,
    batch_size: int = 256,
    latent_dim: int = 8,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruction-error anomaly detector (PyTorch) trained on benign windows."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required for the autoencoder model.") from exc

    torch.manual_seed(random_state)
    n_features = X_train_benign.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class _AE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_features),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.decoder(self.encoder(x))

    model = _AE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    benign_t = torch.tensor(X_train_benign, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(benign_t, benign_t),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for xb, _ in loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            optimizer.step()

    def reconstruction_errors(X: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32, device=device)
            recon = model(xt)
            err = torch.mean((xt - recon) ** 2, dim=1).cpu().numpy()
        return err

    train_err = reconstruction_errors(X_train_benign)
    threshold = _anomaly_threshold_from_benign(train_err, threshold_percentile)
    test_err = reconstruction_errors(X_all)
    return (test_err > threshold).astype(int), test_err


def train_autoencoder(
    split: VehicleSplit,
    *,
    random_state: int = 42,
    threshold_percentile: float = 95.0,
    epochs: int = 30,
    batch_size: int = 256,
    latent_dim: int = 8,
    hidden_dim: int = 32,
    learning_rate: float = 1e-3,
) -> ModelResult:
    y_pred, test_err = predict_autoencoder(
        split.X_train_benign,
        split.X_test,
        random_state=random_state,
        threshold_percentile=threshold_percentile,
        epochs=epochs,
        batch_size=batch_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
    )
    return _result_from_predictions(
        split.vehicle_model,
        "autoencoder",
        TASK_ANOMALY,
        split.y_test,
        y_pred,
        test_err,
    )


MODEL_TRAINERS: dict[str, tuple[str, Callable[..., ModelResult]]] = {
    "logistic_regression": (TASK_BINARY, train_logistic_regression),
    "isolation_forest": (TASK_ANOMALY, train_isolation_forest),
    "autoencoder": (TASK_ANOMALY, train_autoencoder),
}


def train_all_models_for_vehicle(
    split: VehicleSplit,
    *,
    random_state: int = 42,
    include_autoencoder: bool = True,
) -> list[ModelResult]:
    """Train and evaluate all vehicle-level models."""
    results: list[ModelResult] = []
    for name, (_, trainer) in MODEL_TRAINERS.items():
        if name == "autoencoder" and not include_autoencoder:
            continue
        logger.info("Training %s for %s", name, split.vehicle_model)
        try:
            result = trainer(split, random_state=random_state)
        except ImportError as exc:
            logger.warning("Skipping %s: %s", name, exc)
            continue
        results.append(result)
        logger.info(
            "%s | %s | F1=%.4f AUC=%.4f",
            split.vehicle_model,
            name,
            result.f1,
            result.roc_auc,
        )
    return results


def generate_window_predictions(
    df: pd.DataFrame,
    *,
    vehicles: tuple[str, ...] = VEHICLE_MODELS,
    test_size: float = 0.2,
    random_state: int = 42,
    include_autoencoder: bool = True,
) -> pd.DataFrame:
    """
    Fit IDS models per vehicle (train split) and predict on **all** windows.

    Returns long-format predictions: window_id, vehicle metadata, model,
    predicted_label, anomaly_score.
    """
    rows: list[pd.DataFrame] = []
    model_fns: list[tuple[str, Callable[..., tuple[np.ndarray, np.ndarray]]]] = [
        ("logistic_regression", predict_logistic_regression),
        ("isolation_forest", predict_isolation_forest),
    ]
    if include_autoencoder:
        model_fns.append(("autoencoder", predict_autoencoder))

    for vehicle in vehicles:
        subset = df[df["vehicle_model"] == vehicle].copy()
        if subset.empty:
            continue

        X = subset[BEHAVIOURAL_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
        y = subset["label"].to_numpy(dtype=np.int64)
        train_idx, _ = _train_test_indices(y, test_size=test_size, random_state=random_state)
        X_train, y_train = X[train_idx], y[train_idx]
        X_train_benign = X_train[y_train == 0]

        meta_cols = [
            "window_id",
            "vehicle_model",
            "source_file",
            "attack_type",
            "label",
        ]
        meta = subset[[c for c in meta_cols if c in subset.columns]].copy()

        for model_name, predict_fn in model_fns:
            logger.info("Predicting %s for %s (%d windows)", model_name, vehicle, len(subset))
            if model_name in ("isolation_forest", "autoencoder"):
                y_pred, y_score = predict_fn(
                    X_train_benign,
                    X,
                    random_state=random_state,
                )
            else:
                y_pred, y_score = predict_fn(X_train, y_train, X, random_state=random_state)

            part = meta.copy()
            part["model"] = model_name
            part["predicted_label"] = y_pred.astype(int)
            part["anomaly_score"] = y_score.astype(float)
            rows.append(part)

    if not rows:
        return pd.DataFrame(
            columns=["window_id", "vehicle_model", "model", "predicted_label", "anomaly_score"]
        )
    return pd.concat(rows, ignore_index=True)


def generate_vehicle_anomaly_predictions(
    features: pd.DataFrame,
    *,
    primary_model: str = SELF_SUPERVISED_IDS_MODEL,
    test_size: float = 0.2,
    random_state: int = 42,
    include_autoencoder: bool = True,
    strong_threshold: float = 0.80,
    weak_threshold: float = 0.55,
    n_estimators: int = 200,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Generate the canonical self-supervised vehicle IDS output.

    The proposed vehicle-level IDS is self-supervised and trained only on benign
    CAN windows. Attack labels are used only for evaluation, not training.
    """
    if weak_threshold >= strong_threshold:
        raise ValueError("weak_threshold must be lower than strong_threshold")
    if primary_model not in {SELF_SUPERVISED_IDS_MODEL, "isolation_forest"}:
        logger.warning(
            "Ignoring primary_model=%s for proposed IDS; using self-supervised Isolation Forest.",
            primary_model,
        )

    df = features.copy()
    if "label" not in df.columns:
        df["label"] = infer_true_labels(df)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "window_id",
                "vehicle_model",
                "source_file",
                "attack_type",
                "true_label",
                "anomaly_score",
                "local_alert",
                "weak_signal",
                "evidence_level",
            ]
        )

    all_predictions: list[pd.DataFrame] = []
    model_bundle: dict[str, Any] = {
        "model_type": SELF_SUPERVISED_IDS_MODEL,
        "training_mode": "self_supervised_benign_only",
        "feature_columns": BEHAVIOURAL_FEATURE_COLUMNS,
        "strong_threshold": strong_threshold,
        "weak_threshold": weak_threshold,
        "models": {},
        "benign_reference_scores": {},
    }

    for vehicle, subset in df.groupby("vehicle_model", sort=True):
        subset = subset.copy()
        benign_subset = subset[benign_training_mask(subset)].copy()
        if benign_subset.empty:
            logger.warning("Skipping %s: no benign windows available for self-supervised training", vehicle)
            continue

        X_benign = _feature_matrix(benign_subset)
        X_all = _feature_matrix(subset)
        model = fit_self_supervised_isolation_forest(
            X_benign,
            random_state=random_state,
            n_estimators=n_estimators,
        )
        _, anomaly_scores = score_self_supervised_isolation_forest(model, X_all, X_benign)

        meta_cols = ["window_id", "vehicle_model", "source_file", "attack_type", "label"]
        part = subset[[c for c in meta_cols if c in subset.columns]].copy()
        part["true_label"] = subset["label"].astype(int)
        part["anomaly_score"] = anomaly_scores
        all_predictions.append(part)

        model_bundle["models"][vehicle] = model
        model_bundle["benign_reference_scores"][vehicle] = -model.decision_function(X_benign)
        logger.info(
            "Trained self-supervised Isolation Forest for %s on %d benign windows; scored %d windows",
            vehicle,
            len(benign_subset),
            len(subset),
        )

    if not all_predictions:
        raise ValueError("No self-supervised Isolation Forest models were trained.")

    out = pd.concat(all_predictions, ignore_index=True)
    out["anomaly_score"] = out["anomaly_score"].fillna(0.0).clip(0.0, 1.0).astype(float)
    out["local_alert"] = (out["anomaly_score"] >= strong_threshold).astype(int)
    out["weak_signal"] = (
        (out["anomaly_score"] >= weak_threshold) & (out["anomaly_score"] < strong_threshold)
    ).astype(int)
    out["evidence_level"] = np.select(
        [out["local_alert"].eq(1), out["weak_signal"].eq(1)],
        ["strong_local_anomaly", "weak_suspicious_signal"],
        default="normal",
    )
    if model_path is not None:
        save_vehicle_ids_model(model_bundle, model_path)
    return out[
        [
            "window_id",
            "vehicle_model",
            "source_file",
            "attack_type",
            "true_label",
            "anomaly_score",
            "local_alert",
            "weak_signal",
            "evidence_level",
        ]
    ]


def evaluate_vehicle_anomaly_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate predictions against true labels after inference only."""
    rows: list[dict[str, Any]] = []
    for vehicle, group in predictions.groupby("vehicle_model", sort=True):
        y_true = group["true_label"].astype(int).to_numpy()
        y_pred = group["local_alert"].astype(int).to_numpy()
        y_score = group["anomaly_score"].astype(float).to_numpy()
        result = _result_from_predictions(
            vehicle,
            SELF_SUPERVISED_IDS_MODEL,
            "self_supervised_anomaly_detection",
            y_true,
            y_pred,
            y_score,
        ).to_dict()
        result["f1_score"] = result["f1"]
        result["training_data"] = "benign_windows_only"
        result["evaluated_windows"] = int(len(group))
        result["local_alerts"] = int(group["local_alert"].sum())
        result["weak_signals"] = int(group["weak_signal"].sum())
        rows.append(result)

    y_true_all = predictions["true_label"].astype(int).to_numpy()
    y_pred_all = predictions["local_alert"].astype(int).to_numpy()
    y_score_all = predictions["anomaly_score"].astype(float).to_numpy()
    total = _result_from_predictions(
        "ALL",
        SELF_SUPERVISED_IDS_MODEL,
        "self_supervised_anomaly_detection",
        y_true_all,
        y_pred_all,
        y_score_all,
    ).to_dict()
    total["f1_score"] = total["f1"]
    total["training_data"] = "benign_windows_only"
    total["evaluated_windows"] = int(len(predictions))
    total["local_alerts"] = int(predictions["local_alert"].sum())
    total["weak_signals"] = int(predictions["weak_signal"].sum())
    rows.append(total)
    return pd.DataFrame(rows)


def save_vehicle_ids_model(model_bundle: dict[str, Any], path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, out)
    logger.info("Saved self-supervised vehicle IDS model to %s", out)
    return out


def save_vehicle_anomaly_predictions(predictions: pd.DataFrame, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    logger.info("Saved vehicle anomaly predictions to %s", out)
    return out


def save_window_predictions(predictions: pd.DataFrame, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out, index=False)
    logger.info("Saved window predictions to %s", out)
    return out


def load_window_predictions(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def run_vehicle_level_training(
    features_path: Path | str,
    *,
    vehicles: tuple[str, ...] = VEHICLE_MODELS,
    test_size: float = 0.2,
    random_state: int = 42,
    include_autoencoder: bool = True,
    strong_threshold: float = 0.80,
    weak_threshold: float = 0.55,
    model_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Train/evaluate the proposed self-supervised vehicle IDS.

    The Isolation Forest is trained only on benign CAN windows. Ground-truth
    attack labels are used only after inference to calculate evaluation metrics.
    """
    df = load_feature_dataset(features_path)
    if vehicles:
        present = set(df["vehicle_model"].dropna().unique())
        selected = [v for v in vehicles if v in present]
        missing = sorted(set(vehicles) - present)
        for vehicle in missing:
            logger.warning("Skipping missing vehicle: %s", vehicle)
        if selected:
            df = df[df["vehicle_model"].isin(selected)].copy()
    predictions = generate_vehicle_anomaly_predictions(
        df,
        primary_model=SELF_SUPERVISED_IDS_MODEL,
        test_size=test_size,
        random_state=random_state,
        include_autoencoder=include_autoencoder,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
        model_path=model_path,
    )
    return evaluate_vehicle_anomaly_predictions(predictions)

def save_results(results: pd.DataFrame, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    logger.info("Saved vehicle-level results to %s", out)
    return out


def plot_vehicle_confusion_matrices(
    results: pd.DataFrame,
    output_path: Path | str,
    *,
    figsize: tuple[float, float] = (16, 10),
) -> Path:
    """
    Plot confusion matrices in a grid: rows = vehicles, columns = models.
    """
    import ast

    vehicles = sorted(results["vehicle_model"].unique())
    models = [
        m
        for m in ["logistic_regression", "isolation_forest", "autoencoder"]
        if m in results["model"].values
    ]
    if not models:
        logger.warning("No supported model columns available for confusion matrix plot.")
        return Path(output_path)

    nrows, ncols = len(vehicles), len(models)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, vehicle in enumerate(vehicles):
        for j, model in enumerate(models):
            ax = axes[i, j]
            row = results[(results["vehicle_model"] == vehicle) & (results["model"] == model)]
            if row.empty:
                ax.axis("off")
                ax.set_title(f"{vehicle}\n{model}\n(no data)")
                continue
            cm = np.array(ast.literal_eval(row.iloc[0]["confusion_matrix"]))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            f1 = row.iloc[0]["f1"]
            ax.set_title(f"{vehicle} | {model}\nF1={f1:.3f}", fontsize=9)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")

    fig.suptitle("Vehicle-level intrusion detection — confusion matrices", fontsize=13)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote confusion matrix figure to %s", out)
    return out


def print_results_summary(results: pd.DataFrame) -> None:
    """Print PASS-style summary per model."""
    print("\n=== Vehicle-Level IDS Results ===")
    for _, row in results.sort_values(["vehicle_model", "model"]).iterrows():
        print(
            f"{row['vehicle_model']:10} | {row['model']:22} | {row['task']:22} | "
            f"Acc={row['accuracy']:.4f} P={row['precision']:.4f} "
            f"R={row['recall']:.4f} F1={row['f1']:.4f} AUC={row['roc_auc']:.4f}"
        )
    print("=================================\n")
