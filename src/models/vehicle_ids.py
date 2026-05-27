"""Vehicle-level intrusion detection models (per OEM)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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

VEHICLE_MODELS = ("Hyundai", "Kia", "Chevrolet")


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
    if "label" not in df.columns:
        raise ValueError("Feature dataset must include a 'label' column.")
    df = df.dropna(subset=["label", "vehicle_model"])
    df["label"] = df["label"].astype(int)
    return df


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


def predict_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_all: np.ndarray,
    *,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_all), clf.predict_proba(X_all)[:, 1]


def train_random_forest(split: VehicleSplit, *, random_state: int = 42) -> ModelResult:
    """Supervised binary classifier."""
    y_pred, y_score = predict_random_forest(
        split.X_train, split.y_train, split.X_test, random_state=random_state
    )
    return _result_from_predictions(
        split.vehicle_model, "random_forest", TASK_BINARY, split.y_test, y_pred, y_score
    )


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
    "random_forest": (TASK_BINARY, train_random_forest),
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
        ("random_forest", predict_random_forest),
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
    primary_model: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    include_autoencoder: bool = True,
) -> pd.DataFrame:
    """
    Generate the canonical vehicle IDS output used by downstream fleet stages.

    ``predicted_label`` is an ensemble anomaly flag: a window is anomalous if
    any configured vehicle IDS model flags it. ``anomaly_score`` is reported
    from the primary model, usually Random Forest attack probability.
    """
    long_preds = generate_window_predictions(
        features,
        test_size=test_size,
        random_state=random_state,
        include_autoencoder=include_autoencoder,
    )
    if long_preds.empty:
        return pd.DataFrame(
            columns=[
                "window_id",
                "vehicle_model",
                "source_file",
                "attack_type",
                "true_label",
                "predicted_label",
                "anomaly_score",
                "is_anomaly",
            ]
        )

    primary = long_preds[long_preds["model"] == primary_model].copy()
    if primary.empty:
        logger.warning("Primary model %s missing; using max anomaly score.", primary_model)
        primary = (
            long_preds.groupby(["window_id", "vehicle_model"], as_index=False)
            .agg(anomaly_score=("anomaly_score", "max"))
        )
    else:
        primary = primary.drop(columns=["model"], errors="ignore")

    flags = (
        long_preds.groupby(["window_id", "vehicle_model"], as_index=False)["predicted_label"]
        .max()
        .rename(columns={"predicted_label": "is_anomaly"})
    )
    meta_cols = ["window_id", "vehicle_model", "source_file", "attack_type", "label"]
    meta = features[[c for c in meta_cols if c in features.columns]].drop_duplicates(
        subset=["window_id", "vehicle_model"]
    )
    out = meta.merge(
        primary[["window_id", "vehicle_model", "anomaly_score"]],
        on=["window_id", "vehicle_model"],
        how="left",
    ).merge(flags, on=["window_id", "vehicle_model"], how="left")
    out["true_label"] = out["label"].astype(int)
    out["predicted_label"] = out["is_anomaly"].fillna(0).astype(int)
    out["is_anomaly"] = out["predicted_label"].astype(int)
    out["anomaly_score"] = out["anomaly_score"].fillna(0.0).astype(float)
    return out[
        [
            "window_id",
            "vehicle_model",
            "source_file",
            "attack_type",
            "true_label",
            "predicted_label",
            "anomaly_score",
            "is_anomaly",
        ]
    ]


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
) -> pd.DataFrame:
    """Train all models for each vehicle and return results table."""
    df = load_feature_dataset(features_path)
    all_results: list[ModelResult] = []

    for vehicle in vehicles:
        if vehicle not in df["vehicle_model"].unique():
            logger.warning("Skipping missing vehicle: %s", vehicle)
            continue
        split = prepare_vehicle_split(
            df, vehicle, test_size=test_size, random_state=random_state
        )
        logger.info(
            "%s: train=%d (benign=%d) test=%d",
            vehicle,
            len(split.y_train),
            len(split.X_train_benign),
            len(split.y_test),
        )
        all_results.extend(
            train_all_models_for_vehicle(
                split, random_state=random_state, include_autoencoder=include_autoencoder
            )
        )

    return pd.DataFrame([r.to_dict() for r in all_results])


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
        for m in ["random_forest", "logistic_regression", "isolation_forest", "autoencoder"]
        if m in results["model"].values
    ]

    nrows, ncols = len(vehicles), len(models)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

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
