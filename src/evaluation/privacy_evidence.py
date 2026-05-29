"""Privacy evaluation evidence for descriptor transmission.

Strongest privacy proof provided here:

1) Vehicle re-identification (linkability) attack:
   - Attacker tries to predict `vehicle_model` from:
     a) full local descriptors (higher leakage baseline)
     b) transmitted privacy-preserving descriptors (should be much lower)

2) Reconstruction risk:
   - Attacker tries to reconstruct sensitive CAN-ID-structure attributes that are
     removed from the transmitted descriptor payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

from src.features.descriptor_generator import build_transmitted_descriptors
from src.utils.logging import get_logger

logger = get_logger(__name__)


SENSITIVE_CAN_STRUCTURE_COLUMNS = [
    "unique_can_id_count",
    "can_id_entropy",
    "most_common_can_id_ratio",
]


@dataclass(frozen=True)
class PrivacyPaths:
    metrics_reid: Path
    metrics_reconstruction: Path
    fig_reid_full: Path
    fig_reid_transmit: Path
    fig_reid_comparison: Path
    fig_reconstruction: Path


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def _train_vehicle_reid(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[Pipeline, dict[str, Any], np.ndarray]:
    """Train a simple attacker model; return model, metrics, confusion matrix."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    num_cols = _numeric_columns(X)
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    macro_f1 = float(f1_score(y_test, pred, average="macro"))
    cm = confusion_matrix(y_test, pred, labels=np.unique(y))
    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_classes": int(len(np.unique(y))),
        "accuracy": acc,
        "macro_f1": macro_f1,
    }
    return pipe, metrics, cm


def _plot_confusion(cm: np.ndarray, labels: list[str], out: Path, title: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_reid_comparison(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    df_plot = df.melt(id_vars=["feature_set"], value_vars=["accuracy", "macro_f1"])
    cats = df_plot["feature_set"].unique().tolist()
    metrics = df_plot["variable"].unique().tolist()
    x = np.arange(len(cats))
    width = 0.35 if len(metrics) <= 2 else 0.8 / max(1, len(metrics))
    for i, m in enumerate(metrics):
        vals = (
            df_plot[df_plot["variable"] == m]
            .set_index("feature_set")
            .reindex(cats)["value"]
            .to_numpy()
        )
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, vals, width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Vehicle re-identification leakage (lower is better)")
    ax.set_ylabel("Score")
    ax.set_xlabel("Observed payload")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _reconstruction_risk(
    X_tx: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X_tx, y.to_numpy(), test_size=0.25, random_state=seed
    )
    num_cols = _numeric_columns(X_tx)
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=True, with_std=True), num_cols)],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=seed,
        n_jobs=-1,
    )
    pipe = Pipeline([("pre", pre), ("rf", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    return {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }


def _plot_reconstruction_table(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(df))))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.round(6).values,
        colLabels=df.columns.tolist(),
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.3)
    ax.set_title("Reconstruction risk from transmitted descriptors (lower is better)")
    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def evaluate_privacy_evidence(
    *,
    descriptors_path: Path | str,
    metrics_dir: Path | str,
    figures_dir: Path | str,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Produce privacy evidence package from the local descriptor table.

    Inputs:
      - `anomaly_descriptors.csv` (local evaluation table; includes `vehicle_model`
        and CAN-ID-structure columns)

    Outputs:
      - CSV metrics + PNG figures under metrics_dir/figures_dir
    """
    desc = pd.read_csv(descriptors_path)
    if desc.empty:
        raise ValueError("Descriptors are empty; cannot evaluate privacy evidence.")
    if "vehicle_model" not in desc.columns:
        raise ValueError("Descriptors missing `vehicle_model` for re-identification evaluation.")

    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = PrivacyPaths(
        metrics_reid=metrics_dir / "privacy_vehicle_reidentification.csv",
        metrics_reconstruction=metrics_dir / "privacy_reconstruction_risk.csv",
        fig_reid_full=figures_dir / "privacy_reid_confusion_full_descriptor.png",
        fig_reid_transmit=figures_dir / "privacy_reid_confusion_transmitted_descriptor.png",
        fig_reid_comparison=figures_dir / "privacy_reid_leakage_comparison.png",
        fig_reconstruction=figures_dir / "privacy_reconstruction_risk.png",
    )

    # === 1) Vehicle re-identification (linkability) attack ===
    le = LabelEncoder()
    y = le.fit_transform(desc["vehicle_model"].astype(str))
    labels = [str(x) for x in le.classes_]

    # Full descriptor baseline features (exclude direct identity columns if present)
    drop_full = {"event_id", "window_id", "vehicle_model", "source_file", "attack_type", "ground_truth_label"}
    X_full = desc.drop(columns=[c for c in drop_full if c in desc.columns], errors="ignore")
    X_full = X_full.select_dtypes(include=["number"]).fillna(0.0)

    # Transmitted descriptor features (privacy-preserving view)
    tx = build_transmitted_descriptors(desc, quantize_decimals=3)
    X_tx = tx.select_dtypes(include=["number"]).fillna(0.0)

    rows: list[dict[str, Any]] = []

    _, m_full, cm_full = _train_vehicle_reid(X_full, y, seed=seed)
    rows.append({"feature_set": "full_descriptor", **m_full})
    _plot_confusion(cm_full, labels, paths.fig_reid_full, "Vehicle re-identification (full descriptor)")

    _, m_tx, cm_tx = _train_vehicle_reid(X_tx, y, seed=seed)
    rows.append({"feature_set": "transmitted_descriptor", **m_tx})
    _plot_confusion(cm_tx, labels, paths.fig_reid_transmit, "Vehicle re-identification (transmitted descriptor)")

    reid_df = pd.DataFrame(rows)
    reid_df.to_csv(paths.metrics_reid, index=False)
    _plot_reid_comparison(rows, paths.fig_reid_comparison)

    # === 2) Reconstruction risk: recover removed CAN-ID-structure columns from X_tx ===
    recon_rows: list[dict[str, Any]] = []
    for col in SENSITIVE_CAN_STRUCTURE_COLUMNS:
        if col not in desc.columns:
            continue
        y_target = pd.to_numeric(desc[col], errors="coerce").fillna(0.0)
        metrics = _reconstruction_risk(X_tx, y_target, seed=seed)
        recon_rows.append({"target": col, **metrics})

    recon_df = pd.DataFrame(recon_rows)
    recon_df.to_csv(paths.metrics_reconstruction, index=False)
    if not recon_df.empty:
        _plot_reconstruction_table(recon_df[["target", "r2", "mae"]], paths.fig_reconstruction)
    else:
        logger.warning("No sensitive CAN structure columns found; reconstruction risk skipped.")

    logger.info("Saved privacy re-identification metrics to %s", paths.metrics_reid)
    logger.info("Saved privacy reconstruction metrics to %s", paths.metrics_reconstruction)

    return {
        "privacy_vehicle_reidentification": paths.metrics_reid,
        "privacy_reconstruction_risk": paths.metrics_reconstruction,
        "privacy_reid_confusion_full": paths.fig_reid_full,
        "privacy_reid_confusion_transmit": paths.fig_reid_transmit,
        "privacy_reid_leakage_comparison": paths.fig_reid_comparison,
        "privacy_reconstruction_figure": paths.fig_reconstruction,
    }

