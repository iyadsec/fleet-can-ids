"""Vehicle-level self-supervised IDS evaluation for paper Table 1 and Figures 1–3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.models.vehicle_ids import (
    SELF_SUPERVISED_IDS_MODEL,
    benign_training_mask,
    fit_self_supervised_isolation_forest,
    infer_true_labels,
    load_feature_dataset,
    score_self_supervised_isolation_forest,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

IEEE_RC = {
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


@dataclass(frozen=True)
class EvaluationOutputs:
    results_dir: Path
    tables_dir: Path
    figures_dir: Path


@dataclass
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _stratified_train_val_test_indices(
    y: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> SplitIndices:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Splits must sum to 1.0, got {total}")
    idx = np.arange(len(y))
    train_idx, temp_idx = train_test_split(
        idx,
        test_size=(1.0 - train_ratio),
        random_state=random_state,
        stratify=y,
    )
    val_share = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1.0 - val_share),
        random_state=random_state,
        stratify=y[temp_idx],
    )
    return SplitIndices(train=train_idx, val=val_idx, test=test_idx)


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[list(BEHAVIOURAL_FEATURE_COLUMNS)].fillna(0.0).to_numpy(dtype=np.float32)


def _select_threshold_max_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick threshold on validation data that maximizes binary F1."""
    if len(np.unique(y_true)) < 2:
        return float(np.percentile(scores, 95))
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199))):
        pred = (scores >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    roc_auc = float("nan")
    pr_auc = float("nan")
    if len(np.unique(y_true)) >= 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            pass
        try:
            pr_auc = float(average_precision_score(y_true, y_score))
        except ValueError:
            pass
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "recall_pct": recall * 100.0,
        "precision_pct": precision * 100.0,
        "f1_pct": f1 * 100.0,
        "false_positive_rate_pct": fpr * 100.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _window_duration_ms(row: pd.Series) -> float:
    """Proxy window duration from inter-arrival timing (ms)."""
    iat = float(row.get("mean_inter_arrival_time", 0.0) or 0.0)
    frames = float(row.get("frame_count", row.get("n_frames", 1)) or 1)
    return max(iat * frames * 1000.0, 0.0)


def _detection_latency_ms(sub: pd.DataFrame, threshold: float) -> float:
    """
    Mean delay from the start of each attack segment to the first alerted window.

    Uses window order within each source_file; latency sums window durations until alert.
    """
    if sub.empty:
        return float("nan")
    sub = sub.sort_values(["source_file", "start_frame_idx"], na_position="last")
    latencies: list[float] = []
    for _, group in sub.groupby("source_file", sort=False):
        g = group.reset_index(drop=True)
        attack_run = False
        elapsed_ms = 0.0
        for _, row in g.iterrows():
            dur = _window_duration_ms(row)
            is_attack = int(row.get("label", row.get("true_label", 0))) == 1
            if is_attack:
                if not attack_run:
                    attack_run = True
                    elapsed_ms = 0.0
                elapsed_ms += dur
                if float(row["anomaly_score"]) >= threshold:
                    latencies.append(elapsed_ms)
                    attack_run = False
                    elapsed_ms = 0.0
            else:
                attack_run = False
                elapsed_ms = 0.0
    return float(np.mean(latencies)) if latencies else float("nan")


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _df_to_ieee_tex(df: pd.DataFrame, caption: str, label: str) -> str:
    body = df.to_latex(index=False, escape=True, float_format="%.4f")
    return "\n".join(
        [
            "\\begin{table}[t]",
            "\\centering",
            body.strip(),
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table}",
            "",
        ]
    )


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def run_vehicle_level_evaluation(
    features: pd.DataFrame,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    n_estimators: int = 200,
    outputs: EvaluationOutputs,
) -> dict[str, Path]:
    """
    Train Isolation Forest on benign train windows; tune threshold on val; report on test.
    """
    plt.rcParams.update(IEEE_RC)
    df = features.copy()
    if "label" not in df.columns:
        df["label"] = infer_true_labels(df)

    if df["label"].nunique() < 2:
        raise ValueError(
            "Labels are missing or single-class; cannot compute supervised IDS metrics."
        )

    if "start_frame_idx" not in df.columns:
        meta_path = outputs.results_dir.parent / "data/processed/window_metadata.csv"
        if meta_path.exists():
            meta = pd.read_csv(
                meta_path,
                usecols=["window_id", "vehicle_model", "start_frame_idx", "end_frame_idx", "n_frames"],
            )
            df = df.merge(meta, on=["window_id", "vehicle_model"], how="left", suffixes=("", "_meta"))
            if "n_frames" in df.columns and "frame_count" not in df.columns:
                df["frame_count"] = df["n_frames"]

    thresholds: dict[str, float] = {}
    test_frames: list[pd.DataFrame] = []

    for vehicle, subset in df.groupby("vehicle_model", sort=True):
        subset = subset.reset_index(drop=True)
        y = subset["label"].to_numpy(dtype=np.int64)
        splits = _stratified_train_val_test_indices(
            y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=random_state,
        )

        train_df = subset.iloc[splits.train]
        val_df = subset.iloc[splits.val]
        test_df = subset.iloc[splits.test]

        benign_train = train_df[benign_training_mask(train_df)]
        if benign_train.empty:
            logger.warning("Skipping %s: no benign training windows", vehicle)
            continue

        X_benign = _feature_matrix(benign_train)
        model = fit_self_supervised_isolation_forest(
            X_benign, random_state=random_state, n_estimators=n_estimators
        )

        def score_part(part: pd.DataFrame) -> np.ndarray:
            _, scores = score_self_supervised_isolation_forest(
                model, _feature_matrix(part), X_benign
            )
            return scores

        val_scores = score_part(val_df)
        threshold = _select_threshold_max_f1(val_df["label"].to_numpy(), val_scores)
        thresholds[vehicle] = threshold

        test_part = test_df.copy()
        test_part["anomaly_score"] = score_part(test_df)
        test_part["predicted_label"] = (test_part["anomaly_score"] >= threshold).astype(int)
        test_part["vehicle_model"] = vehicle
        test_part["threshold"] = threshold
        test_frames.append(test_part)
        logger.info(
            "%s: threshold=%.4f (val F1 tuning), test windows=%d",
            vehicle,
            threshold,
            len(test_part),
        )

    if not test_frames:
        raise ValueError("No vehicle models evaluated.")

    test_all = pd.concat(test_frames, ignore_index=True)
    y_test = test_all["label"].to_numpy(dtype=np.int64)
    y_pred = test_all["predicted_label"].to_numpy(dtype=np.int64)
    y_score = test_all["anomaly_score"].to_numpy(dtype=np.float64)

    metrics = _binary_metrics(y_test, y_pred, y_score)
    latencies_per_vehicle: list[float] = []
    for vehicle, part in test_all.groupby("vehicle_model"):
        lat = _detection_latency_ms(part, threshold=thresholds[vehicle])
        if not np.isnan(lat):
            latencies_per_vehicle.append(lat)
    pooled_latency = float(np.mean(latencies_per_vehicle)) if latencies_per_vehicle else float("nan")

    table1 = pd.DataFrame(
        [
            {
                "Model": "Isolation Forest (self-supervised)",
                "ROC-AUC": round(metrics["roc_auc"], 4),
                "PR-AUC": round(metrics["pr_auc"], 4),
                "Recall (%)": round(metrics["recall_pct"], 2),
                "Precision (%)": round(metrics["precision_pct"], 2),
                "F1-Score (%)": round(metrics["f1_pct"], 2),
                "False Positive Rate (%)": round(metrics["false_positive_rate_pct"], 2),
                "Detection Latency (ms)": round(pooled_latency, 2),
            }
        ]
    )

    outputs.results_dir.mkdir(parents=True, exist_ok=True)
    outputs.tables_dir.mkdir(parents=True, exist_ok=True)
    outputs.figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = outputs.results_dir / "vehicle_level_metrics.csv"
    table1.to_csv(metrics_csv, index=False)

    threshold_path = outputs.results_dir / "vehicle_level_thresholds.json"
    with threshold_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": SELF_SUPERVISED_IDS_MODEL,
                "training": "benign_train_only",
                "threshold_selection": "validation_max_f1",
                "per_vehicle_threshold": thresholds,
            },
            fh,
            indent=2,
        )

    by_attack = pd.DataFrame()
    # Optional by attack type (test only)
    if "attack_type" in test_all.columns:
        rows: list[dict[str, Any]] = []
        for attack, grp in test_all.groupby("attack_type", sort=True):
            yt = grp["label"].to_numpy(dtype=np.int64)
            yp = grp["predicted_label"].to_numpy(dtype=np.int64)
            ys = grp["anomaly_score"].to_numpy(dtype=np.float64)
            m = _binary_metrics(yt, yp, ys)
            lat_parts = [
                _detection_latency_ms(vgrp, threshold=thresholds[vehicle])
                for vehicle, vgrp in grp.groupby("vehicle_model")
            ]
            lat_vals = [x for x in lat_parts if not np.isnan(x)]
            lat = float(np.mean(lat_vals)) if lat_vals else float("nan")
            rows.append(
                {
                    "Attack Type": attack,
                    "Recall (%)": round(m["recall_pct"], 2),
                    "Precision (%)": round(m["precision_pct"], 2),
                    "F1-Score (%)": round(m["f1_pct"], 2),
                    "Average Anomaly Score": round(float(np.mean(ys)), 4),
                    "Detection Latency (ms)": round(lat, 2) if not np.isnan(lat) else "",
                }
            )
        by_attack = pd.DataFrame(rows)
        by_attack_csv = outputs.results_dir / "vehicle_level_by_attack_type.csv"
        by_attack.to_csv(by_attack_csv, index=False)
        (outputs.tables_dir / "table_vehicle_level_by_attack_type.tex").write_text(
            _df_to_ieee_tex(
                by_attack,
                "Vehicle-level IDS performance by attack type (test split).",
                "tab:vehicle-ids-by-attack",
            ),
            encoding="utf-8",
        )
        (outputs.tables_dir / "table_vehicle_level_by_attack_type.md").write_text(
            "# Table: Vehicle-Level IDS by Attack Type\n\n" + _df_to_markdown(by_attack),
            encoding="utf-8",
        )

    (outputs.tables_dir / "table_vehicle_level_ids.tex").write_text(
        _df_to_ieee_tex(
            table1,
            "Vehicle-level self-supervised IDS performance on the held-out test split.",
            "tab:vehicle-ids",
        ),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_vehicle_level_ids.md").write_text(
        "# Table 1: Vehicle-Level IDS Performance\n\n" + _df_to_markdown(table1),
        encoding="utf-8",
    )

    # Figure 1: ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(fpr, tpr, linewidth=1.8, label=f"IF (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Local IDS (Test)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.text(
        0.5,
        -0.02,
        "Proves the self-supervised IF separates benign vs attack windows on unseen test data.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    _save_figure(fig, outputs.figures_dir / "local_ids_roc_curve")

    # Figure 2: score distribution + threshold (median across vehicles for display)
    display_threshold = float(np.median(list(thresholds.values())))
    benign_scores = test_all.loc[test_all["label"] == 0, "anomaly_score"]
    attack_scores = test_all.loc[test_all["label"] == 1, "anomaly_score"]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.hist(benign_scores, bins=50, alpha=0.6, density=True, label="Benign (test)")
    ax.hist(attack_scores, bins=50, alpha=0.6, density=True, label="Attack (test)")
    ax.axvline(display_threshold, color="red", linestyle="--", linewidth=1.5, label="Val threshold")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution")
    ax.legend(loc="upper center", fontsize=8)
    fig.text(
        0.5,
        -0.02,
        "Shows score separation; threshold tuned on validation only.",
        ha="center",
        fontsize=8,
        style="italic",
    )
    _save_figure(fig, outputs.figures_dir / "local_ids_anomaly_score_distribution")

    # Figure 3: latency by attack type
    if "attack_type" in test_all.columns and not by_attack.empty:
        plot_df = by_attack.copy()
        plot_df["Detection Latency (ms)"] = pd.to_numeric(
            plot_df["Detection Latency (ms)"], errors="coerce"
        )
        plot_df = plot_df.dropna(subset=["Detection Latency (ms)"])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(4.5, 3.0))
            ax.bar(plot_df["Attack Type"].astype(str), plot_df["Detection Latency (ms)"], color="#4472C4")
            ax.set_ylabel("Detection Latency (ms)")
            ax.set_xlabel("Attack Type")
            ax.set_title("Detection Latency by Attack Type")
            plt.xticks(rotation=35, ha="right")
            fig.text(
                0.5,
                -0.08,
                "Mean time from attack onset to first window-level alert (test split).",
                ha="center",
                fontsize=8,
                style="italic",
            )
            _save_figure(fig, outputs.figures_dir / "local_ids_detection_latency_by_attack")

    written = {
        "vehicle_level_metrics": metrics_csv,
        "vehicle_level_thresholds": threshold_path,
        "table_vehicle_level_ids_tex": outputs.tables_dir / "table_vehicle_level_ids.tex",
        "table_vehicle_level_ids_md": outputs.tables_dir / "table_vehicle_level_ids.md",
        "local_ids_roc_curve": outputs.figures_dir / "local_ids_roc_curve.png",
        "local_ids_anomaly_score_distribution": outputs.figures_dir
        / "local_ids_anomaly_score_distribution.png",
    }
    logger.info("Vehicle-level evaluation complete. Metrics: %s", metrics_csv)
    return written
