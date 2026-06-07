"""Vehicle-level self-supervised IDS evaluation for paper Table 1 and Figures 1–4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

SELECTED_METHOD_LABEL = "FPR<=5%"
PRIMARY_FPR_CAP = 0.05
FALLBACK_FPR_CAP = 0.10

THRESHOLD_METHODS = (
    "F1-optimal",
    "FPR<=1%",
    "FPR<=5%",
    "FPR<=10%",
    "validation benign 95th percentile",
    "validation benign 99th percentile",
)


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


@dataclass
class ThresholdPick:
    method: str
    value: float
    used_fallback: bool = False


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


def _threshold_candidates(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return np.array([0.5])
    qs = np.linspace(0.001, 0.999, 300)
    return np.unique(np.quantile(scores, qs))


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
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "precision_pct": precision * 100.0,
        "recall_pct": recall * 100.0,
        "f1_pct": f1 * 100.0,
        "false_positive_rate_pct": fpr * 100.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _metrics_at_threshold(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    pred = (scores >= threshold).astype(int)
    return _binary_metrics(y_true, pred, scores)


def _select_threshold_max_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(np.percentile(scores, 95))
    best_t, best_f1 = 0.5, -1.0
    for t in _threshold_candidates(scores):
        m = _metrics_at_threshold(y_true, scores, float(t))
        if m["f1"] > best_f1:
            best_f1, best_t = m["f1"], float(t)
    return best_t


def _select_threshold_fpr_then_recall(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_fpr: float,
) -> float | None:
    """Among validation thresholds with FPR <= max_fpr, pick highest recall."""
    best_t: float | None = None
    best_recall = -1.0
    for t in _threshold_candidates(scores):
        m = _metrics_at_threshold(y_true, scores, float(t))
        if m["fpr"] <= max_fpr + 1e-12 and m["recall"] > best_recall:
            best_recall = m["recall"]
            best_t = float(t)
    return best_t


def _benign_percentile_threshold(y_true: np.ndarray, scores: np.ndarray, percentile: float) -> float:
    benign = scores[y_true == 0]
    if benign.size == 0:
        return float(np.percentile(scores, percentile))
    return float(np.percentile(benign, percentile))


def _resolve_threshold_for_method(
    method: str,
    y_val: np.ndarray,
    val_scores: np.ndarray,
) -> ThresholdPick:
    if method == "F1-optimal":
        return ThresholdPick(method=method, value=_select_threshold_max_f1(y_val, val_scores))
    if method == "FPR<=1%":
        t = _select_threshold_fpr_then_recall(y_val, val_scores, max_fpr=0.01)
        if t is None:
            t = _select_threshold_fpr_then_recall(y_val, val_scores, max_fpr=PRIMARY_FPR_CAP)
        return ThresholdPick(method=method, value=float(t if t is not None else 0.99))
    if method == "FPR<=5%":
        t = _select_threshold_fpr_then_recall(y_val, val_scores, max_fpr=PRIMARY_FPR_CAP)
        if t is not None:
            return ThresholdPick(method=method, value=t, used_fallback=False)
        t_fb = _select_threshold_fpr_then_recall(y_val, val_scores, max_fpr=FALLBACK_FPR_CAP)
        logger.warning(
            "No validation threshold achieves FPR<=5%%; using FPR<=10%% fallback (t=%.4f).",
            t_fb,
        )
        return ThresholdPick(
            method=f"{method} (fallback FPR<=10%)",
            value=float(t_fb if t_fb is not None else 0.99),
            used_fallback=True,
        )
    if method == "FPR<=10%":
        t = _select_threshold_fpr_then_recall(y_val, val_scores, max_fpr=FALLBACK_FPR_CAP)
        return ThresholdPick(method=method, value=float(t if t is not None else 0.99))
    if method == "validation benign 95th percentile":
        return ThresholdPick(method=method, value=_benign_percentile_threshold(y_val, val_scores, 95.0))
    if method == "validation benign 99th percentile":
        return ThresholdPick(method=method, value=_benign_percentile_threshold(y_val, val_scores, 99.0))
    raise ValueError(f"Unknown threshold method: {method}")


def _comparison_row(
    method: str,
    threshold: float,
    y_val: np.ndarray,
    val_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    latency_ms: float,
) -> dict[str, Any]:
    val_m = _metrics_at_threshold(y_val, val_scores, threshold)
    test_m = _metrics_at_threshold(y_test, test_scores, threshold)
    return {
        "threshold_method": method,
        "threshold_value": round(threshold, 6),
        "validation_precision": round(val_m["precision"], 4),
        "validation_recall": round(val_m["recall"], 4),
        "validation_f1": round(val_m["f1"], 4),
        "validation_fpr": round(val_m["fpr"], 4),
        "test_precision": round(test_m["precision"], 4),
        "test_recall": round(test_m["recall"], 4),
        "test_f1": round(test_m["f1"], 4),
        "test_fpr": round(test_m["fpr"], 4),
        "test_roc_auc": round(test_m["roc_auc"], 4),
        "test_pr_auc": round(test_m["pr_auc"], 4),
        "detection_latency_ms": round(latency_ms, 2) if not np.isnan(latency_ms) else "",
    }


def _window_duration_ms(row: pd.Series) -> float:
    iat = float(row.get("mean_inter_arrival_time", 0.0) or 0.0)
    frames = float(row.get("frame_count", row.get("n_frames", 1)) or 1)
    return max(iat * frames * 1000.0, 0.0)


def _detection_latency_ms(sub: pd.DataFrame, threshold: float) -> float:
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


def _write_interpretation(
    path: Path,
    *,
    f1_row: dict[str, Any],
    selected_row: dict[str, Any],
    used_fallback: bool,
) -> None:
    lines = [
        "# Vehicle-Level IDS — Threshold Interpretation",
        "",
        "## Why not use the F1-optimal threshold?",
        "",
        (
            f"On the held-out test split, the **F1-optimal** validation threshold "
            f"({f1_row['threshold_value']}) yields recall {f1_row['test_recall']:.2%} but "
            f"a false positive rate of **{f1_row['test_fpr']:.2%}**. "
            "That means most benign CAN windows would be flagged as attacks, which is "
            "impractical for in-vehicle deployment."
        ),
        "",
        "## Why use an FPR-constrained threshold?",
        "",
        (
            "Automotive CAN intrusion detection must limit false alarms so operators "
            "can trust alerts. We therefore select the threshold on **validation data only**, "
            "requiring validation FPR ≤ 5%, then choosing the threshold with the **highest "
            "validation recall** among feasible candidates."
        ),
        "",
        "## Selected paper result",
        "",
    ]
    if used_fallback:
        lines.append(
            "- **Note:** No threshold achieved validation FPR ≤ 5%; the pipeline used the "
            "**FPR ≤ 10% fallback** (reported in the threshold comparison table)."
        )
    lines.extend(
        [
            f"- **Selected method:** {selected_row['threshold_method']}",
            f"- **Threshold value:** {selected_row['threshold_value']}",
            f"- **Test recall:** {selected_row['test_recall']:.2%}",
            f"- **Test precision:** {selected_row['test_precision']:.2%}",
            f"- **Test F1:** {selected_row['test_f1']:.2%}",
            f"- **Test FPR:** {selected_row['test_fpr']:.2%}",
            f"- **Test detection latency:** {selected_row['detection_latency_ms']} ms",
            "",
            "## Trade-off",
            "",
            (
                "Compared with the F1-optimal threshold, the FPR-controlled rule reduces "
                "false positives at the cost of lower attack recall. This is expected: "
                "stricter control of benign false alarms necessarily misses some subtle "
                "attack windows that score below the higher threshold."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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
    """Train IF on benign train; tune thresholds on val; report on test with FPR control."""
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

    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    per_vehicle_selected: dict[str, float] = {}

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

        val_part = val_df.copy()
        val_part["anomaly_score"] = score_part(val_df)
        test_part = test_df.copy()
        test_part["anomaly_score"] = score_part(test_df)

        y_val = val_part["label"].to_numpy(dtype=np.int64)
        val_scores = val_part["anomaly_score"].to_numpy(dtype=np.float64)
        pick = _resolve_threshold_for_method("FPR<=5%", y_val, val_scores)
        per_vehicle_selected[vehicle] = pick.value

        val_parts.append(val_part)
        test_parts.append(test_part)
        logger.info(
            "%s: selected threshold=%.4f (%s), test windows=%d",
            vehicle,
            pick.value,
            pick.method,
            len(test_part),
        )

    if not test_parts:
        raise ValueError("No vehicle models evaluated.")

    val_all = pd.concat(val_parts, ignore_index=True)
    test_all = pd.concat(test_parts, ignore_index=True)
    y_val = val_all["label"].to_numpy(dtype=np.int64)
    val_scores = val_all["anomaly_score"].to_numpy(dtype=np.float64)
    y_test = test_all["label"].to_numpy(dtype=np.int64)
    test_scores = test_all["anomaly_score"].to_numpy(dtype=np.float64)

    # Pooled threshold comparison (all methods)
    comparison_rows: list[dict[str, Any]] = []
    pooled_picks: dict[str, ThresholdPick] = {}
    for method in THRESHOLD_METHODS:
        pick = _resolve_threshold_for_method(method, y_val, val_scores)
        pooled_picks[method] = pick
        test_all_scored = test_all.copy()
        test_all_scored["predicted_label"] = (
            test_all_scored["anomaly_score"] >= pick.value
        ).astype(int)
        latencies: list[float] = []
        for vehicle, vgrp in test_all_scored.groupby("vehicle_model"):
            lat = _detection_latency_ms(vgrp, threshold=pick.value)
            if not np.isnan(lat):
                latencies.append(lat)
        pooled_lat = float(np.mean(latencies)) if latencies else float("nan")
        comparison_rows.append(
            _comparison_row(
                pick.method,
                pick.value,
                y_val,
                val_scores,
                y_test,
                test_scores,
                pooled_lat,
            )
        )

    comparison_df = pd.DataFrame(comparison_rows)
    f1_row = comparison_df[comparison_df["threshold_method"] == "F1-optimal"].iloc[0].to_dict()
    selected_pool = pooled_picks["FPR<=5%"]
    selected_row = comparison_df[
        comparison_df["threshold_method"].str.startswith("FPR<=5%")
    ].iloc[0].to_dict()

    # Apply per-vehicle FPR<=5% thresholds for final Table 1 (deployment-realistic)
    test_frames: list[pd.DataFrame] = []
    for vehicle, part in test_all.groupby("vehicle_model"):
        t = per_vehicle_selected[vehicle]
        part = part.copy()
        part["predicted_label"] = (part["anomaly_score"] >= t).astype(int)
        part["threshold"] = t
        test_frames.append(part)
    test_eval = pd.concat(test_frames, ignore_index=True)
    y_pred = test_eval["predicted_label"].to_numpy(dtype=np.int64)

    metrics = _binary_metrics(y_test, y_pred, test_scores)
    latencies_per_vehicle: list[float] = []
    for vehicle, part in test_eval.groupby("vehicle_model"):
        lat = _detection_latency_ms(part, threshold=per_vehicle_selected[vehicle])
        if not np.isnan(lat):
            latencies_per_vehicle.append(lat)
    pooled_latency = float(np.mean(latencies_per_vehicle)) if latencies_per_vehicle else float("nan")

    selected_method_label = selected_pool.method
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
                "Selected Threshold Method": selected_method_label,
            }
        ]
    )

    outputs.results_dir.mkdir(parents=True, exist_ok=True)
    outputs.tables_dir.mkdir(parents=True, exist_ok=True)
    outputs.figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = outputs.results_dir / "vehicle_level_metrics.csv"
    table1.to_csv(metrics_csv, index=False)

    comparison_csv = outputs.results_dir / "vehicle_level_threshold_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    threshold_path = outputs.results_dir / "vehicle_level_thresholds.json"
    with threshold_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": SELF_SUPERVISED_IDS_MODEL,
                "training": "benign_train_only",
                "selected_threshold_method": selected_method_label,
                "selected_pooled_threshold": selected_pool.value,
                "used_fpr_10_fallback": selected_pool.used_fallback,
                "per_vehicle_threshold_fpr_le_5pct": per_vehicle_selected,
                "pooled_thresholds_by_method": {
                    m: pooled_picks[m].value for m in THRESHOLD_METHODS
                },
            },
            fh,
            indent=2,
        )

    # By attack type (selected thresholds)
    by_attack = pd.DataFrame()
    if "attack_type" in test_eval.columns:
        rows: list[dict[str, Any]] = []
        for attack, grp in test_eval.groupby("attack_type", sort=True):
            yt = grp["label"].to_numpy(dtype=np.int64)
            yp = grp["predicted_label"].to_numpy(dtype=np.int64)
            ys = grp["anomaly_score"].to_numpy(dtype=np.float64)
            m = _binary_metrics(yt, yp, ys)
            lat_parts = [
                _detection_latency_ms(vgrp, threshold=per_vehicle_selected[vehicle])
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
        by_attack.to_csv(outputs.results_dir / "vehicle_level_by_attack_type.csv", index=False)

    (outputs.tables_dir / "table_vehicle_level_ids.tex").write_text(
        _df_to_ieee_tex(
            table1,
            "Vehicle-level IDS on test data using FPR-constrained threshold selection.",
            "tab:vehicle-ids",
        ),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_vehicle_level_ids.md").write_text(
        "# Table 1: Vehicle-Level IDS Performance\n\n" + _df_to_markdown(table1),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_threshold_comparison.tex").write_text(
        _df_to_ieee_tex(
            comparison_df,
            "Threshold selection comparison (validation tuning, test evaluation).",
            "tab:threshold-comparison",
        ),
        encoding="utf-8",
    )
    (outputs.tables_dir / "table_threshold_comparison.md").write_text(
        "# Threshold Comparison\n\n" + _df_to_markdown(comparison_df),
        encoding="utf-8",
    )

    _write_interpretation(
        outputs.results_dir / "vehicle_level_interpretation.md",
        f1_row=f1_row,
        selected_row=selected_row,
        used_fallback=selected_pool.used_fallback,
    )

    display_threshold = float(np.median(list(per_vehicle_selected.values())))

    # Figure 1: ROC (score-based; threshold-independent; supporting artefact)
    fpr_curve, tpr_curve, _ = roc_curve(y_test, test_scores)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(fpr_curve, tpr_curve, linewidth=1.8, label=f"IF (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC — Local IDS (Test)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, outputs.figures_dir / "local_ids_roc_curve")

    # Figure 2 (paper): precision–recall curve (imbalanced attack/benign setting)
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, test_scores)
    pr_auc = float(average_precision_score(y_test, test_scores))
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.plot(rec_curve, prec_curve, linewidth=1.8, label=f"IF (AP={pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall — Local IDS (Test)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    _save_figure(fig, outputs.figures_dir / "local_ids_pr_curve")

    # Supporting: per-attack F1 at the selected operating point (not in IEEE paper bundle)
    if not by_attack.empty:
        plot_df = by_attack[by_attack["Attack Type"].astype(str) != "attack_free"].copy()
        plot_df["F1-Score (%)"] = pd.to_numeric(plot_df["F1-Score (%)"], errors="coerce")
        plot_df = plot_df.dropna(subset=["F1-Score (%)"])
        if not plot_df.empty:
            labels = [str(a).replace("_", " ").capitalize() for a in plot_df["Attack Type"]]
            fig, ax = plt.subplots(figsize=(4.5, 3.0))
            bars = ax.bar(labels, plot_df["F1-Score (%)"], color="#4472C4")
            ax.set_ylabel("F1-score (%)")
            ax.set_xlabel("Attack type")
            ax.set_title("Local IDS F1-score by Attack Type (Test)")
            ax.set_ylim(0, 105)
            for bar, val in zip(bars, plot_df["F1-Score (%)"]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            plt.xticks(rotation=20, ha="right")
            _save_figure(fig, outputs.figures_dir / "local_ids_f1_by_attack_type")

    # Supporting: score distribution (not used in IEEE paper bundle)
    benign_scores = test_eval.loc[test_eval["label"] == 0, "anomaly_score"]
    attack_scores = test_eval.loc[test_eval["label"] == 1, "anomaly_score"]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.hist(benign_scores, bins=50, alpha=0.6, density=True, label="Benign (test)")
    ax.hist(attack_scores, bins=50, alpha=0.6, density=True, label="Attack (test)")
    ax.axvline(
        display_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Selected threshold: FPR<=5%",
    )
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title("Anomaly Score Distribution")
    ax.legend(loc="upper center", fontsize=7)
    _save_figure(fig, outputs.figures_dir / "local_ids_anomaly_score_distribution")

    # Supporting: latency by attack type
    if not by_attack.empty:
        plot_df = by_attack.copy()
        plot_df["Detection Latency (ms)"] = pd.to_numeric(
            plot_df["Detection Latency (ms)"], errors="coerce"
        )
        plot_df = plot_df.dropna(subset=["Detection Latency (ms)"])
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(4.5, 3.0))
            ax.bar(
                plot_df["Attack Type"].astype(str),
                plot_df["Detection Latency (ms)"],
                color="#4472C4",
            )
            ax.set_ylabel("Detection Latency (ms)")
            ax.set_xlabel("Attack Type")
            ax.set_title("Detection Latency by Attack Type")
            plt.xticks(rotation=35, ha="right")
            _save_figure(fig, outputs.figures_dir / "local_ids_detection_latency_by_attack")

    # Figure 4: threshold trade-off on pooled validation
    recalls, fprs, f1s, thresh = [], [], [], []
    for t in _threshold_candidates(val_scores):
        m = _metrics_at_threshold(y_val, val_scores, float(t))
        recalls.append(m["recall_pct"])
        fprs.append(m["false_positive_rate_pct"])
        f1s.append(m["f1_pct"])
        thresh.append(float(t))
    fig, ax1 = plt.subplots(figsize=(4.0, 3.0))
    ax1.plot(thresh, recalls, label="Recall (%)", color="#2E75B6")
    ax1.plot(thresh, fprs, label="FPR (%)", color="#C00000")
    ax1.plot(thresh, f1s, label="F1 (%)", color="#548235")
    ax1.axvline(display_threshold, color="black", linestyle="--", linewidth=1.0, label="Selected")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Metric (%)")
    ax1.set_title("Validation Threshold Trade-off")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    _save_figure(fig, outputs.figures_dir / "local_ids_threshold_tradeoff")

    logger.info(
        "Vehicle-level evaluation complete. Test FPR=%.2f%% (FPR-controlled), metrics: %s",
        metrics["false_positive_rate_pct"],
        metrics_csv,
    )

    return {
        "vehicle_level_metrics": metrics_csv,
        "vehicle_level_threshold_comparison": comparison_csv,
        "vehicle_level_thresholds": threshold_path,
        "vehicle_level_interpretation": outputs.results_dir / "vehicle_level_interpretation.md",
        "table_vehicle_level_ids_tex": outputs.tables_dir / "table_vehicle_level_ids.tex",
        "local_ids_threshold_tradeoff": outputs.figures_dir / "local_ids_threshold_tradeoff.png",
    }
