"""Local behavioural feature extraction for CTT windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ctt.constants import OUTPUT_ROOT
from src.ctt.utils import ensure_dir, write_markdown

BYTE_COLS = [f"byte_{i}" for i in range(8)]

LOCAL_FEATURE_COLUMNS: list[str] = [
    "frame_count",
    "unique_can_id_count",
    "can_id_entropy",
    "most_common_can_id_ratio",
    "id_transition_rate",
    "id_repetition_rate",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "min_inter_arrival_time",
    "max_inter_arrival_time",
    "message_rate",
    "mean_dlc",
    "std_dlc",
    "dlc_mode_ratio",
    *[f"byte_mean_{i}" for i in range(8)],
    *[f"byte_std_{i}" for i in range(8)],
    "payload_change_rate",
    "payload_static_ratio",
    # benign-profile deviation slots (filled during extraction when profile provided)
    "deviation_can_id_entropy",
    "deviation_message_rate",
    "deviation_mean_dlc",
    "deviation_byte_mean_norm",
]

METADATA_COLS = [
    "window_id",
    "vehicle_id",
    "attack_type",
    "label",
    "dataset_set",
    "subset_name",
    "source_file",
    "normalized_path",
    "start_frame_idx",
    "end_frame_idx",
]


def _entropy(arr: np.ndarray) -> float:
    if arr.size == 0:
        return np.nan
    _, counts = np.unique(arr, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def compute_benign_profile(benign_features: pd.DataFrame) -> dict[str, float]:
    """Mean benign profile for deviation features."""
    cols = [c for c in LOCAL_FEATURE_COLUMNS if c in benign_features.columns and not c.startswith("deviation")]
    return {c: float(benign_features[c].mean()) for c in cols if c in benign_features.columns}


def extract_window_features(
    window_df: pd.DataFrame,
    benign_profile: dict[str, float] | None = None,
) -> dict[str, float]:
    """Extract vehicle-agnostic behavioural features from one window."""
    n = len(window_df)
    feat: dict[str, float] = {"frame_count": float(n)}

    can_ids = window_df["can_id"].astype(str).to_numpy()
    feat["unique_can_id_count"] = float(len(np.unique(can_ids)))
    feat["can_id_entropy"] = _entropy(can_ids)
    _, counts = np.unique(can_ids, return_counts=True)
    feat["most_common_can_id_ratio"] = float(counts.max() / n) if n else np.nan

    if n > 1:
        transitions = np.sum(can_ids[1:] != can_ids[:-1])
        feat["id_transition_rate"] = float(transitions / (n - 1))
        repeats = n - transitions - 1
        feat["id_repetition_rate"] = float(max(repeats, 0) / (n - 1))
    else:
        feat["id_transition_rate"] = 0.0
        feat["id_repetition_rate"] = 0.0

    ts = pd.to_numeric(window_df["timestamp"], errors="coerce").to_numpy()
    if n > 1:
        inter = np.diff(ts)
        inter = inter[np.isfinite(inter) & (inter >= 0)]
        if inter.size:
            feat["mean_inter_arrival_time"] = float(np.mean(inter))
            feat["std_inter_arrival_time"] = float(np.std(inter))
            feat["min_inter_arrival_time"] = float(np.min(inter))
            feat["max_inter_arrival_time"] = float(np.max(inter))
            span = float(ts[-1] - ts[0]) if ts[-1] > ts[0] else float(inter.sum())
            feat["message_rate"] = float(n / span) if span > 0 else float(n)
        else:
            for k in ("mean_inter_arrival_time", "std_inter_arrival_time", "min_inter_arrival_time",
                      "max_inter_arrival_time", "message_rate"):
                feat[k] = np.nan
    else:
        for k in ("mean_inter_arrival_time", "std_inter_arrival_time", "min_inter_arrival_time",
                  "max_inter_arrival_time", "message_rate"):
            feat[k] = np.nan

    dlc = pd.to_numeric(window_df["dlc"], errors="coerce").to_numpy()
    feat["mean_dlc"] = float(np.nanmean(dlc))
    feat["std_dlc"] = float(np.nanstd(dlc))
    dlc_mode = pd.Series(dlc).mode()
    feat["dlc_mode_ratio"] = float((dlc == dlc_mode.iloc[0]).mean()) if len(dlc_mode) else np.nan

    byte_means = []
    for i, col in enumerate(BYTE_COLS):
        vals = pd.to_numeric(window_df[col], errors="coerce").to_numpy()
        feat[f"byte_mean_{i}"] = float(np.nanmean(vals))
        feat[f"byte_std_{i}"] = float(np.nanstd(vals))
        byte_means.append(feat[f"byte_mean_{i}"])

    if n > 1:
        payload_cols = BYTE_COLS
        changes = 0
        static = 0
        for col in payload_cols:
            v = pd.to_numeric(window_df[col], errors="coerce").to_numpy()
            changes += int(np.sum(v[1:] != v[:-1]))
            static += int(np.sum(v[1:] == v[:-1]))
        total_pairs = (n - 1) * len(payload_cols)
        feat["payload_change_rate"] = float(changes / total_pairs) if total_pairs else 0.0
        feat["payload_static_ratio"] = float(static / total_pairs) if total_pairs else 0.0
    else:
        feat["payload_change_rate"] = 0.0
        feat["payload_static_ratio"] = 0.0

    # Benign profile deviation
    if benign_profile:
        feat["deviation_can_id_entropy"] = abs(feat["can_id_entropy"] - benign_profile.get("can_id_entropy", 0))
        feat["deviation_message_rate"] = abs(feat.get("message_rate", 0) - benign_profile.get("message_rate", 0))
        feat["deviation_mean_dlc"] = abs(feat["mean_dlc"] - benign_profile.get("mean_dlc", 0))
        bm = np.nanmean(byte_means)
        feat["deviation_byte_mean_norm"] = abs(bm - benign_profile.get("byte_mean_0", 0))
    else:
        feat["deviation_can_id_entropy"] = 0.0
        feat["deviation_message_rate"] = 0.0
        feat["deviation_mean_dlc"] = 0.0
        feat["deviation_byte_mean_norm"] = 0.0

    return feat


def extract_features_from_manifest(
    window_manifest: pd.DataFrame,
    output_root: Path = OUTPUT_ROOT,
) -> pd.DataFrame:
    """Extract features for all windows in manifest."""
    feature_rows: list[dict] = []

    for norm_path, group in window_manifest.groupby("normalized_path"):
        frames = pd.read_csv(norm_path)
        for _, meta in group.iterrows():
            start, end = int(meta["start_frame_idx"]), int(meta["end_frame_idx"])
            chunk = frames.iloc[start:end]
            feat = extract_window_features(chunk)
            row = {c: meta[c] for c in METADATA_COLS if c in meta}
            row.update(feat)
            feature_rows.append(row)

    return pd.DataFrame(feature_rows)


def write_feature_schema(output_root: Path = OUTPUT_ROOT) -> None:
    manifest_dir = ensure_dir(output_root / "manifests")
    audit_dir = ensure_dir(output_root / "audit")
    schema_df = pd.DataFrame(
        {
            "feature_name": LOCAL_FEATURE_COLUMNS,
            "vehicle_agnostic": True,
            "used_in_model": True,
            "excluded_fields": "",
        }
    )
    schema_df.to_csv(manifest_dir / "local_feature_schema.csv", index=False)

    sections = {
        "Compatibility": (
            "Features align with the OCSLab framework philosophy: inter-arrival statistics, "
            "message rate, CAN-ID entropy, ID transition/repetition, payload byte statistics, "
            "DLC statistics, and benign-profile deviation features."
        ),
        "Excluded from model features": (
            "vehicle_id, source_file, label, attack_type, campaign metadata, future data"
        ),
    }
    write_markdown(audit_dir / "local_feature_compatibility_report.md", "Local Feature Compatibility", sections)
