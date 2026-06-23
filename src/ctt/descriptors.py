"""Anomaly descriptor generation for fleet layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ctt.constants import OUTPUT_ROOT
from src.ctt.descriptor_sampling import balanced_sample_descriptors
from src.ctt.features import LOCAL_FEATURE_COLUMNS
from src.ctt.utils import ensure_dir, write_markdown

DESCRIPTOR_FEATURE_COLS = [c for c in LOCAL_FEATURE_COLUMNS if not c.startswith("deviation")]


def generate_descriptors(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    output_root: Path = OUTPUT_ROOT,
    max_descriptors: int | None = None,
) -> pd.DataFrame:
    """Generate compact anomaly descriptors for weak candidates."""
    merged = predictions.merge(
        features,
        on=["window_id", "vehicle_id", "dataset_set", "subset_name", "attack_type", "label"],
        how="inner",
        suffixes=("", "_feat"),
    )
    candidates = merged[merged["weak_prediction"] == 1].copy()
    if max_descriptors is not None and len(candidates) > max_descriptors:
        candidates = balanced_sample_descriptors(candidates, max_descriptors)

    desc_rows = []
    meta_rows = []
    for _, row in candidates.iterrows():
        feat_vec = [float(row[c]) if c in row and pd.notna(row[c]) else 0.0 for c in DESCRIPTOR_FEATURE_COLS]
        eid = f"EVT-{row['vehicle_id'][:3].upper()}-{int(row['window_id']):08d}"
        desc_rows.append(
            {
                "event_id": eid,
                "anomaly_score": float(row["anomaly_score"]),
                "descriptor_vector": json.dumps(feat_vec, separators=(",", ":")),
                "descriptor_dim": len(feat_vec),
                "n_local_features": len(feat_vec),
            }
        )
        meta_rows.append(
            {
                "event_id": eid,
                "window_id": row["window_id"],
                "vehicle_id": row["vehicle_id"],
                "manufacturer": row.get("manufacturer", ""),
                "attack_type": row["attack_type"],
                "label": int(row["label"]),
                "dataset_set": row["dataset_set"],
                "subset_name": row["subset_name"],
                "source_file": row.get("source_file", ""),
                "anomaly_score": float(row["anomaly_score"]),
                "weak_prediction": int(row["weak_prediction"]),
                "strong_prediction": int(row["strong_prediction"]),
            }
        )

    desc_df = pd.DataFrame(desc_rows)
    meta_df = pd.DataFrame(meta_rows)
    if not desc_df.empty:
        meta_extra = [c for c in meta_df.columns if c not in desc_df.columns]
        desc_df = desc_df.merge(meta_df[["event_id", *meta_extra]], on="event_id", how="left")

    desc_dir = ensure_dir(output_root / "descriptors")
    transfer_dir = ensure_dir(output_root / "results" / "descriptor_transfer")
    desc_df.to_csv(desc_dir / "fleet_candidate_descriptors.csv", index=False)
    meta_df.to_csv(desc_dir / "descriptor_metadata.csv", index=False)

    schema_df = pd.DataFrame(
        {
            "field": ["event_id", "descriptor_vector", "anomaly_score", "descriptor_dim"],
            "in_model_features": [False, True, True, False],
            "evaluation_only": ["vehicle_id", "attack_type", "label", "source_file"],
        }
    )
    schema_df.to_csv(output_root / "manifests" / "descriptor_schema.csv", index=False)

    # Communication summary
    n_candidates = len(desc_df)
    n_windows = len(merged)
    raw_bytes_per_window = 100 * 16  # approx 100 frames * 16 bytes
    desc_bytes = desc_df["descriptor_vector"].str.len().mean() if n_candidates else 0
    summary = pd.DataFrame(
        [
            {
                "n_windows_total": n_windows,
                "n_weak_candidates": n_candidates,
                "candidate_transmission_rate": n_candidates / n_windows if n_windows else 0,
                "raw_window_bytes_approx": raw_bytes_per_window,
                "mean_descriptor_bytes": desc_bytes,
                "bandwidth_reduction_ratio": 1 - (desc_bytes / raw_bytes_per_window) if raw_bytes_per_window else 0,
            }
        ]
    )
    summary.to_csv(transfer_dir / "communication_summary.csv", index=False)

    write_markdown(
        output_root / "audit" / "descriptor_privacy_and_compatibility_report.md",
        "Descriptor Privacy and Compatibility",
        {
            "Schema": "Descriptor vector contains behavioural features only; no raw CAN IDs, payloads, filenames, or labels.",
            "Compatibility": "Descriptor dimension matches local feature schema (vehicle-agnostic behavioural features).",
            "Option": "Option B — train fleet GNN on CTT training scenarios (cross-dataset framework validation).",
        },
    )
    return desc_df


def load_descriptor_vectors(desc_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    vectors = []
    ids = []
    for _, row in desc_df.iterrows():
        vec = json.loads(row["descriptor_vector"])
        vectors.append([float(v) if v is not None else 0.0 for v in vec])
        ids.append(row["event_id"])
    return np.asarray(vectors, dtype=np.float32), ids
