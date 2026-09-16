#!/usr/bin/env python3
"""Build scored window pool from manuscript car_track Sonata/Soul/Spark traces.

Uses recovered publication balanced_split_manifest.csv for source-trace/segment
train/validation/test assignment (no overlapping-window leakage across splits).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.features.feature_extractor import (  # noqa: E402
    BEHAVIOURAL_FEATURE_COLUMNS,
    extract_features_for_trace,
)
from src.features.window_generator import generate_windows_for_trace  # noqa: E402
from src.models.vehicle_ids import (  # noqa: E402
    fit_self_supervised_isolation_forest,
    score_self_supervised_isolation_forest,
)

from fleet_scaler import FEATURE_NAMES  # noqa: E402

CAR_TRACK = ROOT / "Dataset" / "ocslab_pipeline" / "_source_car_track"
PUB_SPLIT = (
    ROOT
    / "recovered_publication_pipeline"
    / "new_experiments"
    / "final_end_to_end_publication_run_balanced"
    / "manifests"
    / "balanced_split_manifest.csv"
)
BYTE_COLS = [f"byte{i}" for i in range(8)]


def _bytes_from_tokens(tokens: list[str], dlc: int) -> dict[str, float]:
    out = {c: np.nan for c in BYTE_COLS}
    for i, tok in enumerate(tokens[: min(8, max(int(dlc), 0))]):
        try:
            out[f"byte{i}"] = float(int(str(tok), 16))
        except ValueError:
            out[f"byte{i}"] = np.nan
    return out


def resolve_local_path(onedrive_path: str) -> Path:
    """Map publication OneDrive path → local _source_car_track path."""
    p = Path(onedrive_path)
    # .../car_track_*/basename.csv
    parts = p.parts
    track = None
    for part in parts:
        if part.startswith("car_track_"):
            track = part
            break
    if track is None:
        raise FileNotFoundError(f"No car_track folder in {onedrive_path}")
    local = CAR_TRACK / track / p.name
    if not local.exists():
        raise FileNotFoundError(f"Missing local car_track file: {local}")
    return local


def load_car_track_frames(
    path: Path,
    *,
    vehicle_model: str,
    attack_type: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    """Load frames [start, end) from a car_track CSV (payload hex; optional R/T)."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i < start:
                continue
            if i >= end:
                break
            parts = [x.strip() for x in line.strip().split(",")]
            if len(parts) < 4:
                continue
            try:
                ts = float(parts[0])
                dlc = int(float(parts[2]))
            except ValueError:
                continue
            flag = None
            payload_tokens: list[str]
            if parts[-1].upper() in {"R", "T"} and len(parts) >= 5:
                flag = parts[-1].upper()
                payload_tokens = parts[3].split() if len(parts) == 5 else parts[3:-1]
                # common: ts,id,dlc,b0 b1 ... OR ts,id,dlc,b0,b1,...,flag
                if len(parts) > 5:
                    payload_tokens = parts[3:-1]
            else:
                payload_tokens = parts[3].split() if " " in parts[3] else parts[3:]
            if flag is None:
                # attack_free traces have no flag → benign
                label = 0.0
                atk = "attack_free"
            else:
                label = 1.0 if flag == "T" else 0.0
                atk = attack_type if label > 0 else "attack_free"
            rows.append(
                {
                    "timestamp": ts,
                    "can_id": parts[1].upper().replace("0X", ""),
                    "dlc": dlc,
                    **_bytes_from_tokens(payload_tokens, dlc),
                    "label": label,
                    "attack_type": atk,
                    "vehicle_model": vehicle_model,
                    "source_file": str(path),
                    "source_basename": path.name,
                    "frame_idx": i,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def windowize_segment(
    frames: pd.DataFrame,
    *,
    window_size: int = 100,
    stride: int = 50,
    split: str,
    segment_id: str,
) -> pd.DataFrame:
    if frames.empty or len(frames) < window_size:
        return pd.DataFrame()
    frames = frames.reset_index(drop=True)
    meta = generate_windows_for_trace(
        frames,
        window_size=window_size,
        stride=stride,
        source_file=str(frames["source_file"].iloc[0]),
        id_offset=0,
    )
    if meta.empty:
        return pd.DataFrame()
    feats = extract_features_for_trace(frames, meta)
    labels: list[int] = []
    attack_types: list[str] = []
    for _, w in meta.iterrows():
        chunk = frames.iloc[int(w["start_frame_idx"]) : int(w["end_frame_idx"])]
        y = int((chunk["label"] > 0).any())
        labels.append(y)
        if y:
            attack_types.append(
                str(chunk.loc[chunk["label"] > 0, "attack_type"].value_counts().index[0])
            )
        else:
            attack_types.append("attack_free")
    out = feats.copy()
    out["label"] = labels
    out["attack_type"] = attack_types
    out["split"] = split
    out["segment_id"] = segment_id
    out["vehicle_model"] = frames["vehicle_model"].iloc[0]
    out["source_file"] = frames["source_file"].iloc[0]
    out["source_basename"] = frames["source_basename"].iloc[0]
    out["physical_vehicle"] = frames["vehicle_model"].iloc[0]
    out["source_trace"] = frames["source_basename"].iloc[0]
    return out


def derive_gnn9(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["message_rate"] = out["frame_count"].astype(np.float64)
    out["burstiness"] = out["std_inter_arrival_time"].astype(np.float64) / (
        out["mean_inter_arrival_time"].astype(np.float64).abs() + 1e-9
    )
    mean_cols = [c for c in out.columns if c.startswith("byte_mean_")]
    if mean_cols:
        vals = np.abs(out[mean_cols].to_numpy(dtype=np.float64)) + 1e-9
        probs = vals / vals.sum(axis=1, keepdims=True)
        out["payload_entropy"] = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    else:
        out["payload_entropy"] = 0.0
    return out


def build() -> pd.DataFrame:
    assert CAR_TRACK.exists(), CAR_TRACK
    man = pd.read_csv(PUB_SPLIT)
    split_rows: list[dict[str, Any]] = []
    window_parts: list[pd.DataFrame] = []

    for _, row in man.iterrows():
        local = resolve_local_path(str(row["source_file"]))
        start = int(row["segment_start"])
        end = int(row["segment_end"])
        frames = load_car_track_frames(
            local,
            vehicle_model=str(row["vehicle_model"]),
            attack_type=str(row["attack_type"]),
            start=start,
            end=end,
        )
        n_frames = len(frames)
        split_rows.append(
            {
                "source_file": str(local),
                "source_basename": local.name,
                "vehicle_model": row["vehicle_model"],
                "attack_type": row["attack_type"],
                "split": row["split"],
                "segment_id": row["segment_id"],
                "segment_start": start,
                "segment_end": end,
                "n_frames_loaded": n_frames,
                "split_method": row["split_method"],
                "publication_source_file": row["source_file"],
            }
        )
        print(
            f"[{row['split']}] {row['vehicle_model']} {local.name} "
            f"[{start}:{end}) -> {n_frames} frames"
        )
        if n_frames < 100:
            continue
        wins = windowize_segment(
            frames,
            split=str(row["split"]),
            segment_id=str(row["segment_id"]),
        )
        if not wins.empty:
            window_parts.append(wins)
            print(f"  windows={len(wins)}")

    split_df = pd.DataFrame(split_rows)
    # leakage check: source basename+segment should be unique to one split
    leak = (
        split_df.groupby(["source_file", "segment_start", "segment_end"])["split"]
        .nunique()
    )
    leaky = leak[leak > 1]
    if len(leaky):
        raise RuntimeError(f"Split leakage detected: {len(leaky)} segments")

    all_w = pd.concat(window_parts, ignore_index=True)
    all_w = derive_gnn9(all_w)
    all_w["window_uid"] = (
        all_w["vehicle_model"].astype(str)
        + "::"
        + all_w["source_basename"].astype(str)
        + "::"
        + all_w["segment_id"].astype(str)
        + "::"
        + all_w["window_id"].astype(str)
    )

    train_benign = all_w[
        (all_w["split"] == "train") & (all_w["label"] == 0)
    ].copy()
    if train_benign.empty:
        raise RuntimeError("No benign TRAIN windows for Isolation Forest")

    print(f"Fitting IF on {len(train_benign)} benign TRAIN windows...")
    X_benign = train_benign[BEHAVIOURAL_FEATURE_COLUMNS].fillna(0.0).to_numpy(
        dtype=np.float32
    )
    model = fit_self_supervised_isolation_forest(X_benign, random_state=42)
    X_all = all_w[BEHAVIOURAL_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    _, scores = score_self_supervised_isolation_forest(model, X_all, X_benign)
    all_w["anomaly_score"] = scores

    out_dir = EXP / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = EXP / "split_manifest.csv"
    split_df.to_csv(split_path, index=False)

    pool_path = out_dir / "all_scored_windows.csv"
    test_path = out_dir / "test_window_pool.csv"
    all_w.to_csv(pool_path, index=False)
    test = all_w[all_w["split"] == "test"].copy()
    test.to_csv(test_path, index=False)

    summary = {
        "n_all_windows": int(len(all_w)),
        "n_train": int((all_w["split"] == "train").sum()),
        "n_validation": int((all_w["split"] == "validation").sum()),
        "n_test": int(len(test)),
        "n_train_benign_if": int(len(train_benign)),
        "platforms": sorted(all_w["vehicle_model"].unique().tolist()),
        "test_by_platform": {
            str(k): int(v) for k, v in test["vehicle_model"].value_counts().items()
        },
        "test_strong": int(
            ((test["label"] == 1) & (test["anomaly_score"] >= 0.80)).sum()
        ),
        "test_weak": int(
            (
                (test["label"] == 1)
                & (test["anomaly_score"] >= 0.55)
                & (test["anomaly_score"] < 0.80)
            ).sum()
        ),
        "feature_names_9d": FEATURE_NAMES,
        "behavioural_24d": BEHAVIOURAL_FEATURE_COLUMNS,
        "split_manifest": str(split_path),
        "synthetic_descriptors": False,
        "leakage_errors": 0,
    }
    (out_dir / "pool_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return all_w


if __name__ == "__main__":
    build()
