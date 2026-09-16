#!/usr/bin/env python3
"""Build a REAL OCSLab scored TEST window pool for the reviewer ablation.

Uses current runnable FLEET-GUARD components:
  - 24-D behavioural features (``BEHAVIOURAL_FEATURE_COLUMNS``)
  - benign-only Isolation Forest
  - strong/weak thresholds 0.80 / 0.55
  - 9-D GNN view matching parent FEATURE_NAMES

No synthetic descriptors are generated.

Temporal protocol: interleaved contiguous blocks (70/15/15 by block count)
so attack-rich early regions appear in TEST. Each block is windowed
separately so inter-arrival statistics never cross block gaps.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
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

BYTE_COLS = [f"byte{i}" for i in range(8)]
FRAME_COLS = [
    "timestamp",
    "can_id",
    "dlc",
    *BYTE_COLS,
    "label",
    "attack_type",
    "vehicle_model",
    "source_file",
    "block_id",
]

GNN_FEATURE_COLUMNS = [
    "anomaly_score",
    "frame_count",
    "message_rate",
    "burstiness",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "can_id_entropy",
    "most_common_can_id_ratio",
    "payload_entropy",
]


def _bytes_from_tokens(tokens: list[str], dlc: int) -> dict[str, float]:
    out = {c: np.nan for c in BYTE_COLS}
    for i, tok in enumerate(tokens[: min(8, max(int(dlc), 0))]):
        try:
            out[f"byte{i}"] = float(int(str(tok), 16))
        except ValueError:
            out[f"byte{i}"] = np.nan
    return out


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


def _split_block_ranges(
    n: int,
    *,
    max_per_split: int,
    block_size: int = 5000,
    period: int = 20,
    train_blocks: int = 14,
    val_blocks: int = 3,
) -> dict[str, list[tuple[int, int]]]:
    """Interleaved contiguous blocks → train/val/test (70/15/15 by block count)."""
    if n <= 0:
        return {"train": [], "validation": [], "test": []}
    block_size = max(200, min(block_size, n))
    n_blocks = (n + block_size - 1) // block_size
    buckets: dict[str, list[tuple[int, int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for b in range(n_blocks):
        lo = b * block_size
        hi = min(n, lo + block_size)
        r = b % period
        if r < train_blocks:
            buckets["train"].append((lo, hi))
        elif r < train_blocks + val_blocks:
            buckets["validation"].append((lo, hi))
        else:
            buckets["test"].append((lo, hi))

    out: dict[str, list[tuple[int, int]]] = {}
    for split, blocks in buckets.items():
        kept: list[tuple[int, int]] = []
        total = 0
        for lo, hi in blocks:
            if total >= max_per_split:
                break
            take = min(hi - lo, max_per_split - total)
            if take <= 0:
                continue
            kept.append((lo, lo + take))
            total += take
        out[split] = kept
    return out


def _assign_split_block(
    i: int, blocks: dict[str, list[tuple[int, int]]]
) -> tuple[str, int] | None:
    for split, bl in blocks.items():
        for block_id, (lo, hi) in enumerate(bl):
            if lo <= i < hi:
                return split, block_id
    return None


def _parse_classic_csv_line(
    line: str, *, vehicle_model: str, attack_type: str, source: str
) -> dict[str, Any] | None:
    parts = [p.strip() for p in line.strip().split(",")]
    if len(parts) < 12:
        return None
    try:
        ts = float(parts[0])
        dlc = int(float(parts[2]))
    except ValueError:
        return None
    flag = parts[-1].upper()
    if flag not in {"R", "T"}:
        return None
    label = 1.0 if flag == "T" else 0.0
    return {
        "timestamp": ts,
        "can_id": parts[1].upper().replace("0X", ""),
        "dlc": dlc,
        **_bytes_from_tokens(parts[3:11], dlc),
        "label": label,
        "attack_type": attack_type if label > 0 else "attack_free",
        "vehicle_model": vehicle_model,
        "source_file": source,
    }


def load_classic_attack_csv_capped(
    path: Path,
    *,
    vehicle_model: str,
    attack_type: str,
    max_per_split: int,
) -> dict[str, pd.DataFrame]:
    n = _count_lines(path)
    blocks = _split_block_ranges(n, max_per_split=max_per_split)
    needed = {i for bl in blocks.values() for lo, hi in bl for i in range(lo, hi)}
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i not in needed:
                continue
            row = _parse_classic_csv_line(
                line,
                vehicle_model=vehicle_model,
                attack_type=attack_type,
                source=str(path),
            )
            if row is None:
                continue
            assigned = _assign_split_block(i, blocks)
            if assigned is None:
                continue
            split, block_id = assigned
            row = dict(row)
            row["block_id"] = int(block_id)
            rows_by_split[split].append(row)
    return {
        k: pd.DataFrame(v, columns=FRAME_COLS) if v else pd.DataFrame(columns=FRAME_COLS)
        for k, v in rows_by_split.items()
    }


def load_classic_normal_txt_capped(
    path: Path, *, vehicle_model: str, max_per_split: int
) -> dict[str, pd.DataFrame]:
    pat = re.compile(
        r"Timestamp:\s*([0-9.]+)\s+ID:\s*([0-9A-Fa-f]+)\s+\S+\s+DLC:\s*(\d+)\s+(.*)$"
    )
    n = _count_lines(path)
    blocks = _split_block_ranges(n, max_per_split=max_per_split)
    needed = {i for bl in blocks.values() for lo, hi in bl for i in range(lo, hi)}
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    with path.open(encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i not in needed:
                continue
            m = pat.search(line.strip())
            if not m:
                continue
            dlc = int(m.group(3))
            assigned = _assign_split_block(i, blocks)
            if assigned is None:
                continue
            split, block_id = assigned
            rows_by_split[split].append(
                {
                    "timestamp": float(m.group(1)),
                    "can_id": m.group(2).upper(),
                    "dlc": dlc,
                    **_bytes_from_tokens(m.group(4).split(), dlc),
                    "label": 0.0,
                    "attack_type": "attack_free",
                    "vehicle_model": vehicle_model,
                    "source_file": str(path),
                    "block_id": int(block_id),
                }
            )
    return {
        k: pd.DataFrame(v, columns=FRAME_COLS) if v else pd.DataFrame(columns=FRAME_COLS)
        for k, v in rows_by_split.items()
    }


def _iter_challenge_rows(path: Path, *, vehicle_model: str) -> Iterator[dict[str, Any]]:
    df = pd.read_csv(path)
    lower = {c.lower(): c for c in df.columns}
    if "arbitration_id" not in lower and "can_id" in lower:
        lower["arbitration_id"] = lower["can_id"]
    need = ["timestamp", "arbitration_id", "dlc", "data", "class"]
    if any(k not in lower for k in need):
        raise ValueError(f"Unexpected schema {path}: {df.columns.tolist()}")
    subclass_col = lower.get("subclass")
    for _, r in df.iterrows():
        cls = str(r[lower["class"]]).strip().lower()
        label = 1.0 if cls.startswith("attack") else 0.0
        if label > 0 and subclass_col is not None:
            atk = str(r[subclass_col]).strip().lower() or "challenge_attack"
        elif label > 0:
            atk = "challenge_attack"
        else:
            atk = "attack_free"
        try:
            dlc = int(float(r[lower["dlc"]]))
        except (TypeError, ValueError):
            dlc = 8
        yield {
            "timestamp": float(r[lower["timestamp"]]),
            "can_id": str(r[lower["arbitration_id"]]).upper().replace("0X", ""),
            "dlc": dlc,
            **_bytes_from_tokens(str(r[lower["data"]]).split(), dlc),
            "label": label,
            "attack_type": atk,
            "vehicle_model": vehicle_model,
            "source_file": str(path),
        }


def load_challenge_csv_capped(
    path: Path, *, vehicle_model: str, max_per_split: int
) -> dict[str, pd.DataFrame]:
    rows = list(_iter_challenge_rows(path, vehicle_model=vehicle_model))
    n = len(rows)
    blocks = _split_block_ranges(n, max_per_split=max_per_split)
    rows_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for i, row in enumerate(rows):
        assigned = _assign_split_block(i, blocks)
        if assigned is None:
            continue
        split, block_id = assigned
        row = dict(row)
        row["block_id"] = int(block_id)
        rows_by_split[split].append(row)
    return {
        k: pd.DataFrame(v, columns=FRAME_COLS) if v else pd.DataFrame(columns=FRAME_COLS)
        for k, v in rows_by_split.items()
    }


def compute_payload_entropy(df: pd.DataFrame) -> np.ndarray:
    mean_cols = [c for c in df.columns if c.startswith("byte_mean_")]
    if not mean_cols:
        return np.zeros(len(df), dtype=np.float64)
    vals = np.abs(df[mean_cols].to_numpy(dtype=np.float64)) + 1e-9
    probs = vals / vals.sum(axis=1, keepdims=True)
    return -np.sum(probs * np.log(probs + 1e-12), axis=1)


def windowize_trace(
    frames: pd.DataFrame,
    *,
    window_size: int = 100,
    stride: int = 50,
    max_windows: int = 2500,
) -> pd.DataFrame:
    if frames.empty or len(frames) < window_size:
        return pd.DataFrame()
    frames = frames.sort_values("timestamp").reset_index(drop=True)
    meta = generate_windows_for_trace(
        frames,
        window_size=window_size,
        stride=stride,
        source_file=str(frames["source_file"].iloc[0]),
        id_offset=0,
    )
    if meta.empty:
        return pd.DataFrame()
    if len(meta) > max_windows:
        idx = np.linspace(0, len(meta) - 1, max_windows).astype(int)
        meta = meta.iloc[idx].reset_index(drop=True)
        meta["window_id"] = np.arange(len(meta))

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
    feats = feats.copy()
    feats["label"] = labels
    feats["attack_type"] = attack_types
    return feats


def build_pool(
    classic_root: Path,
    challenge_root: Path | None,
    out_dir: Path,
    *,
    strong_threshold: float = 0.80,
    weak_threshold: float = 0.55,
    seed: int = 42,
    max_frames_per_split: int = 40000,
    max_windows_per_source: int = 1200,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    sources: list[tuple[str, Path, str, str]] = []

    classic_specs = [
        (
            "OCSLab_Car",
            classic_root / "normal_run_data" / "normal_run_data.txt",
            "attack_free",
            "normal_txt",
        ),
        (
            "OCSLab_Car",
            classic_root / "DoS_dataset" / "DoS_dataset.csv",
            "flooding",
            "classic_csv",
        ),
        (
            "OCSLab_Car",
            classic_root / "Fuzzy_dataset" / "Fuzzy_dataset.csv",
            "fuzzy",
            "classic_csv",
        ),
        (
            "OCSLab_Car",
            classic_root / "gear_dataset" / "gear_dataset.csv",
            "malfunction_gear",
            "classic_csv",
        ),
        (
            "OCSLab_Car",
            classic_root / "RPM_dataset" / "RPM_dataset.csv",
            "malfunction_rpm",
            "classic_csv",
        ),
    ]
    for item in classic_specs:
        if item[1].exists():
            sources.append(item)

    if challenge_root is not None:
        train_dir = challenge_root / "0_Preliminary" / "0_Training"
        for fname, vehicle in [
            ("Pre_train_D_0.csv", "Challenge_D"),
            ("Pre_train_D_1.csv", "Challenge_D"),
            ("Pre_train_D_2.csv", "Challenge_D"),
            ("Pre_train_S_0.csv", "Challenge_S"),
            ("Pre_train_S_1.csv", "Challenge_S"),
            ("Pre_train_S_2.csv", "Challenge_S"),
        ]:
            p = train_dir / fname
            if p.exists():
                sources.append((vehicle, p, "challenge", "challenge_csv"))

    window_parts: list[pd.DataFrame] = []
    provenance: list[dict[str, Any]] = []

    for vehicle, path, atk, kind in sources:
        print(f"Loading {path.name} ({kind}, {vehicle}) ...", flush=True)
        if kind == "normal_txt":
            split_frames = load_classic_normal_txt_capped(
                path, vehicle_model=vehicle, max_per_split=max_frames_per_split
            )
        elif kind == "classic_csv":
            split_frames = load_classic_attack_csv_capped(
                path,
                vehicle_model=vehicle,
                attack_type=atk,
                max_per_split=max_frames_per_split,
            )
        else:
            split_frames = load_challenge_csv_capped(
                path, vehicle_model=vehicle, max_per_split=max_frames_per_split
            )

        n_frames_total = sum(len(v) for v in split_frames.values())
        n_attack = sum(
            int((v["label"] > 0).sum()) for v in split_frames.values() if not v.empty
        )

        for split_name, part in split_frames.items():
            if part.empty:
                continue
            block_feats: list[pd.DataFrame] = []
            groups = list(part.groupby("block_id", sort=True))
            per_block_cap = max(50, max_windows_per_source // max(len(groups), 1))
            for block_id, block_df in groups:
                frames = block_df.drop(columns=["block_id"], errors="ignore")
                feats_b = windowize_trace(frames, max_windows=per_block_cap)
                if feats_b.empty:
                    continue
                feats_b = feats_b.copy()
                feats_b["block_id"] = int(block_id)
                block_feats.append(feats_b)
            if not block_feats:
                continue
            feats = pd.concat(block_feats, ignore_index=True)
            feats["window_id"] = np.arange(len(feats))
            feats["split"] = split_name
            feats["source_trace"] = path.name
            feats["physical_vehicle"] = vehicle
            window_parts.append(feats)
            print(
                f"  {split_name}: frames={len(part)} blocks={len(groups)} "
                f"windows={len(feats)} attack_windows={(feats['label'] == 1).sum()}",
                flush=True,
            )

        provenance.append(
            {
                "path": str(path),
                "kind": kind,
                "vehicle_model": vehicle,
                "n_frames_loaded": int(n_frames_total),
                "n_attack_frames_loaded": int(n_attack),
                "max_frames_per_split": max_frames_per_split,
                "split_protocol": "interleaved_contiguous_blocks_70_15_15",
            }
        )

    if not window_parts:
        raise RuntimeError("No windows extracted")

    windows = pd.concat(window_parts, ignore_index=True)
    windows["window_uid"] = (
        windows["physical_vehicle"].astype(str)
        + "::"
        + windows["source_trace"].astype(str)
        + "::"
        + windows["window_id"].astype(str)
        + "::"
        + windows["split"].astype(str)
    )

    scored_parts: list[pd.DataFrame] = []
    if_stats: list[dict[str, Any]] = []
    for vehicle, subset in windows.groupby("physical_vehicle"):
        train_benign = subset[(subset["split"] == "train") & (subset["label"] == 0)]
        if train_benign.empty:
            print(f"WARNING: no train benign for {vehicle}")
            continue
        X_benign = train_benign[BEHAVIOURAL_FEATURE_COLUMNS].fillna(0.0).to_numpy(
            dtype=np.float32
        )
        model = fit_self_supervised_isolation_forest(
            X_benign, random_state=seed, n_estimators=200
        )
        X_all = subset[BEHAVIOURAL_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
        _, scores = score_self_supervised_isolation_forest(model, X_all, X_benign)
        part = subset.copy()
        part["anomaly_score"] = np.clip(scores.astype(np.float64), 0.0, 1.0)
        part["local_alert"] = (part["anomaly_score"] >= strong_threshold).astype(int)
        part["weak_signal"] = (
            (part["anomaly_score"] >= weak_threshold)
            & (part["anomaly_score"] < strong_threshold)
        ).astype(int)
        scored_parts.append(part)
        if_stats.append(
            {
                "vehicle": str(vehicle),
                "n_train_benign": int(len(train_benign)),
                "n_scored": int(len(part)),
                "test_strong_attack": int(
                    (
                        (part["split"] == "test")
                        & (part["label"] == 1)
                        & (part["anomaly_score"] >= strong_threshold)
                    ).sum()
                ),
                "test_weak_attack": int(
                    (
                        (part["split"] == "test")
                        & (part["label"] == 1)
                        & (part["anomaly_score"] >= weak_threshold)
                        & (part["anomaly_score"] < strong_threshold)
                    ).sum()
                ),
            }
        )

    scored = pd.concat(scored_parts, ignore_index=True)
    scored["message_rate"] = scored["frame_count"].astype(np.float64)
    scored["burstiness"] = scored["std_inter_arrival_time"].astype(np.float64) / (
        scored["mean_inter_arrival_time"].astype(np.float64).abs() + 1e-9
    )
    scored["payload_entropy"] = compute_payload_entropy(scored)

    test_pool = scored[scored["split"] == "test"].copy()
    test_pool.to_csv(out_dir / "test_window_pool.csv", index=False)
    scored.to_csv(out_dir / "all_scored_windows.csv", index=False)

    summary = {
        "dataset": "OCSLab Car-Hacking (classic) + Car Hacking Challenge D/S",
        "strong_threshold": strong_threshold,
        "weak_threshold": weak_threshold,
        "max_frames_per_split": max_frames_per_split,
        "window_size": 100,
        "stride": 50,
        "split_protocol": "interleaved_contiguous_blocks_70_15_15",
        "n_total_scored_windows": int(len(scored)),
        "n_test_windows": int(len(test_pool)),
        "physical_vehicles": sorted(map(str, scored["physical_vehicle"].unique())),
        "sources": provenance,
        "if_stats": if_stats,
        "test_label_counts": {
            str(k): int(v) for k, v in test_pool["label"].value_counts().items()
        },
        "test_strong_malicious": int(
            (
                (test_pool["label"] == 1)
                & (test_pool["anomaly_score"] >= strong_threshold)
            ).sum()
        ),
        "test_weak_malicious": int(
            (
                (test_pool["label"] == 1)
                & (test_pool["anomaly_score"] >= weak_threshold)
                & (test_pool["anomaly_score"] < strong_threshold)
            ).sum()
        ),
        "test_suspicious_benign": int(
            (
                (test_pool["label"] == 0)
                & (test_pool["anomaly_score"] >= weak_threshold)
            ).sum()
        ),
        "graph_feature_cols": GNN_FEATURE_COLUMNS,
        "behavioural_feature_cols": list(BEHAVIOURAL_FEATURE_COLUMNS),
        "note_virtual_fleet": (
            "Classic OCSLab is one physical platform (OCSLab_Car). "
            "Challenge adds Challenge_D / Challenge_S. "
            "Ablation fleet nodes are virtual instances sampled from held-out TEST windows."
        ),
        "synthetic_descriptors": False,
    }
    (out_dir / "pool_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return test_pool


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--classic-root",
        type=Path,
        default=Path("/workspace/Dataset/ocslab_classic_only"),
    )
    p.add_argument(
        "--challenge-root", type=Path, default=Path("/workspace/Dataset/ocslab")
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path(__file__).resolve().parent / "artifacts"
    )
    p.add_argument("--max-frames-per-split", type=int, default=40000)
    p.add_argument("--max-windows-per-source", type=int, default=1200)
    args = p.parse_args()
    build_pool(
        args.classic_root,
        args.challenge_root if args.challenge_root.exists() else None,
        args.out_dir,
        max_frames_per_split=args.max_frames_per_split,
        max_windows_per_source=args.max_windows_per_source,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
