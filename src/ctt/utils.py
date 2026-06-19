"""Shared utilities for CTT pipeline."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ctt.constants import ATTACK_FILENAME_MAP, SET_VEHICLE_POLICY, SUBSETS


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def parse_attack_from_filename(filename: str) -> tuple[str, str, str]:
    """Return (raw_stem, attack_type, attack_family)."""
    stem = Path(filename).stem
    m = re.match(r"^(.+)-(\d+)$", stem)
    raw = m.group(1) if m else stem
    if raw in ATTACK_FILENAME_MAP:
        atype, family = ATTACK_FILENAME_MAP[raw]
        return raw, atype, family
    return raw, "unknown", "unknown"


def vehicle_for_subset(dataset_set: str, subset_name: str) -> tuple[str, str]:
    """Return (vehicle_id, manufacturer) for a file in given set/subset."""
    policy = SET_VEHICLE_POLICY[dataset_set]
    if subset_name in ("train_01", "test_01_known_vehicle_known_attack", "test_03_known_vehicle_unknown_attack"):
        vid = policy["known"]
        mfr = policy["known_manufacturer"]
    else:
        vid = policy["unknown"]
        mfr = policy["unknown_manufacturer"]
    return vid, mfr


def is_training_subset(subset_name: str) -> bool:
    return subset_name == "train_01"


def is_test_subset(subset_name: str) -> bool:
    return subset_name.startswith("test_")


def subset_condition(subset_name: str) -> tuple[str, str]:
    """Return (vehicle_condition, attack_condition) labels."""
    mapping = {
        "train_01": ("known", "mixed"),
        "test_01_known_vehicle_known_attack": ("known", "known"),
        "test_02_unknown_vehicle_known_attack": ("unknown", "known"),
        "test_03_known_vehicle_unknown_attack": ("known", "unknown"),
        "test_04_unknown_vehicle_unknown_attack": ("unknown", "unknown"),
    }
    return mapping.get(subset_name, ("unknown", "unknown"))


def parse_data_field(data_field: str) -> tuple[int, list[float]]:
    """Parse hex data_field into DLC and byte values."""
    if not data_field or (isinstance(data_field, float) and np.isnan(data_field)):
        return 0, [np.nan] * 8
    s = str(data_field).strip().upper()
    if s.startswith("0X"):
        s = s[2:]
    nbytes = len(s) // 2
    dlc = min(nbytes, 8)
    bytes_out: list[float] = []
    for i in range(8):
        if i < dlc:
            pair = s[i * 2 : i * 2 + 2]
            try:
                bytes_out.append(float(int(pair, 16)))
            except ValueError:
                bytes_out.append(np.nan)
        else:
            bytes_out.append(np.nan)
    return dlc, bytes_out


def discover_ctt_files(dataset_root: Path) -> list[dict[str, str]]:
    """Enumerate all CSV files with metadata."""
    records: list[dict[str, str]] = []
    for dataset_set in SET_VEHICLE_POLICY:
        for subset_name in SUBSETS:
            subdir = dataset_root / dataset_set / subset_name
            if not subdir.exists():
                continue
            for path in sorted(subdir.glob("*.csv")):
                raw, atype, family = parse_attack_from_filename(path.name)
                vid, mfr = vehicle_for_subset(dataset_set, subset_name)
                records.append(
                    {
                        "dataset_set": dataset_set,
                        "subset_name": subset_name,
                        "source_file": str(path),
                        "filename": path.name,
                        "attack_raw": raw,
                        "attack_type": atype,
                        "attack_family": family,
                        "vehicle_id": vid,
                        "manufacturer": mfr,
                        "is_benign": str(atype == "benign").lower(),
                        "is_training": str(is_training_subset(subset_name)).lower(),
                        "is_test": str(is_test_subset(subset_name)).lower(),
                    }
                )
    return records


def count_lines_fast(path: Path) -> int:
    """Count lines in a file (minus header)."""
    count = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return max(count - 1, 0)


def write_markdown(path: Path, title: str, sections: dict[str, str]) -> None:
    lines = [f"# {title}", ""]
    for heading, body in sections.items():
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(body.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b else default


def fmt_pvalue(p: float) -> str:
    if p < 1e-300:
        return "< 1e-300"
    if p == 0.0:
        return "< 1e-300"
    return f"{p:.4g}"
