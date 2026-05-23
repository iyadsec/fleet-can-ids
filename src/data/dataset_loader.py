"""Load, standardize, and merge CAN bus datasets from external and local raw paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Default external dataset (Car Hacking / Car Track challenge layout)
DEFAULT_EXTERNAL_DATASET = Path(
    "/Users/iyadatieh/Library/CloudStorage/OneDrive-Personal/"
    "University of Reading/CodeRepo/Dataset"
)

STANDARD_COLUMNS: list[str] = [
    "timestamp",
    "can_id",
    "dlc",
    "byte0",
    "byte1",
    "byte2",
    "byte3",
    "byte4",
    "byte5",
    "byte6",
    "byte7",
    "label",
    "attack_type",
    "vehicle_model",
    "source_file",
]

BYTE_COLUMNS = [f"byte{i}" for i in range(8)]
SUPPORTED_EXTENSIONS = {".csv", ".txt"}
VEHICLE_FOLDER_MAP = {
    "hyundai": "Hyundai",
    "kia": "Kia",
    "chevrolet": "Chevrolet",
}
FLAG_VALUES = frozenset({"R", "T"})

# Filename token -> canonical attack_type (order matters; check benign first)
_ATTACK_TYPE_RULES: list[tuple[str, str]] = [
    ("attack_free", "attack_free"),
    ("attack-free", "attack_free"),
    ("flooding", "flooding"),
    ("fuzzy", "fuzzy"),
    ("malfunction", "malfunction"),
    ("replay", "replay"),
]


def discover_log_files(root: Path) -> list[Path]:
    """Recursively collect CSV/TXT CAN log paths under *root*."""
    if not root.exists():
        logger.warning("Data root does not exist: %s", root)
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path.resolve())
    return files


def _infer_vehicle_from_path(path: Path) -> str | None:
    """Resolve vehicle from folder name (data/raw/Hyundai/...) or filename tokens."""
    for part in path.parts:
        key = part.lower()
        if key in VEHICLE_FOLDER_MAP:
            return VEHICLE_FOLDER_MAP[key]

    name = path.stem.upper()
    if "CHEVROLET" in name or "_SPARK" in name:
        return "Chevrolet"
    if "HYUNDAI" in name or "HY_SONATA" in name or "_HY_" in name:
        return "Hyundai"
    if "KIA" in name:
        return "Kia"
    return None


def _infer_attack_type(path: Path) -> str | None:
    """Infer attack type from filename; None if unknown (e.g. anonymous release files)."""
    stem = path.stem.lower()
    for token, attack_type in _ATTACK_TYPE_RULES:
        if token in stem:
            return attack_type
    if re.match(r"file_\d+$", stem, re.I):
        return "release"
    return None


def _label_from_attack_type(attack_type: str | None) -> float:
    """Benign = 0, attack = 1, unknown = NaN."""
    if attack_type is None:
        return np.nan
    if attack_type == "attack_free":
        return 0.0
    if attack_type == "release":
        return np.nan
    return 1.0


def _parse_payload_bytes(payload: str, dlc: int | None) -> dict[str, float]:
    """Split space-separated hex payload into byte0..byte7; pad with NaN."""
    out: dict[str, float] = {col: np.nan for col in BYTE_COLUMNS}
    if not payload or (isinstance(payload, float) and np.isnan(payload)):
        return out

    tokens = str(payload).strip().split()
    limit = min(len(tokens), 8)
    if dlc is not None and not np.isnan(dlc):
        try:
            limit = min(limit, int(dlc))
        except (TypeError, ValueError):
            pass

    for idx, token in enumerate(tokens[:limit]):
        try:
            out[f"byte{idx}"] = float(int(token, 16))
        except ValueError:
            out[f"byte{idx}"] = np.nan
    return out


def _parse_line(line: str) -> dict[str, Any] | None:
    """Parse one CAN log line (train or release format)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 4:
        return None

    flag: str | None
    # Release: index,timestamp,can_id,dlc,payload,flag
    if parts[0].isdigit() and len(parts) >= 6:
        timestamp, can_id, dlc_str, payload = parts[1], parts[2], parts[3], parts[4]
        flag = parts[5] if parts[5] in FLAG_VALUES else None
    else:
        # Train: timestamp,can_id,dlc,payload[,flag]
        timestamp, can_id, dlc_str = parts[0], parts[1], parts[2]
        if len(parts) >= 5 and parts[-1] in FLAG_VALUES:
            payload, flag = parts[3], parts[-1]
        else:
            payload, flag = parts[3], None

    try:
        dlc = int(float(dlc_str))
    except (TypeError, ValueError):
        dlc = np.nan

    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        ts = np.nan

    can_id_norm = str(can_id).strip().upper().replace("0X", "")
    byte_vals = _parse_payload_bytes(payload, dlc if not np.isnan(dlc) else None)

    row: dict[str, Any] = {
        "timestamp": ts,
        "can_id": can_id_norm,
        "dlc": dlc,
        **byte_vals,
        "flag": flag,
    }
    return row


def _read_log_file(path: Path) -> pd.DataFrame:
    """Read a single CAN log file into a raw frame (no metadata columns yet)."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parsed = _parse_line(line)
            if parsed is not None:
                rows.append(parsed)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _attach_metadata(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Add label, attack_type, vehicle_model, source_file."""
    vehicle = _infer_vehicle_from_path(path)
    attack_type = _infer_attack_type(path)
    label = _label_from_attack_type(attack_type)

    df = df.copy()
    df["vehicle_model"] = vehicle
    df["attack_type"] = attack_type
    df["label"] = label
    df["source_file"] = str(path)
    if "flag" in df.columns:
        df = df.drop(columns=["flag"])
    return df


def _standardize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce column order, dtypes, and safe missing-value handling."""
    if df.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[STANDARD_COLUMNS]

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["dlc"] = pd.to_numeric(df["dlc"], errors="coerce").clip(lower=0, upper=8)
    df["can_id"] = df["can_id"].astype(str).str.strip().str.upper()

    for col in BYTE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Bytes beyond DLC are not valid payload
    dlc_int = df["dlc"].fillna(8).astype(int)
    for i, col in enumerate(BYTE_COLUMNS):
        mask = dlc_int <= i
        df.loc[mask, col] = np.nan

    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df["attack_type"] = df["attack_type"].astype("string").fillna("unknown")
    df["vehicle_model"] = df["vehicle_model"].astype("string")
    df["source_file"] = df["source_file"].astype("string")

    return df


def load_single_file(path: Path) -> pd.DataFrame:
    """Load and standardize one CAN log file."""
    vehicle = _infer_vehicle_from_path(path)
    attack_type = _infer_attack_type(path)

    # Skip anonymous competition release files without vehicle metadata
    if vehicle is None and attack_type == "release":
        logger.debug("Skipping release file without vehicle metadata: %s", path.name)
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    if vehicle is None:
        logger.warning("Could not infer vehicle for %s — skipping", path)
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    raw = _read_log_file(path)
    if raw.empty:
        logger.warning("No rows parsed from %s", path)
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    framed = _attach_metadata(raw, path)
    return _standardize_frame(framed)


def iter_load_files(paths: list[Path]) -> Iterator[pd.DataFrame]:
    """Yield standardized frames per file."""
    for path in paths:
        logger.info("Loading %s", path.name)
        yield load_single_file(path)


def load_and_merge(
    *,
    external_root: Path | str | None = DEFAULT_EXTERNAL_DATASET,
    raw_root: Path | str | None = None,
    vehicles: tuple[str, ...] = ("Hyundai", "Kia", "Chevrolet"),
) -> pd.DataFrame:
    """
    Load CAN logs from *external_root* first, then *raw_root*.

    Only files mapped to one of *vehicles* are included.
    """
    allowed = {v.lower() for v in vehicles}
    frames: list[pd.DataFrame] = []
    seen: set[Path] = set()

    roots: list[Path] = []
    if external_root is not None:
        roots.append(Path(external_root))
    if raw_root is not None:
        roots.append(Path(raw_root))

    for root in roots:
        for path in discover_log_files(root):
            if path in seen:
                continue
            seen.add(path)
            frame = load_single_file(path)
            if frame.empty:
                continue
            model = str(frame["vehicle_model"].iloc[0]).lower()
            if model not in allowed:
                continue
            frames.append(frame)

    if not frames:
        logger.warning("No CAN frames loaded from configured roots.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["vehicle_model", "attack_type", "timestamp"]).reset_index(
        drop=True
    )
    return merged


def save_clean_dataset(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Persist merged dataset to CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved %d rows to %s", len(df), out)
    return out


def dataset_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Summarize merged dataset for reporting."""
    if df.empty:
        return {
            "n_vehicles": 0,
            "n_attack_types": 0,
            "n_can_frames": 0,
            "vehicles": [],
            "attack_types": [],
        }

    attack_types = sorted(
        {
            str(v)
            for v in df["attack_type"].dropna().unique()
            if str(v) not in ("attack_free", "unknown", "release")
        }
    )
    return {
        "n_vehicles": int(df["vehicle_model"].nunique(dropna=True)),
        "n_attack_types": len(attack_types),
        "n_can_frames": int(len(df)),
        "vehicles": sorted(df["vehicle_model"].dropna().unique().tolist()),
        "attack_types": sorted(attack_types),
        "frames_by_vehicle": df.groupby("vehicle_model", dropna=False).size().to_dict(),
        "frames_by_attack": df.groupby("attack_type", dropna=False).size().to_dict(),
    }


def print_dataset_statistics(df: pd.DataFrame) -> None:
    """Print human-readable dataset summary."""
    stats = dataset_statistics(df)
    print("\n=== Dataset Statistics ===")
    print(f"Number of vehicles:    {stats['n_vehicles']}")
    print(f"Number of attack types: {stats['n_attack_types']}")
    print(f"Number of CAN frames:  {stats['n_can_frames']:,}")
    if stats["vehicles"]:
        print(f"  Vehicles: {', '.join(stats['vehicles'])}")
    if stats["attack_types"]:
        print(f"  Attack types: {', '.join(stats['attack_types'])}")
    if stats.get("frames_by_vehicle"):
        print("\n  Frames by vehicle:")
        for name, count in sorted(stats["frames_by_vehicle"].items()):
            print(f"    {name}: {count:,}")
    if stats.get("frames_by_attack"):
        print("\n  Frames by attack type:")
        for name, count in sorted(stats["frames_by_attack"].items()):
            print(f"    {name}: {count:,}")
    print("==========================\n")
