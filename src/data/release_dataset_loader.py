"""Consolidate headerless OCS Lab Car-Hacking release CAN logs from data/raw/release/."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

RAW_COLUMNS: list[str] = [
    "raw_index",
    "timestamp",
    "can_id",
    "dlc",
    "payload",
    "raw_label",
]

BYTE_COLUMNS: list[str] = [f"byte{i}" for i in range(8)]

RELEASE_STANDARD_COLUMNS: list[str] = [
    "timestamp",
    "can_id",
    "dlc",
    *BYTE_COLUMNS,
    "raw_label",
    "label",
    "attack_type",
    "vehicle_model",
    "source_file",
]

SUPPORTED_EXTENSIONS = {".csv", ".txt"}

ATTACK_FILENAME_RULES: list[tuple[str, str]] = [
    ("flooding", "Flooding"),
    ("fuzzy", "Fuzzy"),
    ("replay", "Replay"),
    ("malfunction", "Malfunction"),
]

VEHICLE_FOLDER_HINTS: dict[str, str] = {
    "hyundai": "Hyundai",
    "kia": "Kia",
    "chevrolet": "Chevrolet",
    "chevy": "Chevrolet",
    "sonata": "Hyundai",
    "soul": "Kia",
    "spark": "Chevrolet",
}


@dataclass
class ReleaseConsolidationResult:
    dataframe: pd.DataFrame
    n_source_files: int = 0
    n_files_skipped: int = 0
    n_invalid_rows_skipped: int = 0
    skipped_files: list[str] = field(default_factory=list)


def _relative_release_path(path: Path, release_root: Path) -> str:
    """Return path relative to release root without following symlinks outside it."""
    root = release_root.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.name


def discover_release_files(release_root: Path) -> list[Path]:
    """Recursively collect CSV/TXT files under *release_root*."""
    root = release_root.resolve()
    if not root.exists():
        logger.warning("Release root does not exist: %s", root)
        return []
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)
    return files


def infer_vehicle_model(path: Path, release_root: Path) -> str:
    """Infer vehicle from parent folder name, else Unknown."""
    try:
        rel = path.resolve().relative_to(release_root.resolve())
        candidates = [rel.parent.name, *rel.parent.parts]
    except ValueError:
        candidates = [path.parent.name, *path.parent.parts]

    for name in candidates:
        key = str(name).lower().replace("-", "_")
        if key in VEHICLE_FOLDER_HINTS:
            return VEHICLE_FOLDER_HINTS[key]
        for hint, model in VEHICLE_FOLDER_HINTS.items():
            if hint in key:
                return model
    return "Unknown"


def infer_attack_type_from_filename(path: Path) -> str | None:
    """Return Title-case attack type when filename contains a known token."""
    stem = path.stem.lower().replace("-", "_")
    for token, attack_type in ATTACK_FILENAME_RULES:
        if token in stem:
            return attack_type
    return None


def resolve_attack_type(path: Path, raw_label: str) -> str:
    """Row-level attack_type using filename first, then raw_label fallback."""
    from_name = infer_attack_type_from_filename(path)
    if from_name is not None:
        return from_name
    if str(raw_label).strip() == "R":
        return "Normal"
    return "Unknown"


def numeric_label(raw_label: str) -> int:
    """R = benign (0); T and other non-R labels = attack (1)."""
    return 0 if str(raw_label).strip() == "R" else 1


def _split_payload_bytes(payload: str) -> list[int] | None:
    """Parse exactly eight space-separated hex bytes into integers 0-255."""
    if payload is None or (isinstance(payload, float) and np.isnan(payload)):
        return None
    tokens = str(payload).strip().split()
    if len(tokens) != 8:
        return None
    values: list[int] = []
    for token in tokens:
        try:
            value = int(token, 16)
        except ValueError:
            return None
        if value < 0 or value > 255:
            return None
        values.append(value)
    return values


def _validate_row(row: pd.Series) -> bool:
    """Return True when required fields and payload bytes are valid."""
    for col in ("timestamp", "can_id", "dlc", "payload", "raw_label"):
        if pd.isna(row[col]) or str(row[col]).strip() == "":
            return False

    try:
        dlc = int(row["dlc"])
    except (TypeError, ValueError):
        return False
    if dlc < 0 or dlc > 8:
        return False

    return _split_payload_bytes(row["payload"]) is not None


def _read_raw_release_file(path: Path) -> pd.DataFrame:
    """Read headerless release CAN log with six fixed columns."""
    try:
        df = pd.read_csv(path, header=None, names=RAW_COLUMNS, dtype=str, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=RAW_COLUMNS)
    return df


def _standardize_release_file(
    path: Path,
    release_root: Path,
) -> tuple[pd.DataFrame, int]:
    """Parse one raw release file into standardized rows."""
    raw = _read_raw_release_file(path)
    if raw.empty:
        return pd.DataFrame(columns=RELEASE_STANDARD_COLUMNS), 0

    source_file = _relative_release_path(path, release_root)
    vehicle_model = infer_vehicle_model(path, release_root)
    filename_attack = infer_attack_type_from_filename(path)

    valid_rows: list[dict[str, object]] = []
    invalid_count = 0

    for _, row in raw.iterrows():
        if not _validate_row(row):
            invalid_count += 1
            continue

        payload_bytes = _split_payload_bytes(row["payload"])
        assert payload_bytes is not None

        raw_label = str(row["raw_label"]).strip()
        attack_type = (
            filename_attack if filename_attack is not None else resolve_attack_type(path, raw_label)
        )

        record: dict[str, object] = {
            "timestamp": row["timestamp"],
            "can_id": row["can_id"],
            "dlc": int(row["dlc"]),
            "raw_label": raw_label,
            "label": numeric_label(raw_label),
            "attack_type": attack_type,
            "vehicle_model": vehicle_model,
            "source_file": source_file,
        }
        for idx, value in enumerate(payload_bytes):
            record[f"byte{idx}"] = value
        valid_rows.append(record)

    if not valid_rows:
        return pd.DataFrame(columns=RELEASE_STANDARD_COLUMNS), invalid_count

    return pd.DataFrame(valid_rows, columns=RELEASE_STANDARD_COLUMNS), invalid_count


def consolidate_release_dataset(release_root: Path | str) -> ReleaseConsolidationResult:
    """Load and merge all release files under *release_root*."""
    root = Path(release_root).resolve()
    files = discover_release_files(root)
    frames: list[pd.DataFrame] = []
    skipped: list[str] = []
    total_invalid = 0

    for path in files:
        display = _relative_release_path(path, root)
        logger.info("Processing %s", display)

        frame, invalid = _standardize_release_file(path, root)
        total_invalid += invalid
        if frame.empty:
            skipped.append(display)
            continue
        frames.append(frame)

    if not frames:
        return ReleaseConsolidationResult(
            dataframe=pd.DataFrame(columns=RELEASE_STANDARD_COLUMNS),
            n_source_files=0,
            n_files_skipped=len(skipped),
            n_invalid_rows_skipped=total_invalid,
            skipped_files=skipped,
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(["source_file", "timestamp"]).reset_index(drop=True)

    return ReleaseConsolidationResult(
        dataframe=merged,
        n_source_files=len(frames),
        n_files_skipped=len(skipped),
        n_invalid_rows_skipped=total_invalid,
        skipped_files=skipped,
    )


def save_clean_release_dataset(df: pd.DataFrame, output_path: Path | str) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    logger.info("Saved %d release rows to %s", len(df), out)
    return out


def save_release_summaries(
    result: ReleaseConsolidationResult,
    metrics_dir: Path | str,
) -> dict[str, Path]:
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = result.dataframe
    written: dict[str, Path] = {}

    summary_path = out_dir / "release_dataset_summary.txt"
    if df.empty:
        summary_path.write_text("No release data consolidated.\n", encoding="utf-8")
        written["release_dataset_summary"] = summary_path
        return written

    src = (
        df.groupby("source_file", dropna=False)
        .size()
        .reset_index(name="record_count")
        .sort_values("record_count", ascending=False)
    )
    veh = (
        df.groupby("vehicle_model", dropna=False)
        .size()
        .reset_index(name="record_count")
        .sort_values("record_count", ascending=False)
    )
    atk = (
        df.groupby("attack_type", dropna=False)
        .size()
        .reset_index(name="record_count")
        .sort_values("record_count", ascending=False)
    )
    lbl = (
        df.groupby("raw_label", dropna=False)
        .size()
        .reset_index(name="record_count")
        .sort_values("record_count", ascending=False)
    )

    src_path = out_dir / "release_source_file_counts.csv"
    veh_path = out_dir / "release_vehicle_distribution.csv"
    atk_path = out_dir / "release_attack_distribution.csv"
    lbl_path = out_dir / "release_label_distribution.csv"
    src.to_csv(src_path, index=False)
    veh.to_csv(veh_path, index=False)
    atk.to_csv(atk_path, index=False)
    lbl.to_csv(lbl_path, index=False)
    written.update(
        {
            "release_source_file_counts": src_path,
            "release_vehicle_distribution": veh_path,
            "release_attack_distribution": atk_path,
            "release_label_distribution": lbl_path,
        }
    )

    lines = [
        "Release Dataset Consolidation Summary",
        "=" * 44,
        f"Source files processed:      {result.n_source_files}",
        f"Source files skipped:        {result.n_files_skipped}",
        f"Total rows consolidated:     {len(df):,}",
        f"Invalid rows skipped:        {result.n_invalid_rows_skipped:,}",
        "",
        "Rows per source_file:",
    ]
    for _, row in src.iterrows():
        lines.append(f"  {row['source_file']}: {int(row['record_count']):,}")
    lines.extend(["", "Rows per vehicle_model:"])
    for _, row in veh.iterrows():
        lines.append(f"  {row['vehicle_model']}: {int(row['record_count']):,}")
    lines.extend(["", "Rows per attack_type:"])
    for _, row in atk.iterrows():
        lines.append(f"  {row['attack_type']}: {int(row['record_count']):,}")
    lines.extend(["", "Rows per raw_label:"])
    for _, row in lbl.iterrows():
        lines.append(f"  {row['raw_label']}: {int(row['record_count']):,}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written["release_dataset_summary"] = summary_path
    return written


def print_release_consolidation_summary(result: ReleaseConsolidationResult) -> None:
    df = result.dataframe
    print("\n=== Release Dataset Consolidation ===")
    print(f"Files processed:           {result.n_source_files}")
    print(f"Total rows generated:      {len(df):,}")
    print(f"Invalid rows skipped:      {result.n_invalid_rows_skipped:,}")
    if not df.empty:
        print("\nRows per source_file:")
        for name, count in df.groupby("source_file").size().items():
            print(f"  {name}: {count:,}")
        print("\nRows per vehicle_model:")
        for name, count in df.groupby("vehicle_model").size().items():
            print(f"  {name}: {count:,}")
        print("\nRows per attack_type:")
        for name, count in df.groupby("attack_type").size().items():
            print(f"  {name}: {count:,}")
        print("\nRows per raw_label:")
        for name, count in df.groupby("raw_label").size().items():
            print(f"  {name}: {count:,}")
    print("=====================================\n")
