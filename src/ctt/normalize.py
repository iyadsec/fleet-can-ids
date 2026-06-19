"""Normalize can-train-and-test CSV files to framework schema."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ctt.constants import NORMALIZED_COLUMNS, OUTPUT_ROOT
from src.ctt.utils import (
    discover_ctt_files,
    ensure_dir,
    parse_attack_from_filename,
    parse_data_field,
    vehicle_for_subset,
    write_markdown,
)


def normalize_file(
    source_path: Path,
    dataset_set: str,
    subset_name: str,
    output_dir: Path,
) -> dict:
    """Normalize one CTT CSV file."""
    _, attack_type, _ = parse_attack_from_filename(source_path.name)
    vehicle_id, manufacturer = vehicle_for_subset(dataset_set, subset_name)

    rel_name = f"{dataset_set}/{subset_name}/{source_path.stem}.csv"
    out_path = output_dir / rel_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunksize = 500_000
    first = True
    total_rows = 0
    source_row_index = 0

    for chunk in pd.read_csv(source_path, chunksize=chunksize):
        rows: list[dict] = []
        for _, row in chunk.iterrows():
            dlc, byte_vals = parse_data_field(row.get("data_field", ""))
            can_id = str(row.get("arbitration_id", "")).strip().upper().replace("0X", "")
            attack_val = int(row.get("attack", 0))
            rows.append(
                {
                    "timestamp": float(row.get("timestamp", np.nan)),
                    "can_id": can_id,
                    "dlc": dlc,
                    "byte_0": byte_vals[0],
                    "byte_1": byte_vals[1],
                    "byte_2": byte_vals[2],
                    "byte_3": byte_vals[3],
                    "byte_4": byte_vals[4],
                    "byte_5": byte_vals[5],
                    "byte_6": byte_vals[6],
                    "byte_7": byte_vals[7],
                    "label": attack_val,
                    "is_attack": attack_val,
                    "attack_type": attack_type,
                    "vehicle_id": vehicle_id,
                    "manufacturer": manufacturer,
                    "dataset_set": dataset_set,
                    "subset_name": subset_name,
                    "source_file": str(source_path),
                    "source_row_index": source_row_index,
                }
            )
            source_row_index += 1

        part_df = pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)
        part_df.to_csv(out_path, mode="w" if first else "a", header=first, index=False)
        first = False
        total_rows += len(part_df)

    return {
        "source_file": str(source_path),
        "normalized_path": str(out_path),
        "dataset_set": dataset_set,
        "subset_name": subset_name,
        "vehicle_id": vehicle_id,
        "attack_type": attack_type,
        "row_count": total_rows,
    }


def run_normalization(
    dataset_root: Path,
    output_root: Path = OUTPUT_ROOT,
) -> pd.DataFrame:
    """Normalize all CTT files."""
    norm_dir = ensure_dir(output_root / "normalized")
    manifest_dir = ensure_dir(output_root / "manifests")
    audit_dir = ensure_dir(output_root / "audit")

    records = discover_ctt_files(dataset_root)
    manifest_rows: list[dict] = []

    for i, rec in enumerate(records):
        path = Path(rec["source_file"])
        result = normalize_file(path, rec["dataset_set"], rec["subset_name"], norm_dir)
        manifest_rows.append(result)
        if (i + 1) % 20 == 0:
            print(f"  Normalized {i + 1}/{len(records)} files...")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_dir / "normalization_manifest.csv", index=False)

    sections = {
        "Summary": (
            f"- Files normalized: **{len(manifest_df)}**\n"
            f"- Total rows: **{int(manifest_df['row_count'].sum()):,}**\n"
            f"- Output directory: `normalized/`"
        ),
        "Rules applied": (
            "- Preserved original timestamps\n"
            "- Derived DLC from hex payload length (capped at 8)\n"
            "- Padded missing payload bytes with NaN beyond DLC\n"
            "- Preserved original row order via `source_row_index`\n"
            "- Labels taken from source `attack` column (not inferred)\n"
            "- Vehicle assigned from set/subset policy (not from filename)"
        ),
        "Schema validation": (
            f"All normalized files use columns: `{', '.join(NORMALIZED_COLUMNS)}`"
        ),
    }
    write_markdown(audit_dir / "normalization_report.md", "Normalization Report", sections)
    return manifest_df
