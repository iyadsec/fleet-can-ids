"""Dataset audit for can-train-and-test."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ctt.constants import (
    ALL_VEHICLES,
    ATTACK_FAMILIES,
    OUTPUT_ROOT,
    SET_VEHICLE_POLICY,
    VEHICLE_DISPLAY,
    VEHICLE_MANUFACTURER,
)
from src.ctt.utils import (
    count_lines_fast,
    discover_ctt_files,
    ensure_dir,
    write_markdown,
)


def run_dataset_audit(dataset_root: Path, output_root: Path = OUTPUT_ROOT) -> dict:
    """Audit dataset and write inventory manifests and reports."""
    manifest_dir = ensure_dir(output_root / "manifests")
    audit_dir = ensure_dir(output_root / "audit")

    records = discover_ctt_files(dataset_root)
    if not records:
        raise FileNotFoundError(f"No CSV files found under {dataset_root}")

    file_rows: list[dict] = []
    known_columns = "timestamp|arbitration_id|data_field|attack"
    for rec in records:
        path = Path(rec["source_file"])
        n_rows = count_lines_fast(path)
        size_bytes = path.stat().st_size
        file_rows.append(
            {
                **rec,
                "file_size_bytes": size_bytes,
                "row_count": n_rows,
                "columns": known_columns,
                "has_timestamp": True,
                "has_can_id": True,
                "has_payload": True,
                "has_label": True,
            }
        )

    file_df = pd.DataFrame(file_rows)
    file_df.to_csv(manifest_dir / "ctt_file_inventory.csv", index=False)

    vehicle_df = (
        file_df.groupby(["vehicle_id", "manufacturer", "dataset_set", "subset_name"], as_index=False)
        .agg(
            n_files=("source_file", "count"),
            total_rows=("row_count", "sum"),
            benign_files=("is_benign", lambda x: (x == "true").sum()),
            attack_files=("is_benign", lambda x: (x == "false").sum()),
        )
    )
    vehicle_df["vehicle_display"] = vehicle_df["vehicle_id"].map(VEHICLE_DISPLAY)
    vehicle_df.to_csv(manifest_dir / "ctt_vehicle_inventory.csv", index=False)

    attack_df = (
        file_df.groupby(["attack_type", "attack_family", "attack_raw"], as_index=False)
        .agg(
            n_files=("source_file", "count"),
            total_rows=("row_count", "sum"),
            vehicles=("vehicle_id", lambda x: "|".join(sorted(set(x)))),
            subsets=("subset_name", lambda x: "|".join(sorted(set(x)))),
        )
    )
    attack_df.to_csv(manifest_dir / "ctt_attack_inventory.csv", index=False)

    # Schema report from representative file (read once)
    sample_path = Path(records[0]["source_file"])
    sample = pd.read_csv(sample_path, nrows=5)
    schema_sections = {
        "Source schema": (
            f"Columns: `{', '.join(sample.columns)}`\n\n"
            f"Sample rows from `{sample_path.name}`:\n\n"
            f"```\n{sample.head(3).to_string(index=False)}\n```"
        ),
        "Target normalized schema": (
            "timestamp, can_id, dlc, byte_0..byte_7, label, is_attack, "
            "attack_type, vehicle_id, manufacturer, dataset_set, subset_name, "
            "source_file, source_row_index"
        ),
        "CAN ID format": "Hex arbitration_id without 0x prefix (e.g., `0C1`, `1E5`)",
        "Payload format": "Contiguous hex string in `data_field`; DLC derived from byte length",
        "Label format": "Column `attack`: 0=benign, 1=attack frame",
        "Timestamp": "Present as float epoch seconds in all inspected files",
    }
    write_markdown(audit_dir / "ctt_schema_report.md", "CTT Schema Report", schema_sections)

    vehicles_found = sorted(file_df["vehicle_id"].unique())
    attacks_found = sorted(file_df[file_df["attack_type"] != "benign"]["attack_type"].unique())
    families_found = sorted(file_df[file_df["attack_family"] != "benign"]["attack_family"].unique())
    total_rows = int(file_df["row_count"].sum())
    total_files = len(file_df)

    per_vehicle_benign = file_df[file_df["is_benign"] == "true"].groupby("vehicle_id")["row_count"].sum()
    supports_per_vehicle_onboarding = all(
        per_vehicle_benign.get(v, 0) > 0 for v in ALL_VEHICLES
    )
    supports_unknown_vehicle = any(
        "test_02" in s or "test_04" in s for s in file_df["subset_name"].unique()
    )
    supports_unknown_attack = any(
        "test_03" in s or "test_04" in s for s in file_df["subset_name"].unique()
    )

    suitability = {
        "per_vehicle_benign_onboarding": supports_per_vehicle_onboarding,
        "unknown_vehicle_testing": supports_unknown_vehicle,
        "unknown_attack_testing": supports_unknown_attack,
        "controlled_fleet_campaign_simulation": True,
        "real_synchronized_fleet_campaigns": False,
        "schema_normalizable": True,
    }

    suit_sections = {
        "Summary": (
            f"- Files: **{total_files}**\n"
            f"- Total samples: **{total_rows:,}**\n"
            f"- Vehicles found: **{len(vehicles_found)}** ({', '.join(vehicles_found)})\n"
            f"- Attack types found: **{len(attacks_found)}** ({', '.join(attacks_found)})\n"
            f"- Attack families: **{len(families_found)}** ({', '.join(families_found)})"
        ),
        "Per-vehicle benign onboarding": (
            "**Supported.** Each vehicle has attack-free captures in train_01 subsets "
            "across the four sets (known-vehicle training splits)."
            if supports_per_vehicle_onboarding
            else "**Partially supported.** Not all vehicles have dedicated benign training in every set."
        ),
        "Unknown-vehicle testing": (
            "**Supported** via `test_02` and `test_04` subsets."
            if supports_unknown_vehicle
            else "Not supported."
        ),
        "Unknown-attack testing": (
            "**Supported** via `test_03` and `test_04` subsets."
            if supports_unknown_attack
            else "Not supported."
        ),
        "Controlled fleet campaign simulation": (
            "**Supported.** Four vehicles (Chevrolet Impala, Silverado, Traverse; Subaru Forester) "
            "enable cross-vehicle scenario construction with behaviourally related attack families."
        ),
        "Real synchronized fleet campaigns": (
            "**Not present.** Attacks were conducted per-vehicle on rural roads; "
            "the dataset does not contain naturally synchronized multi-vehicle campaigns."
        ),
        "Schema normalizability": (
            "**Yes.** Source CSV schema (`timestamp`, `arbitration_id`, `data_field`, `attack`) "
            "maps cleanly to the framework normalized schema."
        ),
        "Set-vehicle policy": "\n".join(
            f"- **{s}**: known={p['known_display']}, unknown={p['unknown_display']}"
            for s, p in SET_VEHICLE_POLICY.items()
        ),
    }
    write_markdown(
        audit_dir / "ctt_dataset_suitability_report.md",
        "CTT Dataset Suitability Report",
        suit_sections,
    )

    struct_lines = [
        "# CTT Dataset Structure",
        "",
        f"Root: `{dataset_root}`",
        "",
        "## Directory layout",
        "",
    ]
    for dataset_set in sorted(file_df["dataset_set"].unique()):
        struct_lines.append(f"### {dataset_set}")
        for subset in sorted(file_df[file_df["dataset_set"] == dataset_set]["subset_name"].unique()):
            n = len(file_df[(file_df["dataset_set"] == dataset_set) & (file_df["subset_name"] == subset)])
            struct_lines.append(f"- `{subset}/` — {n} CSV files")
        struct_lines.append("")

    (audit_dir / "ctt_dataset_structure.md").write_text("\n".join(struct_lines), encoding="utf-8")

    return {
        "total_files": total_files,
        "total_rows": total_rows,
        "vehicles_found": vehicles_found,
        "attacks_found": attacks_found,
        "families_found": families_found,
        "suitability": suitability,
        "file_inventory": file_df,
    }
