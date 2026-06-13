"""Raw and processed benign data inventory for corrected Phase 4 audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.data_splits import is_benign_attack_type, build_split_manifest
from src.experiments.scenario_generator import load_descriptor_tables
from src.experiments.vehicle_instance_builder import build_instance_catalog, source_trace_name


def inventory_raw_benign(dataset_roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in dataset_roots:
        if not root.exists():
            continue
        partition = root.name
        for f in sorted(root.glob("*.csv")):
            name = f.name
            low = name.lower()
            vm = (
                "Hyundai" if "hyundai" in low or "hy_sonata" in low
                else "Kia" if "kia" in low
                else "Chevrolet" if "chevrolet" in low or "spark" in low
                else "unknown"
            )
            is_benign_file = "attack_free" in low or "attack-free" in low
            try:
                nrows = sum(1 for _ in open(f)) - 1
            except OSError:
                nrows = 0
            rows.append(
                {
                    "vehicle_model": vm,
                    "dataset_partition": partition,
                    "folder": str(root),
                    "source_file": name,
                    "raw_label_value": "attack_free_filename",
                    "interpreted_label": "benign" if is_benign_file else "attack",
                    "row_count": nrows,
                    "timestamp_start": "",
                    "timestamp_end": "",
                    "labelled": True,
                    "eligible_for_use": is_benign_file or True,
                    "exclusion_reason": "" if is_benign_file else "attack trace",
                }
            )
    return pd.DataFrame(rows)


def pipeline_stage_counts(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    manifest_label: str,
) -> pd.DataFrame:
    rows = []
    join = descriptors.merge(
        manifest[["window_id", "vehicle_model", "source_file", "split"]].drop_duplicates(),
        on=["window_id", "vehicle_model", "source_file"],
        how="left",
    )
    for vm in ("Hyundai", "Kia", "Chevrolet"):
        sub = join[join.vehicle_model == vm]
        ben = sub[sub.attack_type.map(is_benign_attack_type)]
        test_ben = ben[ben.split == "test"]
        cat_ben = catalog[
            (catalog.vehicle_model == vm) & catalog.attack_types_available.eq("benign")
        ]
        rows.append(
            {
                "vehicle_model": vm,
                "manifest": manifest_label,
                "descriptor_rows": len(sub),
                "benign_descriptor_rows": len(ben),
                "test_benign_descriptor_rows": len(test_ben),
                "eligible_benign_instances": len(cat_ben),
                "benign_instances_ge5": int((cat_ben.benign_events >= 5).sum()) if not cat_ben.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def build_all_model_benign_pool(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    weak_th: float,
    strong_th: float,
) -> pd.DataFrame:
    join = descriptors.merge(
        manifest[["window_id", "vehicle_model", "source_file", "split"]].drop_duplicates(),
        on=["window_id", "vehicle_model", "source_file"],
        how="left",
    )
    ben = join[join.attack_type.map(is_benign_attack_type)].copy()
    ben["source_trace"] = ben["source_file"].map(source_trace_name)
    ben["ground_truth_malicious"] = 0
    ben["normalized_attack_type"] = ben["attack_type"]
    ben["descriptor_eligible"] = ben["split"] == "test"
    ben["exclusion_reason"] = ben["descriptor_eligible"].map(lambda x: "" if x else "not_test_split")
    ben["local_evidence_level"] = ben.get("evidence_level", "benign")
    ben["scenario_vehicle_id"] = ""
    ben["vehicle_token"] = ""
    ben["source_segment"] = ""
    cols = [
        "event_id", "scenario_vehicle_id", "vehicle_token", "vehicle_model",
        "source_file", "source_trace", "source_segment", "split",
        "ground_truth_malicious", "normalized_attack_type", "anomaly_score",
        "local_evidence_level", "descriptor_eligible", "exclusion_reason",
    ]
    for c in cols:
        if c not in ben.columns:
            ben[c] = ""
    return ben[cols]
