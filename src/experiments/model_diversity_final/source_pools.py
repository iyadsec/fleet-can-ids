"""Build Phase 4 source pools from regenerated test descriptors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.campaign_analysis_corrected import STRONG_ATTACK_DEFAULT, WEAK_ATTACK_DEFAULT
from src.experiments.data_splits import is_benign_attack_type
from src.experiments.vehicle_instance_builder import build_instance_catalog


def build_final_source_pools(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    output_root: Path,
    *,
    weak_threshold: float = 0.55,
    strong_threshold: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = descriptors[descriptors["split"] == "test"].copy()
    desc_for_cat = descriptors.drop(columns=["split"], errors="ignore")
    catalog = build_instance_catalog(
        desc_for_cat, manifest, weak_threshold=weak_threshold, strong_threshold=strong_threshold, min_windows_per_segment=10
    )
    pools_dir = output_root / "source_pools"
    pools_dir.mkdir(parents=True, exist_ok=True)

    benign = test[test.attack_type.map(is_benign_attack_type)]
    benign.to_csv(pools_dir / "benign_pool.csv", index=False)
    strong_mal = test[(~test.attack_type.map(is_benign_attack_type)) & (test.anomaly_score >= strong_threshold)]
    weak_mal = test[
        (~test.attack_type.map(is_benign_attack_type))
        & (test.anomaly_score >= weak_threshold)
        & (test.anomaly_score < strong_threshold)
    ]
    strong_mal.to_csv(pools_dir / "strong_malicious_pool.csv", index=False)
    weak_mal.to_csv(pools_dir / "weak_malicious_pool.csv", index=False)

    inv_rows = []
    for vm in ("Hyundai", "Kia", "Chevrolet"):
        for atk in test[test.vehicle_model == vm]["attack_type"].unique():
            sub = test[(test.vehicle_model == vm) & (test.attack_type == atk)]
            is_ben = is_benign_attack_type(atk)
            inv_rows.append(
                {
                    "vehicle_model": vm,
                    "attack_type": atk,
                    "attack_strength": "benign" if is_ben else ("strong" if atk != "fuzzy" else "strong"),
                    "source_trace": sub.apply(lambda r: f"{r['vehicle_model']}::{r['source_file']}", axis=1).nunique(),
                    "available_descriptors": len(sub),
                    "independent_segment_count": int(catalog[catalog.vehicle_model == vm].shape[0]) if not is_ben else int(catalog[catalog.vehicle_model == vm].shape[0]),
                    "eligible_for_attacked_vehicle": not is_ben and len(sub) >= 5,
                    "eligible_for_benign_vehicle": is_ben and len(sub) >= 10,
                    "exclusion_reason": "" if len(sub) > 0 else "no_test_descriptors",
                }
            )
    inventory = pd.DataFrame(inv_rows)
    inventory.to_csv(output_root / "manifests/final_source_pool_inventory.csv", index=False)
    return benign, strong_mal, weak_mal, inventory
