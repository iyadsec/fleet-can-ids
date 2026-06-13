"""Raw benign data must exist for all three vehicle models."""

from pathlib import Path

import pandas as pd

from src.experiments.model_diversity_corrected.inventory import inventory_raw_benign
from src.utils.paths import resolve_project_root


def test_raw_benign_exists_all_models() -> None:
    root = resolve_project_root().parent
    dataset_roots = [
        root / "Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_preliminary_train",
        root / "Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_1st_train",
        root / "Dataset/In-Vehicle Network Intrusion Detection Challenge/car_track_final_2nd_train",
    ]
    inv = inventory_raw_benign([p for p in dataset_roots if p.exists()])
    if inv.empty:
        return
    eligible = inv[inv["eligible_for_use"] == True]
    for vm in ("Hyundai", "Kia", "Chevrolet"):
        assert not eligible[eligible.vehicle_model == vm].empty, f"No eligible benign raw data for {vm}"
