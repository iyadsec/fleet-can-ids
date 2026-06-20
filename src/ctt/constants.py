"""Constants for can-train-and-test cross-dataset validation."""

from __future__ import annotations

from pathlib import Path

# Dataset root (override via CTT_DATASET_ROOT env or runner --dataset-root)
DEFAULT_CTT_DATASET_ROOT = Path(
    "/workspace/Dataset/can-train-and-test"
)

OUTPUT_ROOT = Path("new_experiments/can_train_and_test_cross_dataset_validation")

OCSLAB_PUBLICATION_ROOT = Path("new_experiments/final_end_to_end_publication_run")

SETS = ("set_01", "set_02", "set_03", "set_04")

SUBSETS = (
    "train_01",
    "test_01_known_vehicle_known_attack",
    "test_02_unknown_vehicle_known_attack",
    "test_03_known_vehicle_unknown_attack",
    "test_04_unknown_vehicle_unknown_attack",
)

SET_VEHICLE_POLICY: dict[str, dict[str, str]] = {
    "set_01": {
        "known": "chevrolet_impala",
        "unknown": "chevrolet_silverado",
        "known_display": "Chevrolet Impala",
        "unknown_display": "Chevrolet Silverado",
        "known_manufacturer": "Chevrolet",
        "unknown_manufacturer": "Chevrolet",
    },
    "set_02": {
        "known": "chevrolet_traverse",
        "unknown": "subaru_forester",
        "known_display": "Chevrolet Traverse",
        "unknown_display": "Subaru Forester",
        "known_manufacturer": "Chevrolet",
        "unknown_manufacturer": "Subaru",
    },
    "set_03": {
        "known": "chevrolet_silverado",
        "unknown": "subaru_forester",
        "known_display": "Chevrolet Silverado",
        "unknown_display": "Subaru Forester",
        "known_manufacturer": "Chevrolet",
        "unknown_manufacturer": "Subaru",
    },
    "set_04": {
        "known": "subaru_forester",
        "unknown": "chevrolet_traverse",
        "known_display": "Subaru Forester",
        "unknown_display": "Chevrolet Traverse",
        "known_manufacturer": "Subaru",
        "unknown_manufacturer": "Chevrolet",
    },
}

ALL_VEHICLES = (
    "chevrolet_impala",
    "chevrolet_silverado",
    "chevrolet_traverse",
    "subaru_forester",
)

VEHICLE_DISPLAY = {
    "chevrolet_impala": "Chevrolet Impala",
    "chevrolet_silverado": "Chevrolet Silverado",
    "chevrolet_traverse": "Chevrolet Traverse",
    "subaru_forester": "Subaru Forester",
}

VEHICLE_MANUFACTURER = {
    "chevrolet_impala": "Chevrolet",
    "chevrolet_silverado": "Chevrolet",
    "chevrolet_traverse": "Chevrolet",
    "subaru_forester": "Subaru",
}

ATTACK_FAMILIES = (
    "dos",
    "fuzzing",
    "systematic",
    "gear_spoofing",
    "rpm_spoofing",
    "speed_spoofing",
    "combined_spoofing",
    "standstill",
    "interval",
    "benign",
)

# Filename stem (before -N suffix) -> (attack_type, attack_family)
ATTACK_FILENAME_MAP: dict[str, tuple[str, str]] = {
    "attack-free": ("benign", "benign"),
    "DoS": ("dos", "dos"),
    "fuzzing": ("fuzzing", "fuzzing"),
    "systematic": ("systematic", "systematic"),
    "force-neutral": ("gear_spoofing", "gear_spoofing"),
    "rpm": ("rpm_spoofing", "rpm_spoofing"),
    "speed": ("speed_spoofing", "speed_spoofing"),
    "rpm-accessory": ("combined_spoofing", "combined_spoofing"),
    "speed-accessory": ("combined_spoofing", "combined_spoofing"),
    "accessory": ("combined_spoofing", "combined_spoofing"),
    "double": ("combined_spoofing", "combined_spoofing"),
    "triple": ("combined_spoofing", "combined_spoofing"),
    "standstill": ("standstill", "standstill"),
    "interval": ("interval", "interval"),
}

NORMALIZED_COLUMNS = [
    "timestamp",
    "can_id",
    "dlc",
    "byte_0",
    "byte_1",
    "byte_2",
    "byte_3",
    "byte_4",
    "byte_5",
    "byte_6",
    "byte_7",
    "label",
    "is_attack",
    "attack_type",
    "vehicle_id",
    "manufacturer",
    "dataset_set",
    "subset_name",
    "source_file",
    "source_row_index",
]

WINDOW_SIZE = 100
WINDOW_STRIDE = 50
MIN_VALID_FRAMES = 80

SCENARIO_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]

SCENARIO_DISPLAY_NAMES = {
    "benign_fleet_control": "Benign fleet control",
    "isolated_attack": "Isolated attack",
    "unrelated_incidents": "Unrelated incidents",
    "strong_campaign": "Strong coordinated campaign",
    "weak_campaign": "Weak coordinated campaign",
}

WEAK_THRESHOLD_PERCENTILE = 90.0
STRONG_THRESHOLD_PERCENTILE = 97.5

GRAPH_SIMILARITY_THRESHOLD = 0.85
GRAPH_KNN_CAP = 10
GRAPH_CROSS_VEHICLE_CAP = 20
