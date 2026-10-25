"""CAN trace loading and preprocessing."""

from src.data.dataset_loader import load_and_merge, print_dataset_statistics
from src.data.fleetcan_adapter import (
    default_pilot_sessions,
    load_sessions as load_fleetcan_sessions,
    validate_dataset_coverage,
)
from src.data.loaders import load_can_trace
from src.data.preprocessing import preprocess_trace
from src.data.validate_dataset import run_validation, save_validation_report

__all__ = [
    "load_and_merge",
    "print_dataset_statistics",
    "load_can_trace",
    "preprocess_trace",
    "run_validation",
    "save_validation_report",
    "load_fleetcan_sessions",
    "default_pilot_sessions",
    "validate_dataset_coverage",
]
