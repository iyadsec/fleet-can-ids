"""Run configuration for staged CTT cross-dataset validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.ctt.constants import OUTPUT_ROOT

Stage = Literal["audit", "pilot", "set_pilot", "full"]

# Pilot defaults (set_01 train + known/known test)
PILOT_DATASET_SET = "set_01"
PILOT_TRAIN_SUBSET = "train_01"
PILOT_TEST_SUBSET = "test_01_known_vehicle_known_attack"

# Set pilot defaults (one complete set, moderate safety caps)
SET_PILOT_DEFAULT_SET = "set_01"
SET_PILOT_DEFAULT_MAX_ROWS = 475_000
SET_PILOT_DEFAULT_MAX_WINDOWS = 400_000
SET_PILOT_DEFAULT_MAX_DESCRIPTORS = 50_000


@dataclass
class RunConfig:
    """Controls stage, resource caps, and resume behaviour."""

    stage: Stage = "audit"
    dataset_root: Path = field(default_factory=lambda: Path("/workspace/Dataset/can-train-and-test"))
    output_root: Path = field(default_factory=lambda: OUTPUT_ROOT)

    max_files: int | None = None
    max_rows_per_file: int | None = None
    max_windows: int | None = None
    max_graph_nodes: int | None = None
    max_descriptors: int | None = None

    resume: bool = False
    skip_existing: bool = False
    confirm_large_run: bool = False

    # Pilot / set_pilot scope
    pilot_dataset_set: str = PILOT_DATASET_SET
    pilot_train_subset: str = PILOT_TRAIN_SUBSET
    pilot_test_subset: str = PILOT_TEST_SUBSET
    set_id: str | None = None

    @classmethod
    def for_stage(cls, stage: Stage, **kwargs) -> RunConfig:
        """Factory with stage-appropriate defaults."""
        cfg = cls(stage=stage, **kwargs)
        if stage == "audit":
            cfg.max_files = None
            cfg.max_rows_per_file = None
            cfg.max_windows = 0
        elif stage == "pilot":
            cfg.max_files = kwargs.get("max_files") if kwargs.get("max_files") is not None else 20
            cfg.max_rows_per_file = (
                kwargs.get("max_rows_per_file") if kwargs.get("max_rows_per_file") is not None else 100_000
            )
            cfg.max_windows = kwargs.get("max_windows") if kwargs.get("max_windows") is not None else 20_000
            cfg.max_descriptors = (
                kwargs.get("max_descriptors") if kwargs.get("max_descriptors") is not None else 5_000
            )
            cfg.max_graph_nodes = (
                kwargs.get("max_graph_nodes") if kwargs.get("max_graph_nodes") is not None else 5_000
            )
        elif stage == "set_pilot":
            cfg.set_id = kwargs.get("set_id") or SET_PILOT_DEFAULT_SET
            if not cfg.confirm_large_run:
                cfg.max_rows_per_file = (
                    kwargs.get("max_rows_per_file")
                    if kwargs.get("max_rows_per_file") is not None
                    else SET_PILOT_DEFAULT_MAX_ROWS
                )
                cfg.max_windows = (
                    kwargs.get("max_windows")
                    if kwargs.get("max_windows") is not None
                    else SET_PILOT_DEFAULT_MAX_WINDOWS
                )
                cfg.max_descriptors = (
                    kwargs.get("max_descriptors")
                    if kwargs.get("max_descriptors") is not None
                    else SET_PILOT_DEFAULT_MAX_DESCRIPTORS
                )
            else:
                cfg.max_rows_per_file = kwargs.get("max_rows_per_file")
                cfg.max_windows = kwargs.get("max_windows")
                cfg.max_descriptors = kwargs.get("max_descriptors")
            cfg.max_graph_nodes = kwargs.get("max_graph_nodes")
            cfg.max_files = kwargs.get("max_files")
        elif stage == "full":
            cfg.max_files = kwargs.get("max_files")
            cfg.max_rows_per_file = kwargs.get("max_rows_per_file")
            cfg.max_windows = kwargs.get("max_windows")
            cfg.max_descriptors = kwargs.get("max_descriptors")
            cfg.max_graph_nodes = kwargs.get("max_graph_nodes")
        return cfg

    def set_work_root(self) -> Path:
        if self.stage == "set_pilot" and self.set_id:
            return self.output_root / "set_pilot" / self.set_id
        return self.output_root

    def log_path(self) -> Path:
        root = self.set_work_root() if self.stage == "set_pilot" else self.output_root
        logs = root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        if self.stage == "set_pilot" and self.set_id:
            return logs / f"stage_set_pilot_{self.set_id}.log"
        return logs / f"stage_{self.stage}.log"

    def stage_marker_path(self) -> Path:
        if self.stage == "set_pilot" and self.set_id:
            return self.set_work_root() / "manifests" / f"stage_set_pilot_{self.set_id}_complete.json"
        return self.output_root / "manifests" / f"stage_{self.stage}_complete.json"

    def should_process_record(self, rec: dict) -> bool:
        """Filter files by stage scope."""
        if self.stage == "audit":
            return False
        if self.stage == "pilot":
            if rec["dataset_set"] != self.pilot_dataset_set:
                return False
            if rec["subset_name"] not in (self.pilot_train_subset, self.pilot_test_subset):
                return False
            if rec["subset_name"] == self.pilot_train_subset and rec["attack_type"] != "benign":
                return False
            return True
        if self.stage == "set_pilot":
            if rec["dataset_set"] != (self.set_id or SET_PILOT_DEFAULT_SET):
                return False
            if rec["subset_name"] == "train_01" and rec["attack_type"] != "benign":
                return False
            return True
        return True  # full: all files (subject to caps)

    def is_benign_train_file(self, rec: dict) -> bool:
        return rec["subset_name"].endswith("train_01") or rec["subset_name"] == "train_01"
