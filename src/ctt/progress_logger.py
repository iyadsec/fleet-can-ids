"""Progress logging for staged CTT pipeline runs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path


class ProgressLogger:
    """Logs progress at configured intervals to file and stdout."""

    ROW_INTERVAL = 1_000_000
    WINDOW_INTERVAL = 10_000

    def __init__(self, log_path: Path, stage: str) -> None:
        self.log_path = log_path
        self.stage = stage
        self._files_processed = 0
        self._rows_read = 0
        self._windows_generated = 0
        self._scenarios_completed = 0
        self._last_row_milestone = 0
        self._last_window_milestone = 0

        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(f"ctt.{stage}")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self._logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(sh)

    def _ts(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def stage_start(self) -> None:
        self.info(f"=== STAGE {self.stage.upper()} START @ {self._ts()} ===")

    def stage_end(self, status: str = "OK") -> None:
        self.info(
            f"=== STAGE {self.stage.upper()} END ({status}) | "
            f"files={self._files_processed} rows={self._rows_read:,} "
            f"windows={self._windows_generated:,} scenarios={self._scenarios_completed} ==="
        )

    def file_processed(self, path: str, rows: int = 0, windows: int = 0) -> None:
        self._files_processed += 1
        self._rows_read += rows
        self._windows_generated += windows
        self.info(
            f"[file {self._files_processed}] {Path(path).name} | "
            f"rows={rows:,} windows={windows:,} | cumulative rows={self._rows_read:,} windows={self._windows_generated:,}"
        )
        self._check_row_milestone()
        self._check_window_milestone()

    def rows_read(self, n: int) -> None:
        self._rows_read += n
        self._check_row_milestone()

    def windows_generated(self, n: int) -> None:
        self._windows_generated += n
        self._check_window_milestone()

    def scenario_completed(self, name: str, seed: int | None = None) -> None:
        self._scenarios_completed += 1
        seed_str = f" seed={seed}" if seed is not None else ""
        self.info(f"[scenario {self._scenarios_completed}] completed: {name}{seed_str}")

    def _check_row_milestone(self) -> None:
        milestone = (self._rows_read // self.ROW_INTERVAL) * self.ROW_INTERVAL
        if milestone > self._last_row_milestone and milestone > 0:
            self._last_row_milestone = milestone
            self.info(f"[progress] {milestone:,} rows read")

    def _check_window_milestone(self) -> None:
        milestone = (self._windows_generated // self.WINDOW_INTERVAL) * self.WINDOW_INTERVAL
        if milestone > self._last_window_milestone and milestone > 0:
            self._last_window_milestone = milestone
            self.info(f"[progress] {milestone:,} windows generated")
