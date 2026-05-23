"""Load raw CAN bus traces from disk."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.paths import ProjectPaths


# Expected columns after normalization (implement per dataset format)
CAN_COLUMNS = ("timestamp", "can_id", "dlc", "payload", "label")


def load_can_trace(
    path: str | Path,
    *,
    paths: ProjectPaths | None = None,
) -> pd.DataFrame:
    """
    Load a single CAN trace file into a normalized DataFrame.

    Supported formats will be added per dataset (CSV, ASC, etc.).
    Raises NotImplementedError until dataset-specific parsers are wired.
    """
    file_path = Path(path)
    if not file_path.is_absolute() and paths is not None:
        file_path = paths.raw_dir / file_path
    if not file_path.exists():
        raise FileNotFoundError(f"CAN trace not found: {file_path}")
    raise NotImplementedError(
        f"Parser for {file_path.suffix!r} not implemented. "
        "Add dataset-specific logic in src/data/loaders.py."
    )
