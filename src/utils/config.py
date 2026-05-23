"""Load and merge YAML configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import resolve_project_root


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file; paths in the file remain relative to project root."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = resolve_project_root() / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {config_path}")
    return data


def get_nested(cfg: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely traverse nested config keys."""
    node: Any = cfg
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
