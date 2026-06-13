"""Framework-level ablation (C1 local-only, C2 similarity-only, C3 graph-based)."""

from src.experiments.framework_ablation.config import (
    CONFIG_LABELS,
    FRAMEWORK_CONFIGS,
    METHOD_TO_FRAMEWORK,
    SCENARIO_MAP,
)

__all__ = [
    "CONFIG_LABELS",
    "FRAMEWORK_CONFIGS",
    "METHOD_TO_FRAMEWORK",
    "SCENARIO_MAP",
]
