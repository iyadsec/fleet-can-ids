"""Controlled scenario experiments for fleet-aware CAN IDS ablation study."""

from src.experiments.scenario_registry import SCENARIO_REGISTRY, ScenarioSpec, list_scenarios
from src.experiments.result_writer import ExperimentRunContext, ProtectedOutputGuard

__all__ = [
    "SCENARIO_REGISTRY",
    "ScenarioSpec",
    "list_scenarios",
    "ExperimentRunContext",
    "ProtectedOutputGuard",
]
