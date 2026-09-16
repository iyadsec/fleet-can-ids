"""Controlled experimental scenario definitions (S0–S4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

EvidenceLevel = Literal["none", "strong", "weak", "mixed"]
ScenarioId = Literal["S0", "S1", "S2", "S3", "S4"]


@dataclass(frozen=True)
class ScenarioSpec:
    """Metadata for one controlled scenario."""

    key: str
    scenario_id: ScenarioId
    name: str
    description: str
    attack_vehicle_count: int | Literal["variable"]
    expect_coordinated_campaign: bool
    expect_multi_vehicle_alerts: bool
    evidence_level: EvidenceLevel
    coordination_strength_range: tuple[float, float]
    is_primary: bool
    attack_family_mode: Literal["none", "single", "distinct_per_vehicle", "shared"]
    ground_truth_campaign_label: str | None

    def validate_expectations(self) -> list[str]:
        """Return list of validation warnings (empty if consistent)."""
        warnings: list[str] = []
        if self.scenario_id == "S0":
            if self.expect_coordinated_campaign or self.attack_vehicle_count != 0:
                warnings.append("S0 must have zero attacks and no campaigns.")
        if self.scenario_id == "S1":
            if self.attack_vehicle_count != 1 or self.expect_coordinated_campaign:
                warnings.append("S1 must have exactly one attacked vehicle and no campaign.")
        if self.scenario_id == "S2" and self.expect_coordinated_campaign:
            warnings.append("S2 must not expect a single coordinated campaign.")
        if self.scenario_id in ("S3", "S4") and not self.expect_coordinated_campaign:
            warnings.append(f"{self.scenario_id} must expect a coordinated campaign.")
        if self.scenario_id == "S3" and self.evidence_level != "strong":
            warnings.append("S3 requires strong local evidence (score >= strong_threshold).")
        if self.scenario_id == "S4" and self.evidence_level != "weak":
            warnings.append("S4 requires weak local evidence (weak <= score < strong).")
        return warnings


SCENARIO_REGISTRY: dict[str, ScenarioSpec] = {
    "S0_benign_control": ScenarioSpec(
        key="S0_benign_control",
        scenario_id="S0",
        name="Benign fleet control",
        description=(
            "All vehicles contain benign traffic only. Measures local FPR and "
            "fleet-level false campaign alerts; confirms graph correlation does not "
            "create artificial campaigns."
        ),
        attack_vehicle_count=0,
        expect_coordinated_campaign=False,
        expect_multi_vehicle_alerts=False,
        evidence_level="none",
        coordination_strength_range=(0.0, 0.0),
        is_primary=False,
        attack_family_mode="none",
        ground_truth_campaign_label=None,
    ),
    "S1_isolated": ScenarioSpec(
        key="S1_isolated",
        scenario_id="S1",
        name="Isolated single-vehicle attack",
        description=(
            "One vehicle contains malicious events; remaining vehicles are benign. "
            "Confirms local IDS detection and fleet classification as isolated incident."
        ),
        attack_vehicle_count=1,
        expect_coordinated_campaign=False,
        expect_multi_vehicle_alerts=False,
        evidence_level="strong",
        coordination_strength_range=(0.0, 0.0),
        is_primary=False,
        attack_family_mode="single",
        ground_truth_campaign_label=None,
    ),
    "S2_non_coordinated": ScenarioSpec(
        key="S2_non_coordinated",
        scenario_id="S2",
        name="Non-coordinated multi-vehicle incidents",
        description=(
            "Multiple vehicles attacked with different behavioural manifestations. "
            "Non-coordination defined by distinct attack families and low descriptor "
            "similarity — not by timing."
        ),
        attack_vehicle_count="variable",
        expect_coordinated_campaign=False,
        expect_multi_vehicle_alerts=True,
        evidence_level="strong",
        coordination_strength_range=(0.0, 0.25),
        is_primary=False,
        attack_family_mode="distinct_per_vehicle",
        ground_truth_campaign_label=None,
    ),
    "S3_strong_campaign": ScenarioSpec(
        key="S3_strong_campaign",
        scenario_id="S3",
        name="Strong coordinated multi-vehicle campaign",
        description=(
            "Behaviourally similar attacks across N vehicles with strong local anomaly "
            "evidence (anomaly_score >= strong_threshold)."
        ),
        attack_vehicle_count="variable",
        expect_coordinated_campaign=True,
        expect_multi_vehicle_alerts=True,
        evidence_level="strong",
        coordination_strength_range=(0.75, 1.0),
        is_primary=False,
        attack_family_mode="shared",
        ground_truth_campaign_label="CAMP-S3",
    ),
    "S4_weak_campaign": ScenarioSpec(
        key="S4_weak_campaign",
        scenario_id="S4",
        name="Weak coordinated multi-vehicle campaign",
        description=(
            "Behaviourally similar attacks across N vehicles with weak local evidence "
            "(weak_threshold <= anomaly_score < strong_threshold). Primary contribution "
            "scenario for fleet correlation value."
        ),
        attack_vehicle_count="variable",
        expect_coordinated_campaign=True,
        expect_multi_vehicle_alerts=True,
        evidence_level="weak",
        coordination_strength_range=(0.75, 1.0),
        is_primary=True,
        attack_family_mode="shared",
        ground_truth_campaign_label="CAMP-S4",
    ),
}

METHOD_IDS: dict[str, str] = {
    "local": "local_ids",
    "local_ids": "local_ids",
    "clustering": "descriptor_clustering",
    "descriptor_clustering": "descriptor_clustering",
    "standard_gnn": "standard_gnn",
    "gcn": "standard_gnn",
    "fcgnn": "fcgnn",
    "proposed": "fcgnn",
}

METHOD_LABELS: dict[str, str] = {
    "local_ids": "M1 Local IDS",
    "descriptor_clustering": "M2 Descriptor clustering",
    "standard_gnn": "M3 Standard GNN",
    "fcgnn": "M4 Proposed FCGNN (GraphSAGEFleetCorrelator)",
}


def list_scenarios() -> list[ScenarioSpec]:
    return list(SCENARIO_REGISTRY.values())


def get_scenario(key: str) -> ScenarioSpec:
    if key not in SCENARIO_REGISTRY:
        known = ", ".join(sorted(SCENARIO_REGISTRY))
        raise KeyError(f"Unknown scenario '{key}'. Known: {known}")
    return SCENARIO_REGISTRY[key]


def resolve_method(method: str) -> str:
    normalized = method.strip().lower()
    if normalized not in METHOD_IDS:
        known = ", ".join(sorted(set(METHOD_IDS.keys())))
        raise KeyError(f"Unknown method '{method}'. Known: {known}")
    return METHOD_IDS[normalized]


def enumerate_run_plan(
    *,
    scenario_keys: list[str],
    methods: list[str],
    seeds: list[int],
    campaign_sizes: list[int],
    coordination_strengths: list[float],
    include_edge_sensitivity: bool = False,
) -> list[dict[str, Any]]:
    """Enumerate all run combinations for dry-run / manifest generation."""
    runs: list[dict[str, Any]] = []
    for scenario_key in scenario_keys:
        spec = get_scenario(scenario_key)
        sizes = [0] if spec.attack_vehicle_count == 0 else (
            [1] if spec.attack_vehicle_count == 1 else campaign_sizes
        )
        cs_values = (
            [0.0]
            if spec.scenario_id in ("S0", "S1")
            else _coordination_values_for_scenario(spec, coordination_strengths)
        )
        for seed in seeds:
            for n in sizes:
                for cs in cs_values:
                    for method in methods:
                        method_id = resolve_method(method)
                        if include_edge_sensitivity and method_id not in (
                            "standard_gnn",
                            "fcgnn",
                        ):
                            continue
                        runs.append(
                            {
                                "scenario_key": scenario_key,
                                "scenario_id": spec.scenario_id,
                                "method": method_id,
                                "seed": seed,
                                "campaign_size": n,
                                "coordination_strength": cs,
                            }
                        )
    return runs


def _coordination_values_for_scenario(
    spec: ScenarioSpec,
    all_strengths: list[float],
) -> list[float]:
    lo, hi = spec.coordination_strength_range
    return [cs for cs in all_strengths if lo - 1e-9 <= cs <= hi + 1e-9]


def validate_registry() -> list[str]:
    """Validate all scenario specs; return error messages."""
    errors: list[str] = []
    for key, spec in SCENARIO_REGISTRY.items():
        if spec.key != key:
            errors.append(f"Registry key mismatch: {key} vs {spec.key}")
        errors.extend(spec.validate_expectations())
    return errors
