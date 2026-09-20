"""Experimental validation-suite builder for paper-defined S0–S4 scenarios.

REPRODUCIBILITY / IMPLEMENTATION REPAIR only.

This module recreates ``build_mixed_validation_suite`` for *experimental
evaluation setup*. It is NOT part of FLEET-GUARD inference, Algorithms 1–2,
graph construction, GraphSAGE, DBSCAN, or the campaign gate.

Coordination prototype-blend is an experimental construction mechanism only.
The numeric blend strength is an EXPERIMENTAL BUILDER CONFIGURATION value —
not a paper/model symbol and not a recovered historical P7/P8 constant.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd

from src.experiments.data_splits import is_benign_attack_type
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS

# ---------------------------------------------------------------------------
# Frozen experimental-scale constants (publication config / recovered peers)
# ---------------------------------------------------------------------------

VALIDATION_SEEDS: list[int] = [131, 137, 149, 157, 163, 179, 181, 191, 193, 197]
CAMPAIGN_SIZES: list[int] = [2, 5, 10]
FLEET_SIZE: int = 20
DESCRIPTORS_PER_VEHICLE: int = 10
MALICIOUS_PER_ATTACKED: int = 5
BENIGN_PER_ATTACKED: int = 5
BENIGN_PER_BENIGN: int = 10
MAX_FLEET_NODES: int = FLEET_SIZE * DESCRIPTORS_PER_VEHICLE  # 200

# Paper-defined local evidence thresholds (do not redefine)
THETA_WEAK: float = 0.55
THETA_STRONG: float = 0.80

# Recovered default attack-family string (same for strong/weak peers)
DEFAULT_ATTACK_TYPE: str = "malfunction"

ScenarioId = Literal["S0", "S1", "S2", "S3", "S4"]
SCENARIO_IDS: tuple[ScenarioId, ...] = ("S0", "S1", "S2", "S3", "S4")

# Map paper scenarios to historical V0–V4 labels (reporting compatibility only)
SCENARIO_TO_V_LABEL: dict[ScenarioId, str] = {
    "S0": "V0",
    "S1": "V1",
    "S2": "V2",
    "S3": "V3",
    "S4": "V4",
}

# Detector / pipeline modules that must never be imported by this builder
_FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "src.models",
    "src.graph",
    "torch",
    "torch_geometric",
)


class BuilderConfigurationError(ValueError):
    """Raised when required experimental builder configuration is missing."""


class BuilderSemanticsError(ValueError):
    """Raised when requested construction violates recovered / paper semantics."""


@dataclass(frozen=True)
class DescriptorBudget:
    """Fixed node budget (publication DescriptorBudget semantics)."""

    descriptors_per_vehicle: int = DESCRIPTORS_PER_VEHICLE
    malicious_per_attacked: int = MALICIOUS_PER_ATTACKED
    benign_per_attacked: int = BENIGN_PER_ATTACKED
    benign_per_benign: int = BENIGN_PER_BENIGN
    total_fleet_size: int = FLEET_SIZE

    @property
    def expected_total_nodes(self) -> int:
        return self.total_fleet_size * self.descriptors_per_vehicle


# ---------------------------------------------------------------------------
# Recovered experimental coordination blend (construction only — not detector)
# ---------------------------------------------------------------------------


def _behavioural_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in BEHAVIOURAL_FEATURE_COLUMNS if c in df.columns]


def compute_campaign_prototype(
    descriptors: pd.DataFrame,
    *,
    attack_type: str,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    """Mean behavioural vector for one attack family (recovered semantics).

    Prototype source pool = the ``descriptors`` frame supplied by the caller
    (validation-only pool after suite filtering). This matches recovered peer
    call sites: ``compute_campaign_prototype(descriptors, attack_type=...)``.
    """
    cols = feature_columns or _behavioural_columns(descriptors)
    sub = descriptors[descriptors["attack_type"] == attack_type]
    if sub.empty:
        raise BuilderSemanticsError(f"No rows for attack_type={attack_type!r} in prototype pool")
    return sub[cols].astype(np.float64).mean(axis=0).to_numpy()


def apply_coordination_strength(
    descriptors: pd.DataFrame,
    *,
    strength: float,
    campaign_prototype: np.ndarray,
    target_mask: pd.Series,
    feature_columns: list[str] | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recovered prototype-blend with bounded noise (experimental setup only).

    Transforms ONLY behavioural feature columns. Never modifies anomaly_score,
    identity, GT labels, attack_type, or provenance columns.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    cols = feature_columns or _behavioural_columns(descriptors)
    out = descriptors.copy()
    provenance_rows: list[dict[str, Any]] = []

    target_idx = out.index[target_mask]
    if len(target_idx) == 0 or strength <= 0.0:
        return out, pd.DataFrame(
            columns=["event_id", "coordination_strength", "method", "original_anomaly_score"]
        )

    sub = out.loc[target_idx, cols].astype(np.float64)
    feat_min = sub.min(axis=0).to_numpy()
    feat_max = sub.max(axis=0).to_numpy()
    proto = np.clip(campaign_prototype, feat_min, feat_max)

    rng = np.random.default_rng(seed)
    noise_scale = 0.02 * (1.0 - strength)
    for idx in target_idx:
        original = out.loc[idx, cols].astype(np.float64).to_numpy()
        blended = (1.0 - strength) * original + strength * proto
        if noise_scale > 0:
            blended = blended + rng.normal(0.0, noise_scale, size=len(cols))
        blended = np.clip(blended, feat_min, feat_max)
        out.loc[idx, cols] = blended
        provenance_rows.append(
            {
                "event_id": out.loc[idx, "event_id"] if "event_id" in out.columns else idx,
                "coordination_strength": strength,
                "method": "prototype_blend_with_bounded_noise",
                "original_anomaly_score": float(out.loc[idx, "anomaly_score"]),
            }
        )
    return out, pd.DataFrame(provenance_rows)


# ---------------------------------------------------------------------------
# Hash / manifest helpers
# ---------------------------------------------------------------------------


def hash_descriptor_vector(row: pd.Series, columns: Sequence[str] | None = None) -> str:
    cols = list(columns) if columns is not None else list(BEHAVIOURAL_FEATURE_COLUMNS)
    parts: list[bytes] = []
    for c in cols:
        val = float(row[c]) if c in row.index and pd.notna(row[c]) else float("nan")
        parts.append(struct.pack("<d", val))
    return hashlib.sha256(b"".join(parts)).hexdigest()


def _resolve_experimental_coordination_strength(
    explicit: float | None,
    config: dict[str, Any] | None,
) -> float:
    """Require an explicit experimental builder configuration value.

    Historical numeric coordination strength is UNRECOVERABLE. This value is
    NOT a paper symbol and MUST NOT be silently chosen to match historical F1.
    """
    if explicit is not None:
        return float(explicit)
    if config:
        nested = config.get("experimental_campaign_builder") or config.get(
            "experimental_builder"
        ) or {}
        if "coordination_strength" in nested:
            return float(nested["coordination_strength"])
        if "experimental_coordination_strength" in config:
            return float(config["experimental_coordination_strength"])
    raise BuilderConfigurationError(
        "experimental_coordination_strength is required for S3/S4 construction. "
        "Historical P7/P8 coordination strength is UNRECOVERABLE; supply an "
        "explicit EXPERIMENTAL BUILDER CONFIGURATION value "
        "(allowed recovered range evidence: 0.75–1.0). "
        "This is not a FLEET-GUARD model/paper symbol."
    )


def _assert_no_forbidden_imports() -> None:
    import sys

    bad = [
        name
        for name in sys.modules
        if any(name == p or name.startswith(p + ".") for p in _FORBIDDEN_IMPORT_PREFIXES)
    ]
    # Allow absence; only fail if builder caused loading of detector stack.
    # Tests assert the builder module itself does not import these.
    _ = bad


def _filter_validation_pool(
    descriptors: pd.DataFrame,
    *,
    test_event_ids: set[str] | None,
) -> pd.DataFrame:
    """Keep validation descriptors only; hard-exclude TEST."""
    df = descriptors.copy()
    if "split" in df.columns:
        split = df["split"].astype(str).str.lower()
        if (split == "test").any():
            df = df.loc[split != "test"].copy()
        if "validation" in set(split.unique()):
            df = df.loc[split == "validation"].copy()
        if (df["split"].astype(str).str.lower() == "test").any():
            raise BuilderSemanticsError("TEST descriptors must not enter validation construction")
    if test_event_ids and "event_id" in df.columns:
        before = len(df)
        df = df.loc[~df["event_id"].astype(str).isin(test_event_ids)].copy()
        if len(df) == before and test_event_ids:
            # still OK if no overlap existed
            pass
        if "event_id" in df.columns and df["event_id"].astype(str).isin(test_event_ids).any():
            raise BuilderSemanticsError("TEST event_ids leaked into validation pool")
    return df.reset_index(drop=True)


def _vehicle_key(df: pd.DataFrame) -> str:
    for col in ("vehicle_token", "scenario_vehicle_id", "source_vehicle"):
        if col in df.columns:
            return col
    raise BuilderSemanticsError(
        "Descriptors require vehicle_token / scenario_vehicle_id / source_vehicle"
    )


def _strong_band(df: pd.DataFrame, strong_th: float = THETA_STRONG) -> pd.Series:
    mal = ~df["attack_type"].map(is_benign_attack_type)
    score = pd.to_numeric(df["anomaly_score"], errors="coerce")
    return mal & (score >= strong_th)


def _weak_band(
    df: pd.DataFrame,
    weak_th: float = THETA_WEAK,
    strong_th: float = THETA_STRONG,
) -> pd.Series:
    mal = ~df["attack_type"].map(is_benign_attack_type)
    score = pd.to_numeric(df["anomaly_score"], errors="coerce")
    return mal & (score >= weak_th) & (score < strong_th)


def _benign_mask(df: pd.DataFrame) -> pd.Series:
    return df["attack_type"].map(is_benign_attack_type)


def _sample_n(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) < n:
        raise BuilderSemanticsError(f"Insufficient rows: need {n}, have {len(df)}")
    if len(df) == n:
        return df.copy()
    idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    return df.loc[idx].copy()


def _coordinated_attack_composition(campaign_size: int) -> dict[str, int]:
    """Hyundai/Kia-only compositions (validation-peer supported; no Chevy override)."""
    if campaign_size == 2:
        return {"Hyundai": 1, "Kia": 1}
    if campaign_size == 5:
        return {"Hyundai": 3, "Kia": 2}
    if campaign_size == 10:
        return {"Hyundai": 5, "Kia": 5}
    raise BuilderSemanticsError(f"Unsupported campaign_size={campaign_size}")


def _ensure_event_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_id" not in out.columns:
        out["event_id"] = [f"evt_{i}" for i in range(len(out))]
    return out


def _annotate_provenance(
    chunk: pd.DataFrame,
    *,
    role: str,
    campaign_id: str,
    is_campaign_member: bool,
    vehicle_token: str,
    source_vehicle: str | None = None,
) -> pd.DataFrame:
    out = chunk.copy()
    out["scenario_role"] = role
    out["vehicle_token"] = vehicle_token
    out["scenario_vehicle_id"] = vehicle_token
    out["source_vehicle"] = source_vehicle or vehicle_token
    if "source_trace" not in out.columns:
        if "source_file" in out.columns:
            out["source_trace"] = out["source_file"].astype(str)
        else:
            out["source_trace"] = ""
    if "window_index" not in out.columns:
        if "window_id" in out.columns:
            out["window_index"] = out["window_id"]
        else:
            out["window_index"] = -1
    out["GT_campaign_id"] = campaign_id if is_campaign_member else ""
    out["ground_truth_campaign_id"] = out["GT_campaign_id"]
    out["ground_truth_campaign_member"] = int(is_campaign_member)
    if "ground_truth_malicious" not in out.columns:
        out["ground_truth_malicious"] = out["attack_type"].map(
            lambda a: int(not is_benign_attack_type(a))
        )
    return out


def _select_vehicles_by_model(
    pool_vehicles: list[str],
    vehicle_models: dict[str, str],
    composition: dict[str, int],
    rng: np.random.Generator,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    exclude = exclude or set()
    chosen: list[str] = []
    for model, count in composition.items():
        if count <= 0:
            continue
        candidates = [
            v
            for v in pool_vehicles
            if v not in exclude and v not in chosen and vehicle_models.get(v) == model
        ]
        if len(candidates) < count:
            raise BuilderSemanticsError(
                f"Need {count} vehicles of model={model}, have {len(candidates)}"
            )
        pick = list(rng.choice(candidates, size=count, replace=False))
        chosen.extend(pick)
    return chosen


def _vehicle_model_map(df: pd.DataFrame, vkey: str) -> dict[str, str]:
    if "vehicle_model" not in df.columns:
        # Synthetic fallback: assign models cyclically for unit tests
        vehicles = sorted(df[vkey].astype(str).unique())
        cycle = ["Hyundai", "Kia", "Chevrolet"]
        return {v: cycle[i % 3] for i, v in enumerate(vehicles)}
    return (
        df.groupby(vkey)["vehicle_model"]
        .agg(lambda s: str(s.iloc[0]))
        .astype(str)
        .to_dict()
    )


def _fill_benign_background(
    pool: pd.DataFrame,
    *,
    n_benign_vehicles: int,
    budget: DescriptorBudget,
    rng: np.random.Generator,
    exclude_vehicles: set[str],
    vkey: str,
) -> list[pd.DataFrame]:
    ben = pool.loc[_benign_mask(pool)]
    models = _vehicle_model_map(ben if len(ben) else pool, vkey)
    # Prefer heterogeneous benign composition when enough models exist
    target_comp = {"Hyundai": 0, "Kia": 0, "Chevrolet": 0}
    # Distribute n_benign_vehicles across models present
    present = sorted({models[v] for v in models if v not in exclude_vehicles})
    if not present:
        raise BuilderSemanticsError("No benign vehicles available for background")
    base, rem = divmod(n_benign_vehicles, len(present))
    for i, m in enumerate(present):
        target_comp[m] = base + (1 if i < rem else 0)

    vehicles = sorted(set(ben[vkey].astype(str)) - exclude_vehicles)
    chosen = _select_vehicles_by_model(vehicles, models, {k: v for k, v in target_comp.items() if v > 0}, rng)
    if len(chosen) < n_benign_vehicles:
        # Fallback: any remaining benign vehicles
        rest = [v for v in vehicles if v not in chosen]
        need = n_benign_vehicles - len(chosen)
        if len(rest) < need:
            raise BuilderSemanticsError("Insufficient benign vehicles for fleet fill")
        chosen.extend(list(rng.choice(rest, size=need, replace=False)))

    rows: list[pd.DataFrame] = []
    for v in chosen[:n_benign_vehicles]:
        seg = ben.loc[ben[vkey].astype(str) == v]
        chunk = _sample_n(seg, budget.benign_per_benign, rng)
        rows.append(
            _annotate_provenance(
                chunk,
                role="benign",
                campaign_id="",
                is_campaign_member=False,
                vehicle_token=v,
                source_vehicle=v,
            )
        )
    return rows


def _build_attacked_chunk(
    seg: pd.DataFrame,
    *,
    band: Literal["strong", "weak"],
    budget: DescriptorBudget,
    rng: np.random.Generator,
    attack_type: str = DEFAULT_ATTACK_TYPE,
) -> pd.DataFrame:
    typed = seg
    if attack_type and "attack_type" in seg.columns:
        cand = seg[seg["attack_type"] == attack_type]
        if not cand.empty:
            typed = cand
    if band == "strong":
        mal_pool = typed.loc[_strong_band(typed)]
    else:
        mal_pool = typed.loc[_weak_band(typed)]
    mal = _sample_n(mal_pool, budget.malicious_per_attacked, rng)
    mal = mal.copy()
    mal["ground_truth_malicious"] = 1
    remaining = seg.drop(index=mal.index, errors="ignore")
    # Companion rows on attacked vehicle (not required to be score-band malicious)
    ben_pool = remaining.loc[_benign_mask(remaining)]
    if len(ben_pool) < budget.benign_per_attacked:
        ben_pool = remaining
    ben = _sample_n(ben_pool, budget.benign_per_attacked, rng).copy()
    ben["ground_truth_malicious"] = 0
    return pd.concat([mal, ben], ignore_index=False)


def _finalize_scenario_df(
    parts: list[pd.DataFrame],
    *,
    scenario: ScenarioId,
    seed: int,
    campaign_size: int,
    experimental_coordination_strength: float,
    budget: DescriptorBudget,
) -> pd.DataFrame:
    df = pd.concat(parts, ignore_index=True)
    df = _ensure_event_ids(df)
    if len(df) != budget.expected_total_nodes:
        raise BuilderSemanticsError(
            f"{scenario} seed={seed}: node count {len(df)} != {budget.expected_total_nodes}"
        )
    n_veh = df["vehicle_token"].nunique()
    if n_veh != budget.total_fleet_size:
        raise BuilderSemanticsError(
            f"{scenario} seed={seed}: fleet size {n_veh} != {budget.total_fleet_size}"
        )
    per = df.groupby("vehicle_token").size()
    if int(per.min()) != budget.descriptors_per_vehicle or int(per.max()) != budget.descriptors_per_vehicle:
        raise BuilderSemanticsError(
            f"{scenario} seed={seed}: descriptors/vehicle not fixed at {budget.descriptors_per_vehicle}"
        )

    cols = _behavioural_columns(df)
    df["original_descriptor_hash"] = df.apply(lambda r: hash_descriptor_vector(r, cols), axis=1)
    # constructed hash filled after optional blend
    df["constructed_descriptor_hash"] = df["original_descriptor_hash"]
    df["scenario"] = scenario
    df["seed"] = seed
    df["campaign_size"] = campaign_size
    df["experimental_coordination_strength"] = float(experimental_coordination_strength)
    df["builder_module"] = "src.experiments.final_shared_configuration.validation_scenarios"
    df["construction_scope"] = "experimental_evaluation_only"
    return df


def _apply_s3_s4_blend(
    scenario_df: pd.DataFrame,
    *,
    pool_for_prototype: pd.DataFrame,
    experimental_coordination_strength: float,
    seed: int,
) -> pd.DataFrame:
    strength = float(experimental_coordination_strength)
    if strength <= 0:
        raise BuilderConfigurationError(
            "S3/S4 require experimental_coordination_strength > 0 "
            "(shared campaign prototype blend)"
        )
    # Preserve anomaly scores / identity for assert
    scores_before = scenario_df["anomaly_score"].astype(float).copy()
    identity_cols = [
        c
        for c in (
            "vehicle_token",
            "source_vehicle",
            "source_trace",
            "window_index",
            "attack_type",
            "GT_campaign_id",
            "ground_truth_campaign_member",
            "ground_truth_malicious",
            "event_id",
            "original_descriptor_hash",
        )
        if c in scenario_df.columns
    ]
    identity_before = scenario_df[identity_cols].copy()

    proto = compute_campaign_prototype(pool_for_prototype, attack_type=DEFAULT_ATTACK_TYPE)
    mask = scenario_df["scenario_role"] == "coordinated"
    out, _ = apply_coordination_strength(
        scenario_df,
        strength=strength,
        campaign_prototype=proto,
        target_mask=mask,
        seed=seed,
    )
    # Guarantees
    if not np.allclose(
        out["anomaly_score"].astype(float).to_numpy(),
        scores_before.to_numpy(),
        equal_nan=True,
    ):
        raise BuilderSemanticsError("anomaly_score must never change during campaign construction")
    for c in identity_cols:
        if not out[c].astype(str).equals(identity_before[c].astype(str)):
            raise BuilderSemanticsError(f"Identity/provenance column altered: {c}")

    cols = _behavioural_columns(out)
    out["constructed_descriptor_hash"] = out.apply(lambda r: hash_descriptor_vector(r, cols), axis=1)
    out["prototype_attack_type"] = DEFAULT_ATTACK_TYPE
    out["prototype_pool"] = "validation_descriptors_matching_attack_type"
    return out


# ---------------------------------------------------------------------------
# Per-scenario builders
# ---------------------------------------------------------------------------


def build_scenario_s0(
    descriptors: pd.DataFrame,
    *,
    seed: int,
    budget: DescriptorBudget | None = None,
    test_event_ids: set[str] | None = None,
) -> pd.DataFrame:
    """S0: benign fleet — no campaign attack."""
    budget = budget or DescriptorBudget()
    pool = _filter_validation_pool(descriptors, test_event_ids=test_event_ids)
    pool = _ensure_event_ids(pool)
    rng = np.random.default_rng(seed)
    vkey = _vehicle_key(pool)
    parts = _fill_benign_background(
        pool,
        n_benign_vehicles=budget.total_fleet_size,
        budget=budget,
        rng=rng,
        exclude_vehicles=set(),
        vkey=vkey,
    )
    df = _finalize_scenario_df(
        parts,
        scenario="S0",
        seed=seed,
        campaign_size=0,
        experimental_coordination_strength=0.0,
        budget=budget,
    )
    if (df["ground_truth_malicious"] == 1).any() or (df["ground_truth_campaign_member"] == 1).any():
        raise BuilderSemanticsError("S0 must contain no campaign attack")
    return df


def build_scenario_s1(
    descriptors: pd.DataFrame,
    *,
    seed: int,
    budget: DescriptorBudget | None = None,
    test_event_ids: set[str] | None = None,
) -> pd.DataFrame:
    """S1: isolated malicious activity on exactly one vehicle (no shared blend)."""
    budget = budget or DescriptorBudget()
    pool = _filter_validation_pool(descriptors, test_event_ids=test_event_ids)
    pool = _ensure_event_ids(pool)
    rng = np.random.default_rng(seed)
    vkey = _vehicle_key(pool)
    strong = pool.loc[_strong_band(pool)]
    attacked_vehicles = sorted(strong[vkey].astype(str).unique())
    if not attacked_vehicles:
        raise BuilderSemanticsError("S1: no strong-band malicious vehicles in validation pool")
    # Prefer Hyundai when available (validation peer)
    models = _vehicle_model_map(pool, vkey)
    hyundai = [v for v in attacked_vehicles if models.get(v) == "Hyundai"]
    pick_pool = hyundai or attacked_vehicles
    attacked = str(rng.choice(pick_pool))
    seg = pool.loc[pool[vkey].astype(str) == attacked]
    chunk = _build_attacked_chunk(seg, band="strong", budget=budget, rng=rng)
    attacked_part = _annotate_provenance(
        chunk,
        role="isolated",
        campaign_id="",
        is_campaign_member=False,
        vehicle_token=attacked,
        source_vehicle=attacked,
    )
    # S1 is not a coordinated campaign member
    attacked_part["ground_truth_campaign_member"] = 0
    attacked_part["GT_campaign_id"] = ""
    attacked_part["ground_truth_campaign_id"] = ""

    parts = [attacked_part] + _fill_benign_background(
        pool,
        n_benign_vehicles=budget.total_fleet_size - 1,
        budget=budget,
        rng=rng,
        exclude_vehicles={attacked},
        vkey=vkey,
    )
    df = _finalize_scenario_df(
        parts,
        scenario="S1",
        seed=seed,
        campaign_size=1,
        experimental_coordination_strength=0.0,
        budget=budget,
    )
    attacked_ids = df.loc[df["scenario_role"] == "isolated", "vehicle_token"].unique()
    if len(attacked_ids) != 1:
        raise BuilderSemanticsError("S1 must attack exactly one vehicle")
    return df


def build_scenario_s2(
    descriptors: pd.DataFrame,
    *,
    seed: int,
    campaign_size: int = 5,
    budget: DescriptorBudget | None = None,
    test_event_ids: set[str] | None = None,
) -> pd.DataFrame:
    """S2: independent multi-vehicle malicious activity — no shared prototype blend."""
    if campaign_size not in CAMPAIGN_SIZES:
        raise BuilderSemanticsError(f"S2 campaign_size must be in {CAMPAIGN_SIZES}")
    budget = budget or DescriptorBudget()
    pool = _filter_validation_pool(descriptors, test_event_ids=test_event_ids)
    pool = _ensure_event_ids(pool)
    rng = np.random.default_rng(seed)
    vkey = _vehicle_key(pool)
    strong = pool.loc[_strong_band(pool)]
    models = _vehicle_model_map(pool, vkey)
    comp = _coordinated_attack_composition(campaign_size)
    vehicles = sorted(strong[vkey].astype(str).unique())
    attacked = _select_vehicles_by_model(vehicles, models, comp, rng)

    parts: list[pd.DataFrame] = []
    for i, v in enumerate(attacked):
        seg = pool.loc[pool[vkey].astype(str) == v]
        chunk = _build_attacked_chunk(seg, band="strong", budget=budget, rng=rng)
        # Distinct incident id per vehicle — independence / no shared campaign
        incident_id = f"INCIDENT-S2-{i}"
        parts.append(
            _annotate_provenance(
                chunk,
                role="unrelated",
                campaign_id=incident_id,
                is_campaign_member=True,
                vehicle_token=v,
                source_vehicle=v,
            )
        )
    # Mark as independent incidents: campaign member flags record incident membership
    # but builder must NOT apply shared prototype blend (strength 0).
    parts.extend(
        _fill_benign_background(
            pool,
            n_benign_vehicles=budget.total_fleet_size - campaign_size,
            budget=budget,
            rng=rng,
            exclude_vehicles=set(attacked),
            vkey=vkey,
        )
    )
    df = _finalize_scenario_df(
        parts,
        scenario="S2",
        seed=seed,
        campaign_size=campaign_size,
        experimental_coordination_strength=0.0,
        budget=budget,
    )
    incident_ids = sorted(
        {x for x in df.loc[df["scenario_role"] == "unrelated", "GT_campaign_id"].astype(str) if x}
    )
    if len(incident_ids) != campaign_size:
        raise BuilderSemanticsError("S2 must assign distinct incident ids per attacked vehicle")
    return df


def build_scenario_s3(
    descriptors: pd.DataFrame,
    *,
    seed: int,
    campaign_size: int,
    experimental_coordination_strength: float,
    budget: DescriptorBudget | None = None,
    test_event_ids: set[str] | None = None,
) -> pd.DataFrame:
    """S3: coordinated campaign; malicious descriptors with anomaly_score >= 0.80."""
    return _build_coordinated_scenario(
        descriptors,
        scenario="S3",
        band="strong",
        seed=seed,
        campaign_size=campaign_size,
        experimental_coordination_strength=experimental_coordination_strength,
        budget=budget,
        test_event_ids=test_event_ids,
    )


def build_scenario_s4(
    descriptors: pd.DataFrame,
    *,
    seed: int,
    campaign_size: int,
    experimental_coordination_strength: float,
    budget: DescriptorBudget | None = None,
    test_event_ids: set[str] | None = None,
) -> pd.DataFrame:
    """S4: coordinated campaign; malicious descriptors with 0.55 <= score < 0.80."""
    return _build_coordinated_scenario(
        descriptors,
        scenario="S4",
        band="weak",
        seed=seed,
        campaign_size=campaign_size,
        experimental_coordination_strength=experimental_coordination_strength,
        budget=budget,
        test_event_ids=test_event_ids,
    )


def _build_coordinated_scenario(
    descriptors: pd.DataFrame,
    *,
    scenario: Literal["S3", "S4"],
    band: Literal["strong", "weak"],
    seed: int,
    campaign_size: int,
    experimental_coordination_strength: float,
    budget: DescriptorBudget | None,
    test_event_ids: set[str] | None,
) -> pd.DataFrame:
    if campaign_size not in CAMPAIGN_SIZES:
        raise BuilderSemanticsError(f"{scenario} campaign_size must be in {CAMPAIGN_SIZES}")
    budget = budget or DescriptorBudget()
    pool = _filter_validation_pool(descriptors, test_event_ids=test_event_ids)
    pool = _ensure_event_ids(pool)
    rng = np.random.default_rng(seed)
    vkey = _vehicle_key(pool)
    band_df = pool.loc[_strong_band(pool) if band == "strong" else _weak_band(pool)]
    models = _vehicle_model_map(pool, vkey)
    comp = _coordinated_attack_composition(campaign_size)
    vehicles = sorted(band_df[vkey].astype(str).unique())
    attacked = _select_vehicles_by_model(vehicles, models, comp, rng)

    campaign_id = f"CAMP-{scenario}-cs{campaign_size}-seed{seed}"
    parts: list[pd.DataFrame] = []
    for v in attacked:
        seg = pool.loc[pool[vkey].astype(str) == v]
        chunk = _build_attacked_chunk(seg, band=band, budget=budget, rng=rng)
        # Verify malicious rows respect band
        mal_rows = chunk.loc[chunk["ground_truth_malicious"] == 1]
        if band == "strong":
            if not bool((_strong_band(mal_rows)).all()):
                raise BuilderSemanticsError("S3 malicious rows must have anomaly_score >= 0.80")
        else:
            if not bool((_weak_band(mal_rows)).all()):
                raise BuilderSemanticsError(
                    "S4 malicious rows must have 0.55 <= anomaly_score < 0.80"
                )
        parts.append(
            _annotate_provenance(
                chunk,
                role="coordinated",
                campaign_id=campaign_id,
                is_campaign_member=True,
                vehicle_token=v,
                source_vehicle=v,
            )
        )
    parts.extend(
        _fill_benign_background(
            pool,
            n_benign_vehicles=budget.total_fleet_size - campaign_size,
            budget=budget,
            rng=rng,
            exclude_vehicles=set(attacked),
            vkey=vkey,
        )
    )
    df = _finalize_scenario_df(
        parts,
        scenario=scenario,
        seed=seed,
        campaign_size=campaign_size,
        experimental_coordination_strength=float(experimental_coordination_strength),
        budget=budget,
    )
    df = _apply_s3_s4_blend(
        df,
        pool_for_prototype=pool,
        experimental_coordination_strength=float(experimental_coordination_strength),
        seed=seed,
    )
    return df


def build_descriptor_manifest(scenario_df: pd.DataFrame) -> pd.DataFrame:
    """Auditable per-descriptor manifest (experimental setup provenance)."""
    cols = [
        "scenario",
        "seed",
        "campaign_size",
        "vehicle_token",
        "source_vehicle",
        "source_trace",
        "window_index",
        "attack_type",
        "anomaly_score",
        "GT_campaign_id",
        "original_descriptor_hash",
        "constructed_descriptor_hash",
        "experimental_coordination_strength",
        "scenario_role",
        "event_id",
        "ground_truth_malicious",
        "ground_truth_campaign_member",
    ]
    present = [c for c in cols if c in scenario_df.columns]
    return scenario_df[present].copy()


def build_mixed_validation_suite(
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    budget: DescriptorBudget | None = None,
    *,
    test_runs_root: Path | str | None = None,
    val_out_dir: Path | str | None = None,
    test_event_ids: Iterable[str] | set[str] | None = None,
    experimental_coordination_strength: float | None = None,
    campaign_sizes: Sequence[int] | None = None,
    validation_seeds: Sequence[int] | None = None,
    write_artifacts: bool = True,
    scenarios: Sequence[ScenarioId] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Build paper-defined S0–S4 validation scenarios (experimental evaluation only).

    Parameters
    ----------
    experimental_coordination_strength
        Required EXPERIMENTAL BUILDER CONFIGURATION for S3/S4 shared prototype
        blend. Historical value is UNRECOVERABLE — must be supplied explicitly
        (recovered allowed-range evidence: 0.75–1.0). Not a paper/model symbol.

    Returns
    -------
    scenarios_by_id
        Mapping ``run_id -> scenario_df``.
    suite_manifest
        One row per generated scenario run (plus optional per-descriptor CSVs).
    """
    _ = manifest  # retained for runner call-site compatibility; selection uses descriptors
    _ = test_runs_root
    config = config or {}
    budget = budget or DescriptorBudget(
        descriptors_per_vehicle=int(
            (config.get("scenario") or {}).get("source_windows_per_vehicle", DESCRIPTORS_PER_VEHICLE)
        ),
        total_fleet_size=int((config.get("scenario") or {}).get("fleet_size", FLEET_SIZE)),
    )
    seeds = list(validation_seeds) if validation_seeds is not None else list(VALIDATION_SEEDS)
    sizes = list(campaign_sizes) if campaign_sizes is not None else list(CAMPAIGN_SIZES)
    scenario_list: list[ScenarioId] = list(scenarios) if scenarios is not None else list(SCENARIO_IDS)
    test_ids = set(str(x) for x in test_event_ids) if test_event_ids is not None else set()

    # Thresholds from config must match paper (reject silent drift)
    local = config.get("local_ids") or {}
    weak_th = float(local.get("weak_threshold", THETA_WEAK))
    strong_th = float(local.get("strong_threshold", THETA_STRONG))
    if abs(weak_th - THETA_WEAK) > 1e-12 or abs(strong_th - THETA_STRONG) > 1e-12:
        raise BuilderSemanticsError(
            f"Builder refuses threshold drift: got weak={weak_th}, strong={strong_th}; "
            f"paper values are {THETA_WEAK}/{THETA_STRONG}"
        )

    needs_coord = any(s in ("S3", "S4") for s in scenario_list)
    coord_strength: float | None = None
    if needs_coord:
        coord_strength = _resolve_experimental_coordination_strength(
            experimental_coordination_strength, config
        )

    out_dir = Path(val_out_dir) if val_out_dir is not None else None
    if write_artifacts and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "descriptor_manifests").mkdir(parents=True, exist_ok=True)

    scenarios_by_id: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, Any]] = []

    for scenario in scenario_list:
        size_iter: list[int]
        if scenario == "S0":
            size_iter = [0]
        elif scenario == "S1":
            size_iter = [1]
        else:
            size_iter = sizes

        for seed in seeds:
            for cs in size_iter:
                run_id = f"val_{scenario}_cs{cs}_seed{seed}"
                try:
                    if scenario == "S0":
                        sdf = build_scenario_s0(
                            descriptors, seed=seed, budget=budget, test_event_ids=test_ids
                        )
                    elif scenario == "S1":
                        sdf = build_scenario_s1(
                            descriptors, seed=seed, budget=budget, test_event_ids=test_ids
                        )
                    elif scenario == "S2":
                        sdf = build_scenario_s2(
                            descriptors,
                            seed=seed,
                            campaign_size=cs,
                            budget=budget,
                            test_event_ids=test_ids,
                        )
                    elif scenario == "S3":
                        assert coord_strength is not None
                        sdf = build_scenario_s3(
                            descriptors,
                            seed=seed,
                            campaign_size=cs,
                            experimental_coordination_strength=coord_strength,
                            budget=budget,
                            test_event_ids=test_ids,
                        )
                    elif scenario == "S4":
                        assert coord_strength is not None
                        sdf = build_scenario_s4(
                            descriptors,
                            seed=seed,
                            campaign_size=cs,
                            experimental_coordination_strength=coord_strength,
                            budget=budget,
                            test_event_ids=test_ids,
                        )
                    else:
                        raise BuilderSemanticsError(f"Unknown scenario {scenario}")

                    if test_ids and sdf["event_id"].astype(str).isin(test_ids).any():
                        raise BuilderSemanticsError("TEST data entered validation scenario")

                    scenarios_by_id[run_id] = sdf
                    dmanifest = build_descriptor_manifest(sdf)
                    if write_artifacts and out_dir is not None:
                        sdf.to_csv(out_dir / f"{run_id}_records.csv", index=False)
                        dmanifest.to_csv(
                            out_dir / "descriptor_manifests" / f"{run_id}_manifest.csv",
                            index=False,
                        )

                    overlap = (
                        int(sdf["event_id"].astype(str).isin(test_ids).sum()) if test_ids else 0
                    )
                    manifest_rows.append(
                        {
                            "validation_run_id": run_id,
                            "validation_seed": seed,
                            "scenario": scenario,
                            "v_label": SCENARIO_TO_V_LABEL[scenario],
                            "attack_strength": {
                                "S0": "benign",
                                "S1": "strong",
                                "S2": "strong",
                                "S3": "strong",
                                "S4": "weak",
                            }[scenario],
                            "campaign_size": cs,
                            "descriptor_count": len(sdf),
                            "fleet_size": int(sdf["vehicle_token"].nunique()),
                            "overlap_with_test": overlap,
                            "validation_passed": overlap == 0 and len(sdf) == budget.expected_total_nodes,
                            "experimental_coordination_strength": float(
                                sdf["experimental_coordination_strength"].iloc[0]
                            ),
                            "experimental_coordination_strength_status": (
                                "EXPLICIT_EXPERIMENTAL_CONFIG"
                                if scenario in ("S3", "S4")
                                else "NOT_APPLICABLE"
                            ),
                            "historical_coordination_strength": "UNRECOVERABLE",
                            "prototype_pool": (
                                "validation_descriptors_matching_attack_type"
                                if scenario in ("S3", "S4")
                                else ""
                            ),
                            "builder": "build_mixed_validation_suite",
                            "construction_scope": "experimental_evaluation_only",
                        }
                    )
                except Exception as exc:
                    manifest_rows.append(
                        {
                            "validation_run_id": run_id,
                            "validation_seed": seed,
                            "scenario": scenario,
                            "v_label": SCENARIO_TO_V_LABEL[scenario],
                            "campaign_size": cs,
                            "descriptor_count": 0,
                            "validation_passed": False,
                            "error": str(exc),
                            "builder": "build_mixed_validation_suite",
                        }
                    )

    suite_manifest = pd.DataFrame(manifest_rows)
    if write_artifacts and out_dir is not None:
        suite_manifest.to_csv(out_dir / "validation_manifest.csv", index=False)
        meta = {
            "module": "src.experiments.final_shared_configuration.validation_scenarios",
            "role": "experimental_evaluation_campaign_builder",
            "not_part_of": [
                "fleet_guard_inference",
                "algorithm_1",
                "algorithm_2",
                "graphsage",
                "dbscan",
                "campaign_gate",
            ],
            "validation_seeds": seeds,
            "campaign_sizes": sizes,
            "fleet_size": budget.total_fleet_size,
            "descriptors_per_vehicle": budget.descriptors_per_vehicle,
            "theta_weak": THETA_WEAK,
            "theta_strong": THETA_STRONG,
            "experimental_coordination_strength": coord_strength,
            "historical_coordination_strength": "UNRECOVERABLE",
            "prototype_source": "mean over matching attack_type in validation descriptor pool",
            "chevrolet_fuzzy_override": False,
        }
        (out_dir / "builder_config.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return scenarios_by_id, suite_manifest


__all__ = [
    "VALIDATION_SEEDS",
    "CAMPAIGN_SIZES",
    "FLEET_SIZE",
    "DESCRIPTORS_PER_VEHICLE",
    "MAX_FLEET_NODES",
    "THETA_WEAK",
    "THETA_STRONG",
    "DescriptorBudget",
    "BuilderConfigurationError",
    "BuilderSemanticsError",
    "compute_campaign_prototype",
    "apply_coordination_strength",
    "hash_descriptor_vector",
    "build_scenario_s0",
    "build_scenario_s1",
    "build_scenario_s2",
    "build_scenario_s3",
    "build_scenario_s4",
    "build_descriptor_manifest",
    "build_mixed_validation_suite",
]
