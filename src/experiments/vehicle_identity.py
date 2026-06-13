"""Opaque vehicle-instance tokens and graph identity resolution."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.experiments.vehicle_instance_builder import source_trace_name


@dataclass
class VehicleTokenAllocator:
    """Assign opaque production tokens V_0001, V_0002, ..."""

    _next: int = 1
    _by_key: dict[str, str] = field(default_factory=dict)

    def token_for(self, key: str) -> str:
        if key not in self._by_key:
            self._by_key[key] = f"V_{self._next:04d}"
            self._next += 1
        return self._by_key[key]

    def mapping_table(self) -> pd.DataFrame:
        rows = [{"instance_key": k, "vehicle_token": v} for k, v in self._by_key.items()]
        return pd.DataFrame(rows)


def resolve_graph_vehicle_column(meta: pd.DataFrame) -> str:
    """
    Operational graph vehicle identifier for kNN constraints and PyG vehicle_id.

    Priority: vehicle_token → scenario_vehicle_id.
    Never returns vehicle_model.
    """
    if "vehicle_token" in meta.columns and meta["vehicle_token"].notna().all():
        return "vehicle_token"
    if "scenario_vehicle_id" in meta.columns and meta["scenario_vehicle_id"].notna().all():
        return "scenario_vehicle_id"
    raise ValueError(
        "Fleet graph requires opaque vehicle_token or scenario_vehicle_id on all rows; "
        "vehicle_model must not be used as the operational identifier."
    )


def resolve_vehicle_id_column(meta: pd.DataFrame) -> str:
    """Alias used by experiment_pipeline and method runners."""
    return resolve_graph_vehicle_column(meta)


def assign_identity_to_chunk(
    chunk: pd.DataFrame,
    *,
    allocator: VehicleTokenAllocator,
    instance_key: str,
    vehicle_model: str,
    source_file: str,
    segment_start: int | None = None,
    segment_end: int | None = None,
) -> pd.DataFrame:
    """Attach opaque tokens and evaluation metadata to a scenario chunk."""
    out = chunk.copy()
    token = allocator.token_for(instance_key)
    out["vehicle_token"] = token
    out["scenario_vehicle_id"] = token
    out["vehicle_model"] = vehicle_model
    if source_file:
        out["source_trace"] = source_trace_name(str(source_file))
        out["source_file"] = source_file
    if segment_start is not None:
        out["segment_start"] = segment_start
    if segment_end is not None:
        out["segment_end"] = segment_end
    return out


def build_offline_identity_mapping(scenario_df: pd.DataFrame) -> pd.DataFrame:
    """Offline evaluation mapping — never fed to fleet-layer features."""
    cols = [
        c
        for c in (
            "vehicle_token",
            "scenario_vehicle_id",
            "vehicle_model",
            "source_trace",
            "source_file",
            "segment_start",
            "segment_end",
        )
        if c in scenario_df.columns
    ]
    if not cols:
        return pd.DataFrame()
    key_cols = [c for c in ("vehicle_token", "source_file", "segment_start", "segment_end") if c in cols]
    return scenario_df[cols].drop_duplicates(subset=key_cols or cols).reset_index(drop=True)


def count_attacked_vehicle_instances(
    df: pd.DataFrame,
    *,
    malicious_column: str = "ground_truth_malicious",
) -> int:
    """Count distinct attacked vehicle instances using opaque tokens."""
    col = resolve_graph_vehicle_column(df) if malicious_column in df.columns else "vehicle_token"
    if col not in df.columns:
        return 0
    mal = df[df[malicious_column] == 1] if malicious_column in df.columns else df
    return int(mal[col].nunique())


def attach_opaque_tokens_for_audit(descriptors: pd.DataFrame) -> pd.DataFrame:
    """Assign opaque vehicle_token per distinct source_file group (audit fixtures only)."""
    if "vehicle_token" in descriptors.columns and descriptors["vehicle_token"].notna().all():
        return descriptors
    allocator = VehicleTokenAllocator()
    parts: list[pd.DataFrame] = []
    group_cols = [c for c in ("source_file", "vehicle_model") if c in descriptors.columns]
    if not group_cols:
        group_cols = ["vehicle_model"] if "vehicle_model" in descriptors.columns else []
    if not group_cols:
        out = descriptors.copy()
        out["vehicle_token"] = allocator.token_for("default")
        out["scenario_vehicle_id"] = out["vehicle_token"]
        return out
    for i, (_, grp) in enumerate(descriptors.groupby(group_cols, sort=False)):
        vm = str(grp["vehicle_model"].iloc[0]) if "vehicle_model" in grp.columns else "unknown"
        sf = str(grp["source_file"].iloc[0]) if "source_file" in grp.columns else ""
        parts.append(
            assign_identity_to_chunk(
                grp,
                allocator=allocator,
                instance_key=f"audit-{i}",
                vehicle_model=vm,
                source_file=sf,
            )
        )
    return pd.concat(parts, ignore_index=True)


def count_vehicle_model_diversity(df: pd.DataFrame, *, malicious_only: bool = True) -> int:
    """Evaluation-only model diversity (offline metadata)."""
    if "vehicle_model" not in df.columns:
        return 0
    sub = df
    if malicious_only and "ground_truth_malicious" in df.columns:
        sub = df[df["ground_truth_malicious"] == 1]
    return int(sub["vehicle_model"].nunique())
