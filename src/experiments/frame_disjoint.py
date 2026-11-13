"""Source-frame overlap validation, provenance, and construction checks.

This module isolates the *frame-disjoint construction* concern. It does not touch
descriptor coordination, GraphSAGE, DBSCAN, or campaign-decision logic. It operates
on the logical-vehicle instances produced by
``vehicle_instance_builder.build_instance_catalog`` /
``select_instances_for_fleet`` (dict records or a DataFrame), using the half-open
frame intervals ``[start_frame_idx, end_frame_idx)`` carried on each instance as
``window_frames`` (and, as a fallback, a ``window_id -> (start, end)`` lookup).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

Interval = tuple[int, int]
Instance = Mapping[str, Any]


@dataclass(frozen=True)
class NodeBudget:
    """Lightweight budget mirror (avoids importing the coordination chain)."""

    descriptors_per_vehicle: int = 10
    malicious_per_attacked: int = 5
    benign_per_attacked: int = 5
    benign_per_benign: int = 10
    total_fleet_size: int = 20

    @property
    def expected_total_nodes(self) -> int:
        return self.total_fleet_size * self.descriptors_per_vehicle


# --------------------------------------------------------------------------- #
# Frame-interval helpers (half-open [start, end))
# --------------------------------------------------------------------------- #
def _merge_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    """Coalesce overlapping/contiguous half-open intervals for coverage."""
    ivs = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
    merged: list[Interval] = []
    for s, e in ivs:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _shared_frames(a: Sequence[Interval], b: Sequence[Interval]) -> int:
    """Total number of frames shared by two interval sets (half-open)."""
    ma, mb = _merge_intervals(a), _merge_intervals(b)
    i = j = 0
    total = 0
    while i < len(ma) and j < len(mb):
        lo = max(ma[i][0], mb[j][0])
        hi = min(ma[i][1], mb[j][1])
        if hi > lo:
            total += hi - lo
        if ma[i][1] <= mb[j][1]:
            i += 1
        else:
            j += 1
    return total


def _instance_intervals(
    inst: Instance,
    *,
    frame_lookup: Mapping[int, Interval] | None,
) -> list[Interval]:
    wf = inst.get("window_frames")
    if wf:
        return [(int(s), int(e)) for s, e in wf]
    if frame_lookup is not None:
        out: list[Interval] = []
        for wid in inst.get("window_ids", []) or []:
            iv = frame_lookup.get(int(wid))
            if iv is not None:
                out.append((int(iv[0]), int(iv[1])))
        return out
    raise ValueError(
        f"Instance {inst.get('scenario_vehicle_id')!r} has no 'window_frames' and no "
        "frame_lookup was supplied; cannot validate source-frame overlap."
    )


def _as_records(instances: pd.DataFrame | Sequence[Instance]) -> list[dict[str, Any]]:
    if isinstance(instances, pd.DataFrame):
        return instances.to_dict("records")
    return [dict(i) for i in instances]


# --------------------------------------------------------------------------- #
# Source-frame overlap validator (Requirement 3)
# --------------------------------------------------------------------------- #
def find_source_frame_overlaps(
    instances: pd.DataFrame | Sequence[Instance],
    *,
    frame_lookup: Mapping[int, Interval] | None = None,
) -> list[dict[str, Any]]:
    """Return cross-vehicle source-frame overlaps.

    Two different logical vehicles overlap only when their ``source_file`` and
    ``vehicle_model`` match and their half-open frame intervals intersect. Each
    returned record reports the pair and the number of shared frames.
    """
    recs = _as_records(instances)
    overlaps: list[dict[str, Any]] = []
    for x in range(len(recs)):
        for y in range(x + 1, len(recs)):
            a, b = recs[x], recs[y]
            if a.get("scenario_vehicle_id") == b.get("scenario_vehicle_id"):
                continue
            if str(a.get("source_file")) != str(b.get("source_file")):
                continue
            if str(a.get("vehicle_model")) != str(b.get("vehicle_model")):
                continue
            shared = _shared_frames(
                _instance_intervals(a, frame_lookup=frame_lookup),
                _instance_intervals(b, frame_lookup=frame_lookup),
            )
            if shared > 0:
                overlaps.append(
                    {
                        "vehicle_a": a.get("scenario_vehicle_id"),
                        "vehicle_b": b.get("scenario_vehicle_id"),
                        "vehicle_model": a.get("vehicle_model"),
                        "source_file": a.get("source_file"),
                        "shared_frames": int(shared),
                    }
                )
    return overlaps


def validate_no_source_frame_overlap(
    instances: pd.DataFrame | Sequence[Instance],
    *,
    frame_lookup: Mapping[int, Interval] | None = None,
) -> list[str]:
    """Return error strings for any cross-vehicle source-frame overlap (empty if none)."""
    return [
        f"Source-frame overlap between {o['vehicle_a']} and {o['vehicle_b']} "
        f"({o['vehicle_model']} / {Path(str(o['source_file'])).name}): "
        f"{o['shared_frames']} shared frames"
        for o in find_source_frame_overlaps(instances, frame_lookup=frame_lookup)
    ]


# --------------------------------------------------------------------------- #
# Per-run provenance (Requirement 6)
# --------------------------------------------------------------------------- #
PROVENANCE_COLUMNS = [
    "seed",
    "scenario",
    "scenario_vehicle_id",
    "vehicle_model",
    "source_file",
    "window_id",
    "start_frame_idx",
    "end_frame_idx",
    "attack_label",
]


def build_fleet_provenance(
    instances: pd.DataFrame | Sequence[Instance],
    manifest: pd.DataFrame,
    *,
    seed: int,
    scenario: str,
) -> pd.DataFrame:
    """One row per (logical vehicle, window) with full source provenance.

    Columns: seed, scenario, scenario_vehicle_id, vehicle_model, source_file,
    window_id, start_frame_idx, end_frame_idx, attack_label.
    """
    recs = _as_records(instances)
    label_col = "label" if "label" in manifest.columns else None
    mkey = manifest.drop_duplicates(subset=["window_id"]).set_index("window_id")
    out: list[dict[str, Any]] = []
    for inst in recs:
        for wid in inst.get("window_ids", []) or []:
            wid = int(wid)
            row = mkey.loc[wid] if wid in mkey.index else None
            out.append(
                {
                    "seed": int(seed),
                    "scenario": str(scenario),
                    "scenario_vehicle_id": inst.get("scenario_vehicle_id"),
                    "vehicle_model": inst.get("vehicle_model"),
                    "source_file": inst.get("source_file"),
                    "window_id": wid,
                    "start_frame_idx": int(row["start_frame_idx"]) if row is not None else None,
                    "end_frame_idx": int(row["end_frame_idx"]) if row is not None else None,
                    "attack_label": (
                        int(row[label_col]) if (row is not None and label_col and pd.notna(row[label_col])) else None
                    ),
                }
            )
    return pd.DataFrame(out, columns=PROVENANCE_COLUMNS)


def write_fleet_provenance(df: pd.DataFrame, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


# --------------------------------------------------------------------------- #
# Construction checks (Requirement 7)
# --------------------------------------------------------------------------- #
def check_no_duplicate_window_ids(
    instances: pd.DataFrame | Sequence[Instance],
) -> list[str]:
    recs = _as_records(instances)
    seen: dict[int, str] = {}
    errors: list[str] = []
    for inst in recs:
        sid = inst.get("scenario_vehicle_id")
        for wid in inst.get("window_ids", []) or []:
            wid = int(wid)
            if wid in seen and seen[wid] != sid:
                errors.append(f"window_id {wid} assigned to both {seen[wid]} and {sid}")
            seen[wid] = sid
    return errors


def check_no_train_test_leakage(
    instances: pd.DataFrame | Sequence[Instance],
    *,
    manifest: pd.DataFrame | None = None,
    expected_split: str = "test",
) -> list[str]:
    recs = _as_records(instances)
    errors: list[str] = []
    for inst in recs:
        if str(inst.get("split", expected_split)) != expected_split:
            errors.append(
                f"{inst.get('scenario_vehicle_id')} is split={inst.get('split')} (expected {expected_split})"
            )
    if manifest is not None and "split" in manifest.columns:
        split_by_wid = manifest.drop_duplicates(subset=["window_id"]).set_index("window_id")["split"].to_dict()
        for inst in recs:
            for wid in inst.get("window_ids", []) or []:
                sp = str(split_by_wid.get(int(wid), expected_split))
                if sp != expected_split:
                    errors.append(
                        f"{inst.get('scenario_vehicle_id')} window {wid} is in '{sp}' split (expected {expected_split})"
                    )
    return errors


def check_participant_count(
    instances: pd.DataFrame | Sequence[Instance],
    *,
    expected: int,
) -> list[str]:
    n = len(_as_records(instances))
    return [] if n == expected else [f"Expected {expected} logical vehicles, got {n}"]


def check_node_count(
    attacked: Sequence[Instance],
    benign: Sequence[Instance],
    *,
    budget: NodeBudget,
    attack_strength: str = "strong",
) -> tuple[int, list[str]]:
    """Return (expected_total_nodes, errors). Verifies per-vehicle budget feasibility."""
    errors: list[str] = []
    atk_col = "strong_attack_events" if attack_strength == "strong" else "weak_attack_events"
    for inst in attacked:
        if int(inst.get(atk_col, 0)) < budget.malicious_per_attacked:
            errors.append(
                f"{inst.get('scenario_vehicle_id')} has {inst.get(atk_col, 0)} {attack_strength} events "
                f"< malicious_per_attacked={budget.malicious_per_attacked}"
            )
        if int(inst.get("benign_events", 0)) < budget.benign_per_attacked:
            errors.append(
                f"{inst.get('scenario_vehicle_id')} has {inst.get('benign_events', 0)} benign events "
                f"< benign_per_attacked={budget.benign_per_attacked}"
            )
    for inst in benign:
        if int(inst.get("benign_events", 0)) < budget.benign_per_benign:
            errors.append(
                f"{inst.get('scenario_vehicle_id')} has {inst.get('benign_events', 0)} benign events "
                f"< benign_per_benign={budget.benign_per_benign}"
            )
    return budget.expected_total_nodes, errors


def run_construction_checks(
    attacked: Sequence[Instance],
    benign: Sequence[Instance],
    *,
    budget: NodeBudget,
    manifest: pd.DataFrame | None = None,
    attack_strength: str = "strong",
    frame_lookup: Mapping[int, Interval] | None = None,
) -> dict[str, Any]:
    """Aggregate all construction checks into a report dict."""
    fleet = list(attacked) + list(benign)
    expected_nodes, node_errors = check_node_count(
        attacked, benign, budget=budget, attack_strength=attack_strength
    )
    checks = {
        "duplicate_window_ids": check_no_duplicate_window_ids(fleet),
        "source_frame_overlap": validate_no_source_frame_overlap(fleet, frame_lookup=frame_lookup),
        "train_test_leakage": check_no_train_test_leakage(fleet, manifest=manifest),
        "participant_count": check_participant_count(fleet, expected=budget.total_fleet_size),
        "node_count": node_errors,
    }
    report = {name: {"ok": not errs, "errors": errs} for name, errs in checks.items()}
    report["expected_total_nodes"] = expected_nodes
    report["participants"] = len(fleet)
    report["ok"] = all(v["ok"] for v in report.values() if isinstance(v, dict))
    return report
