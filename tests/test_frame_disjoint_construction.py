"""Construction-layer checks for frame-disjoint fleet building.

These tests isolate the logical-vehicle *construction* (segmentation + selection +
provenance + validators). They do NOT run descriptor coordination, GraphSAGE,
DBSCAN, or campaign decisions. Synthetic windows reproduce the real OCSLab
geometry: 100-frame windows at 50-frame stride (so consecutive windows share 50
frames), three physical vehicle models, benign (attack_free) and attack traces,
test split only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.frame_disjoint import (
    NodeBudget,
    build_fleet_provenance,
    find_source_frame_overlaps,
    run_construction_checks,
    PROVENANCE_COLUMNS,
)
from src.experiments.vehicle_instance_builder import (
    build_instance_catalog,
    select_instances_for_fleet,
)

WINDOW = 100
STRIDE = 50
MODELS = ("Hyundai", "Kia", "Chevrolet")


def _trace_windows(wid0, vehicle, source_file, attack_type, n, *, strong):
    """One trace's windows. attack segments alternate 5 malicious / 5 benign per 10."""
    d_rows, m_rows = [], []
    for i in range(n):
        wid = wid0 + i
        s, e = i * STRIDE, i * STRIDE + WINDOW
        if attack_type != "attack_free" and (i % 10) < 5:
            label, score, at = 1, (0.9 if strong else 0.6), attack_type
        else:
            label, score, at = 0, 0.1, "attack_free"
        d_rows.append(
            {
                "window_id": wid,
                "vehicle_model": vehicle,
                "source_file": source_file,
                "attack_type": at,
                "anomaly_score": score,
            }
        )
        m_rows.append(
            {
                "window_id": wid,
                "vehicle_model": vehicle,
                "source_file": source_file,
                "split": "test",
                "start_frame_idx": s,
                "end_frame_idx": e,
                "label": label,
            }
        )
    return d_rows, m_rows, wid0 + n


def _synthetic_pool():
    d_all, m_all, wid = [], [], 0
    for vm in MODELS:
        # benign-only trace (200 windows) -> many benign candidate vehicles
        d, m, wid = _trace_windows(wid, vm, f"/data/{vm}/Attack_free_{vm}.csv", "attack_free", 200, strong=True)
        d_all += d; m_all += m
        # attack trace -> attacked candidate vehicles (Hyundai/Kia larger, Chevrolet smaller)
        n_atk = 25 if vm == "Chevrolet" else 40
        d, m, wid = _trace_windows(wid, vm, f"/data/{vm}/Malfunction_{vm}.csv", "malfunction", n_atk, strong=True)
        d_all += d; m_all += m
    return pd.DataFrame(d_all), pd.DataFrame(m_all)


def _select_fleet(catalog, seed=11):
    return select_instances_for_fleet(
        catalog,
        n_attacked=5,
        n_benign=15,
        attack_strength="strong",
        attack_type="malfunction",
        model_composition={"Hyundai": 2, "Kia": 2, "Chevrolet": 1},
        seed=seed,
        strong_threshold=0.80,
        weak_threshold=0.55,
        min_attack_events=5,
        min_benign_on_attacked=0,
        min_benign_on_benign=10,
    )


def test_consecutive_windows_share_50_frames():
    # Geometry sanity: window i and i+1 overlap by exactly 50 frames.
    _, m = _synthetic_pool()
    g = m[m["source_file"] == "/data/Hyundai/Attack_free_Hyundai.csv"].sort_values("window_id")
    r0, r1 = g.iloc[0], g.iloc[1]
    assert r0["end_frame_idx"] - r1["start_frame_idx"] == 50


def test_historical_construction_permits_boundary_overlap():
    desc, man = _synthetic_pool()
    catalog = build_instance_catalog(
        desc, man, strong_threshold=0.80, min_windows_per_segment=10, enforce_frame_disjoint=False
    )
    overlaps = find_source_frame_overlaps(catalog)
    assert overlaps, "historical (ID-only) segmentation should permit source-frame overlap"
    # Each boundary overlap is exactly the one shared window = 50 frames.
    assert any(o["shared_frames"] == 50 for o in overlaps)


def test_frame_disjoint_construction_prevents_overlap():
    desc, man = _synthetic_pool()
    catalog = build_instance_catalog(
        desc, man, strong_threshold=0.80, min_windows_per_segment=10, enforce_frame_disjoint=True
    )
    assert find_source_frame_overlaps(catalog) == []


def test_fleet_feasible_and_checks_pass_under_frame_disjoint():
    desc, man = _synthetic_pool()
    catalog = build_instance_catalog(
        desc, man, strong_threshold=0.80, min_windows_per_segment=10, enforce_frame_disjoint=True
    )
    attacked, benign = _select_fleet(catalog)
    assert len(attacked) == 5 and len(benign) == 15

    budget = NodeBudget()
    report = run_construction_checks(attacked, benign, budget=budget, manifest=man, attack_strength="strong")
    assert report["ok"], report
    assert report["expected_total_nodes"] == 200
    assert report["participants"] == 20
    for name in ("duplicate_window_ids", "source_frame_overlap", "train_test_leakage", "participant_count", "node_count"):
        assert report[name]["ok"], (name, report[name]["errors"])


def test_provenance_output_shape_and_content():
    desc, man = _synthetic_pool()
    catalog = build_instance_catalog(
        desc, man, strong_threshold=0.80, min_windows_per_segment=10, enforce_frame_disjoint=True
    )
    attacked, benign = _select_fleet(catalog)
    prov = build_fleet_provenance(attacked + benign, man, seed=11, scenario="strong_campaign")
    assert list(prov.columns) == PROVENANCE_COLUMNS
    assert len(prov) == 20 * 10  # 20 logical vehicles x 10 windows
    assert prov["scenario_vehicle_id"].nunique() == 20
    assert prov["start_frame_idx"].notna().all()
    assert prov["end_frame_idx"].notna().all()
    assert prov["attack_label"].notna().all()
    assert set(prov["vehicle_model"].unique()) <= set(MODELS)


def test_frame_disjoint_preserves_most_data():
    # Frame-disjoint drops at most ~1 boundary window per segment; candidate pool stays ample.
    desc, man = _synthetic_pool()
    hist = build_instance_catalog(desc, man, strong_threshold=0.80, min_windows_per_segment=10, enforce_frame_disjoint=False)
    disj = build_instance_catalog(desc, man, strong_threshold=0.80, min_windows_per_segment=10, enforce_frame_disjoint=True)
    # Still enough benign (>=15) and attacked (>=5) candidates to build the fleet.
    assert (disj["attack_types_available"] == "benign").sum() >= 15
    assert (disj["attack_types_available"] != "benign").sum() >= 5
    # Candidate count shrinks only modestly (guard drops 1 window per boundary).
    assert len(disj) >= len(hist) - len(hist["source_file"].unique()) * 2
