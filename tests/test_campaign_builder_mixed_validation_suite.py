"""Unit/integration tests for experimental S0–S4 campaign builder.

Does NOT evaluate detection performance. Does NOT run GraphSAGE/DBSCAN/gate.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.experiments.final_shared_configuration import validation_scenarios as vs
from src.experiments.final_shared_configuration.validation_scenarios import (
    CAMPAIGN_SIZES,
    DESCRIPTORS_PER_VEHICLE,
    FLEET_SIZE,
    MAX_FLEET_NODES,
    THETA_STRONG,
    THETA_WEAK,
    VALIDATION_SEEDS,
    BuilderConfigurationError,
    DescriptorBudget,
    apply_coordination_strength,
    build_mixed_validation_suite,
    build_scenario_s0,
    build_scenario_s1,
    build_scenario_s2,
    build_scenario_s3,
    build_scenario_s4,
    compute_campaign_prototype,
)
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS


def _synth_pool(
    *,
    n_vehicles: int = 60,
    per_vehicle: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic validation descriptors covering strong/weak/benign bands."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    models = ["Hyundai", "Kia", "Chevrolet"]
    eid = 0
    for vi in range(n_vehicles):
        model = models[vi % 3]
        vtoken = f"V_{vi:04d}"
        # First third: benign-only vehicles; rest: mixed attack vehicles
        benign_only = vi < n_vehicles // 3
        for wi in range(per_vehicle):
            if benign_only or wi < per_vehicle // 2:
                attack = "benign"
                score = float(rng.uniform(0.05, 0.40))
            elif wi < (3 * per_vehicle) // 4:
                attack = "malfunction"
                score = float(rng.uniform(THETA_WEAK, THETA_STRONG - 1e-6))
            else:
                attack = "malfunction"
                score = float(rng.uniform(THETA_STRONG, 0.99))
            row = {
                "event_id": f"E{eid}",
                "window_id": wi,
                "window_index": wi,
                "vehicle_token": vtoken,
                "scenario_vehicle_id": vtoken,
                "source_vehicle": vtoken,
                "vehicle_model": model,
                "attack_type": attack,
                "anomaly_score": score,
                "source_file": f"{model}_{vtoken}.csv",
                "source_trace": f"{model}_{vtoken}.csv",
                "split": "validation",
            }
            for j, col in enumerate(BEHAVIOURAL_FEATURE_COLUMNS):
                row[col] = float(rng.normal(loc=j * 0.01 + vi * 0.1, scale=0.05))
            rows.append(row)
            eid += 1
    # Inject a few explicit TEST rows that must never be selected
    for wi in range(5):
        row = {
            "event_id": f"TEST_E{wi}",
            "window_id": wi,
            "window_index": wi,
            "vehicle_token": "TEST_VEH",
            "scenario_vehicle_id": "TEST_VEH",
            "source_vehicle": "TEST_VEH",
            "vehicle_model": "Hyundai",
            "attack_type": "malfunction",
            "anomaly_score": 0.95,
            "source_file": "test_trace.csv",
            "source_trace": "test_trace.csv",
            "split": "test",
        }
        for j, col in enumerate(BEHAVIOURAL_FEATURE_COLUMNS):
            row[col] = float(j)
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pool() -> pd.DataFrame:
    return _synth_pool()


def test_s0_contains_no_campaign_attack(pool: pd.DataFrame) -> None:
    df = build_scenario_s0(pool, seed=131)
    assert (df["ground_truth_malicious"] == 0).all()
    assert (df["ground_truth_campaign_member"] == 0).all()
    assert (df["GT_campaign_id"].astype(str) == "").all()
    assert df["vehicle_token"].nunique() == FLEET_SIZE
    assert len(df) == MAX_FLEET_NODES


def test_s1_attacks_one_vehicle(pool: pd.DataFrame) -> None:
    df = build_scenario_s1(pool, seed=137)
    isolated = df.loc[df["scenario_role"] == "isolated"]
    assert isolated["vehicle_token"].nunique() == 1
    assert (isolated["ground_truth_malicious"] == 1).sum() == 5
    assert df["vehicle_token"].nunique() == FLEET_SIZE
    assert len(df) == MAX_FLEET_NODES
    # No shared blend config
    assert float(df["experimental_coordination_strength"].iloc[0]) == 0.0


def test_s2_independent_without_shared_campaign_construction(pool: pd.DataFrame) -> None:
    df = build_scenario_s2(pool, seed=149, campaign_size=5)
    unrelated = df.loc[df["scenario_role"] == "unrelated"]
    ids = sorted(unrelated["GT_campaign_id"].astype(str).unique())
    assert len(ids) == 5
    assert float(df["experimental_coordination_strength"].iloc[0]) == 0.0
    # Behavioural vectors for unrelated vehicles should not all be identical
    # (no shared prototype blend applied)
    cols = [c for c in BEHAVIOURAL_FEATURE_COLUMNS if c in df.columns]
    mal = unrelated.loc[unrelated["ground_truth_malicious"] == 1, cols]
    # At least two distinct rows expected under no shared blend
    assert len(mal.drop_duplicates()) >= 2


@pytest.mark.parametrize("cs", CAMPAIGN_SIZES)
def test_s3_uses_strong_band(pool: pd.DataFrame, cs: int) -> None:
    df = build_scenario_s3(
        pool,
        seed=157,
        campaign_size=cs,
        experimental_coordination_strength=0.75,
    )
    mal = df.loc[
        (df["scenario_role"] == "coordinated") & (df["ground_truth_malicious"] == 1)
    ]
    assert (mal["anomaly_score"] >= THETA_STRONG).all()
    assert df["vehicle_token"].nunique() == FLEET_SIZE
    assert len(df) == MAX_FLEET_NODES
    assert int(df["campaign_size"].iloc[0]) == cs


@pytest.mark.parametrize("cs", CAMPAIGN_SIZES)
def test_s4_uses_weak_band(pool: pd.DataFrame, cs: int) -> None:
    df = build_scenario_s4(
        pool,
        seed=163,
        campaign_size=cs,
        experimental_coordination_strength=0.75,
    )
    mal = df.loc[
        (df["scenario_role"] == "coordinated") & (df["ground_truth_malicious"] == 1)
    ]
    assert (mal["anomaly_score"] >= THETA_WEAK).all()
    assert (mal["anomaly_score"] < THETA_STRONG).all()
    assert len(df) == MAX_FLEET_NODES


def test_fleet_and_node_budget(pool: pd.DataFrame) -> None:
    df = build_scenario_s0(pool, seed=179)
    assert df["vehicle_token"].nunique() == FLEET_SIZE
    assert (df.groupby("vehicle_token").size() == DESCRIPTORS_PER_VEHICLE).all()
    assert len(df) == MAX_FLEET_NODES == 200


def test_anomaly_score_unchanged_by_construction(pool: pd.DataFrame) -> None:
    # Capture pre-blend scores via building then comparing to pool
    df = build_scenario_s3(
        pool,
        seed=181,
        campaign_size=2,
        experimental_coordination_strength=1.0,
    )
    # Scores on constructed rows must equal original pool scores for same event_id
    merged = df.merge(
        pool[["event_id", "anomaly_score"]],
        on="event_id",
        suffixes=("_out", "_src"),
    )
    assert np.allclose(
        merged["anomaly_score_out"].to_numpy(),
        merged["anomaly_score_src"].to_numpy(),
    )


def test_identity_and_provenance_preserved(pool: pd.DataFrame) -> None:
    df = build_scenario_s3(
        pool,
        seed=191,
        campaign_size=2,
        experimental_coordination_strength=0.75,
    )
    for col in (
        "vehicle_token",
        "source_vehicle",
        "source_trace",
        "window_index",
        "attack_type",
        "event_id",
    ):
        assert col in df.columns
        assert df[col].notna().all()
    # GT campaign id shared for coordinated role
    camp = df.loc[df["scenario_role"] == "coordinated", "GT_campaign_id"].unique()
    assert len(camp) == 1 and str(camp[0]).startswith("CAMP-S3")


def test_only_behavioural_features_transformed(pool: pd.DataFrame) -> None:
    strength = 1.0
    df = build_scenario_s3(
        pool,
        seed=193,
        campaign_size=2,
        experimental_coordination_strength=strength,
    )
    coord = df.loc[df["scenario_role"] == "coordinated"]
    cols = [c for c in BEHAVIOURAL_FEATURE_COLUMNS if c in coord.columns]
    meta_cols = ["anomaly_score", "attack_type", "vehicle_token"]
    merged = coord.merge(
        pool[["event_id"] + meta_cols + cols],
        on="event_id",
        suffixes=("_out", "_src"),
    )
    assert np.allclose(merged["anomaly_score_out"], merged["anomaly_score_src"])
    assert (merged["attack_type_out"].astype(str) == merged["attack_type_src"].astype(str)).all()
    assert (merged["vehicle_token_out"].astype(str) == merged["vehicle_token_src"].astype(str)).all()
    changed = False
    for c in cols:
        if not np.allclose(merged[f"{c}_out"], merged[f"{c}_src"]):
            changed = True
            break
    assert changed


def test_test_data_cannot_enter_validation_construction(pool: pd.DataFrame) -> None:
    test_ids = {f"TEST_E{i}" for i in range(5)}
    df = build_scenario_s0(pool, seed=197, test_event_ids=test_ids)
    assert not df["event_id"].astype(str).isin(test_ids).any()
    assert not (df.get("split", pd.Series(dtype=str)).astype(str).str.lower() == "test").any()


def test_builder_requires_explicit_coordination_config(pool: pd.DataFrame) -> None:
    with pytest.raises(BuilderConfigurationError):
        build_mixed_validation_suite(
            pool,
            write_artifacts=False,
            scenarios=["S3"],
            validation_seeds=[131],
            campaign_sizes=[2],
            # experimental_coordination_strength intentionally omitted
        )


def test_builder_module_does_not_import_detector_stack() -> None:
    src = Path(vs.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
    forbidden = (
        "src.models",
        "src.graph",
        "torch",
        "torch_geometric",
        "sklearn.cluster",
    )
    for mod in imported:
        assert not any(mod == f or mod.startswith(f + ".") for f in forbidden), mod


def test_validation_seeds_constant() -> None:
    assert VALIDATION_SEEDS == [131, 137, 149, 157, 163, 179, 181, 191, 193, 197]


def test_suite_smoke_s0_s1_only(pool: pd.DataFrame, tmp_path: Path) -> None:
    scenarios, manifest = build_mixed_validation_suite(
        pool,
        val_out_dir=tmp_path,
        write_artifacts=True,
        scenarios=["S0", "S1"],
        validation_seeds=[131],
        test_event_ids={f"TEST_E{i}" for i in range(5)},
    )
    assert any(k.startswith("val_S0_") for k in scenarios)
    assert any(k.startswith("val_S1_") for k in scenarios)
    assert (tmp_path / "validation_manifest.csv").exists()
    assert manifest["validation_passed"].all()


def test_s3_s4_suite_with_explicit_config(pool: pd.DataFrame) -> None:
    scenarios, manifest = build_mixed_validation_suite(
        pool,
        write_artifacts=False,
        scenarios=["S3", "S4"],
        validation_seeds=[131],
        campaign_sizes=[2],
        experimental_coordination_strength=0.75,
        test_event_ids={f"TEST_E{i}" for i in range(5)},
    )
    assert len(scenarios) == 2
    assert set(manifest["experimental_coordination_strength_status"]) == {
        "EXPLICIT_EXPERIMENTAL_CONFIG"
    }
    assert (manifest["historical_coordination_strength"] == "UNRECOVERABLE").all()


def test_apply_coordination_does_not_touch_score() -> None:
    cols = list(BEHAVIOURAL_FEATURE_COLUMNS)
    df = pd.DataFrame(
        [
            {
                "event_id": "a",
                "anomaly_score": 0.9,
                "attack_type": "malfunction",
                **{c: 1.0 for c in cols},
            },
            {
                "event_id": "b",
                "anomaly_score": 0.85,
                "attack_type": "malfunction",
                **{c: 2.0 for c in cols},
            },
        ]
    )
    proto = compute_campaign_prototype(df, attack_type="malfunction")
    out, _ = apply_coordination_strength(
        df,
        strength=1.0,
        campaign_prototype=proto,
        target_mask=pd.Series([True, True]),
        seed=0,
    )
    assert list(out["anomaly_score"]) == [0.9, 0.85]
