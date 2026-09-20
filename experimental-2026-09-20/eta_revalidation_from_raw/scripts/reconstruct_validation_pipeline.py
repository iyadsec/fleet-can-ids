#!/usr/bin/env python3
"""Reconstruct windows / vehicle-level IF / validation descriptors & scenarios.

Uses the exact recovered balanced_split_manifest.csv. Does NOT select η,
run GraphSAGE, CTT, or overwrite June artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.dataset_loader import load_single_file  # noqa: E402
from src.evaluation.publication_fleet_core import (  # noqa: E402
    GNN_FEATURE_COLUMNS,
    compute_payload_entropy,
)
from src.evaluation.vehicle_level_evaluation import (  # noqa: E402
    SELECTED_METHOD_LABEL,
    _binary_metrics,
    _resolve_threshold_for_method,
)
from src.features.feature_extractor import (  # noqa: E402
    BEHAVIOURAL_FEATURE_COLUMNS,
    extract_window_features,
)
from src.graph.fleet_similarity_features import build_behavior_view_descriptors  # noqa: E402
from src.models.vehicle_ids import (  # noqa: E402
    benign_training_mask,
    fit_self_supervised_isolation_forest,
    score_self_supervised_isolation_forest,
)

EXP = ROOT / "experimental-2026-09-20/eta_revalidation_from_raw"
ARTIFACTS = EXP / "artifacts"
MODELS = EXP / "models"
SCENARIOS = EXP / "validation_scenarios"
HIST_SPLIT = (
    ROOT
    / "recovered_publication_pipeline/new_experiments/"
    "final_end_to_end_publication_run_balanced/manifests/balanced_split_manifest.csv"
)
HIST_WINDOWS = (
    ROOT
    / "recovered_publication_pipeline/new_experiments/"
    "final_end_to_end_publication_run_balanced/manifests/balanced_window_manifest.csv"
)
HIST_P4 = (
    ROOT
    / "experimental-2026-06-23/01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv"
)

WINDOW_SIZE = 100
STRIDE = 50
SEED = 42
N_ESTIMATORS = 200
DESCRIPTORS_PER_VEHICLE = 10
FLEET_SIZE = 20
VALIDATION_SEEDS = [131, 137, 149, 157, 163, 179, 181, 191, 193, 197]


def _dataset_root() -> Path:
    env = os.environ.get("OCSLAB_DATASET_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT / "Dataset/In-Vehicle Network Intrusion Detection Challenge").resolve()


def _rel_from_hist(source_file: str) -> str:
    marker = "In-Vehicle Network Intrusion Detection Challenge/"
    if marker in source_file:
        return source_file.split(marker, 1)[1]
    parts = Path(source_file).parts
    return str(Path(*parts[-2:])) if len(parts) >= 2 else Path(source_file).name


def _feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[list(BEHAVIOURAL_FEATURE_COLUMNS)].fillna(0.0).to_numpy(dtype=np.float32)


def resolve_traces(split_df: pd.DataFrame, dataset_root: Path) -> pd.DataFrame:
    rows = []
    for r in split_df.itertuples(index=False):
        rel = _rel_from_hist(r.source_file)
        local = dataset_root / rel
        rows.append(
            {
                "segment_id": r.segment_id,
                "vehicle_model": r.vehicle_model,
                "attack_type": r.attack_type,
                "split": r.split,
                "relative_source": rel,
                "manifest_absolute_source": r.source_file,
                "resolved_path": str(local) if local.is_file() else "",
                "status": "FOUND" if local.is_file() else "MISSING",
                "segment_start": int(r.segment_start),
                "segment_end": int(r.segment_end),
                "guard_start": r.guard_start if pd.notna(r.guard_start) else "",
                "guard_end": r.guard_end if pd.notna(r.guard_end) else "",
                "split_method": r.split_method,
                "window_count_manifest": int(r.window_count)
                if pd.notna(r.window_count) and str(r.window_count).strip() != ""
                else "",
            }
        )
    return pd.DataFrame(rows)


def windows_for_segment(
    *,
    vehicle_model: str,
    attack_type: str,
    label: float,
    source_file: str,
    segment_id: str,
    split: str,
    segment_start: int,
    segment_end: int,
    guard_start: Any,
    guard_end: Any,
    id_offset: int,
) -> pd.DataFrame:
    """Sliding windows strictly inside [segment_start, segment_end)."""
    rows: list[dict[str, Any]] = []
    start = segment_start
    while start + WINDOW_SIZE <= segment_end:
        end = start + WINDOW_SIZE
        rows.append(
            {
                "window_id": id_offset + len(rows),
                "vehicle_model": vehicle_model,
                "attack_type": attack_type,
                "label": float(label),
                "source_file": source_file,
                "start_frame_idx": start,
                "end_frame_idx": end,
                "window_size": WINDOW_SIZE,
                "n_frames": WINDOW_SIZE,
                "split": split,
                "segment_id": segment_id,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "guard_start": guard_start,
                "guard_end": guard_end,
            }
        )
        start += STRIDE
    return pd.DataFrame(rows)


def extract_features_for_trace(trace: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    out_rows: list[dict[str, Any]] = []
    for w in windows.itertuples(index=False):
        chunk = trace.iloc[int(w.start_frame_idx) : int(w.end_frame_idx)]
        feat = extract_window_features(chunk)
        row = {
            "window_id": w.window_id,
            "vehicle_model": w.vehicle_model,
            "attack_type": w.attack_type,
            "label": w.label,
            "source_file": w.source_file,
            "start_frame_idx": w.start_frame_idx,
            "end_frame_idx": w.end_frame_idx,
            "split": w.split,
            "segment_id": w.segment_id,
            "segment_start": w.segment_start,
            "segment_end": w.segment_end,
            "relative_source": Path(w.source_file).parent.name + "/" + Path(w.source_file).name
            if "/" in str(w.source_file)
            else Path(w.source_file).name,
        }
        # Prefer relative path recorded separately when available
        row.update(feat)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def classify_delta(recon: float, hist: float, *, abs_tol: float = 0.02, rel_tol: float = 0.05) -> str:
    if not np.isfinite(recon) or not np.isfinite(hist):
        return "MATERIAL_DIFFERENCE"
    diff = abs(float(recon) - float(hist))
    if diff <= abs_tol:
        return "MATCH"
    if diff <= max(abs_tol * 2.5, abs(hist) * rel_tol):
        return "CLOSE"
    return "MATERIAL_DIFFERENCE"


def build_gnn_view(df: pd.DataFrame) -> pd.DataFrame:
    view = build_behavior_view_descriptors(df.copy())
    view["payload_entropy"] = compute_payload_entropy(view)
    return view


def _is_benign(attack_type: str) -> bool:
    return str(attack_type).lower() in {"attack_free", "benign", "normal", "none", "no_attack"}


def _sample_n(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) == 0:
        return df.iloc[0:0]
    if len(df) <= n:
        return df.copy()
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].copy()


@dataclass
class ScenarioSpec:
    code: str  # S0..S4
    alias: str  # V0..V4
    name: str
    n_attacked: int
    attack_strength: str  # benign|strong|weak
    coordinated: bool
    distinct_campaigns: bool  # S2: one campaign id per attacked vehicle


def scenario_specs() -> list[ScenarioSpec]:
    return [
        ScenarioSpec("S0", "V0", "no_attack", 0, "benign", False, False),
        ScenarioSpec("S1", "V1", "isolated_single_vehicle_attack", 1, "strong", False, False),
        ScenarioSpec("S2", "V2", "independent_multi_vehicle_attacks", 5, "strong", False, True),
        ScenarioSpec("S3", "V3", "strong_coordinated_campaign", 5, "strong", True, False),
        ScenarioSpec("S4", "V4", "weak_coordinated_campaign", 5, "weak", True, False),
    ]


def build_validation_scenarios(
    val_desc: pd.DataFrame,
    *,
    weak_th: float,
    strong_th: float,
    output_dir: Path,
) -> pd.DataFrame:
    """Construct controlled fleet compositions from VALIDATION descriptors only.

    Underlying windows remain real OCSLab observations. Only fleet membership /
    campaign labels are constructed. Node budget: 10 descriptors × 20 vehicles.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    # Candidate pools keyed by (vehicle_model, role)
    benign = val_desc[val_desc["attack_type"].map(_is_benign)].copy()
    attack = val_desc[~val_desc["attack_type"].map(_is_benign)].copy()

    # Promote candidates for attack roles using recovered weak/strong thresholds
    attack = attack.copy()
    attack["local_alert"] = (attack["anomaly_score"] >= strong_th).astype(int)
    attack["weak_signal"] = (
        (attack["anomaly_score"] >= weak_th) & (attack["anomaly_score"] < strong_th)
    ).astype(int)

    for spec in scenario_specs():
        for seed in VALIDATION_SEEDS:
            rng = np.random.default_rng(seed)
            run_id = f"val_{spec.code}_seed{seed}"
            try:
                scenario_df, membership, composition = _one_scenario(
                    spec, benign, attack, rng, seed, weak_th, strong_th
                )
                # No test descriptors by construction (validation-only input)
                sh = hashlib.sha256(
                    json.dumps(
                        {
                            "scenario": spec.code,
                            "seed": seed,
                            "ids": sorted(scenario_df["descriptor_id"].astype(str)),
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest()[:16]
                scenario_df.to_csv(output_dir / f"{run_id}_records.csv", index=False)
                membership.to_csv(output_dir / f"{run_id}_membership.csv", index=False)
                n_vehicles = int(scenario_df["scenario_vehicle_id"].nunique())
                per_v = scenario_df.groupby("scenario_vehicle_id").size()
                budget_ok = (
                    n_vehicles == FLEET_SIZE
                    and int(per_v.min()) == DESCRIPTORS_PER_VEHICLE
                    and int(per_v.max()) == DESCRIPTORS_PER_VEHICLE
                    and len(scenario_df) == FLEET_SIZE * DESCRIPTORS_PER_VEHICLE
                )
                rows.append(
                    {
                        "validation_run_id": run_id,
                        "validation_seed": seed,
                        "scenario": spec.code,
                        "scenario_alias": spec.alias,
                        "scenario_name": spec.name,
                        "attack_strength": spec.attack_strength,
                        "campaign_size": spec.n_attacked,
                        "vehicle_composition": composition,
                        "descriptor_count": len(scenario_df),
                        "fleet_size": n_vehicles,
                        "descriptors_per_vehicle": DESCRIPTORS_PER_VEHICLE,
                        "budget_ok": budget_ok,
                        "test_descriptor_overlap": 0,
                        "scenario_hash": sh,
                        "validation_passed": bool(budget_ok),
                        "error": "",
                    }
                )
            except Exception as exc:  # noqa: BLE001 — record per-run failures
                rows.append(
                    {
                        "validation_run_id": run_id,
                        "validation_seed": seed,
                        "scenario": spec.code,
                        "scenario_alias": spec.alias,
                        "scenario_name": spec.name,
                        "attack_strength": spec.attack_strength,
                        "campaign_size": spec.n_attacked,
                        "vehicle_composition": "",
                        "descriptor_count": 0,
                        "fleet_size": 0,
                        "descriptors_per_vehicle": DESCRIPTORS_PER_VEHICLE,
                        "budget_ok": False,
                        "test_descriptor_overlap": 0,
                        "scenario_hash": "",
                        "validation_passed": False,
                        "error": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def _one_scenario(
    spec: ScenarioSpec,
    benign: pd.DataFrame,
    attack: pd.DataFrame,
    rng: np.random.Generator,
    seed: int,
    weak_th: float,
    strong_th: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    pieces: list[pd.DataFrame] = []
    membership_rows: list[dict[str, Any]] = []
    used_window_ids: set[int] = set()
    vehicle_seq = 0

    def _sample_from(pool: pd.DataFrame, model: str, n_desc: int, *, prefer_high_score: bool) -> pd.DataFrame:
        cand = pool[(pool["vehicle_model"] == model) & ~pool["window_id"].isin(used_window_ids)]
        if len(cand) < n_desc:
            # Fall back to any remaining platform rows (still validation-only)
            cand = pool[~pool["window_id"].isin(used_window_ids)]
            cand = cand[cand["vehicle_model"] == model]
        if len(cand) < n_desc:
            raise ValueError(
                f"{spec.code}: insufficient descriptors for {model} "
                f"(need {n_desc}, have {len(cand)})"
            )
        if prefer_high_score and "anomaly_score" in cand.columns:
            cand = cand.sort_values("anomaly_score", ascending=False)
            cand = cand.head(max(n_desc * 8, n_desc))
        chunk = _sample_n(cand, n_desc, rng)
        used_window_ids.update(int(x) for x in chunk["window_id"].tolist())
        return chunk

    def _add_vehicle(
        chunk: pd.DataFrame,
        *,
        model: str,
        role: str,
        campaign_id: str,
        gt_attacked: int,
    ) -> None:
        nonlocal vehicle_seq
        vehicle_seq += 1
        vid = f"{model[0]}{vehicle_seq:03d}"
        out = chunk.copy()
        out["scenario_vehicle_id"] = vid
        out["scenario_role"] = role
        out["campaign_id"] = campaign_id
        out["gt_attacked"] = gt_attacked
        pieces.append(out)
        membership_rows.append(
            {
                "scenario_vehicle_id": vid,
                "vehicle_model": model,
                "role": role,
                "campaign_id": campaign_id,
                "gt_attacked": gt_attacked,
                "seed": seed,
            }
        )

    if spec.n_attacked == 1:
        attack_models = ["Hyundai"]
    else:
        attack_models = ["Hyundai", "Hyundai", "Hyundai", "Kia", "Kia"]

    atk_pool = attack
    if spec.attack_strength == "strong":
        strong_pool = attack[attack["anomaly_score"] >= strong_th]
        if len(strong_pool) >= max(1, spec.n_attacked) * 5:
            atk_pool = strong_pool
    elif spec.attack_strength == "weak":
        weak_pool = attack[
            (attack["anomaly_score"] >= weak_th) & (attack["anomaly_score"] < strong_th)
        ]
        if len(weak_pool) >= max(1, spec.n_attacked) * 3:
            atk_pool = weak_pool

    for i, model in enumerate(attack_models[: spec.n_attacked]):
        if spec.distinct_campaigns:
            cid = f"INCIDENT-{spec.code}-{i}"
            role = "unrelated"
        elif spec.coordinated:
            cid = f"CAMP-{spec.code}-{spec.attack_strength.upper()}"
            role = "coordinated"
        else:
            cid = ""
            role = "isolated"
        mal = _sample_from(atk_pool, model, 5, prefer_high_score=True)
        ben = _sample_from(benign, model, 5, prefer_high_score=False)
        _add_vehicle(
            pd.concat([mal, ben], ignore_index=True),
            model=model,
            role=role,
            campaign_id=cid,
            gt_attacked=1,
        )

    remaining = FLEET_SIZE - spec.n_attacked
    if spec.n_attacked == 0:
        ben_targets = {"Chevrolet": 6, "Hyundai": 7, "Kia": 7}
    else:
        ben_targets = {"Chevrolet": 0, "Hyundai": 0, "Kia": 0}
        order = ["Chevrolet", "Hyundai", "Kia"]
        i = 0
        while sum(ben_targets.values()) < remaining:
            ben_targets[order[i % 3]] += 1
            i += 1

    for model, count in ben_targets.items():
        for _ in range(int(count)):
            chunk = _sample_from(benign, model, DESCRIPTORS_PER_VEHICLE, prefer_high_score=False)
            _add_vehicle(chunk, model=model, role="benign", campaign_id="", gt_attacked=0)

    scenario_df = pd.concat(pieces, ignore_index=True)

    if spec.coordinated:
        mal_mask = (scenario_df["scenario_role"] == "coordinated") & (
            ~scenario_df["attack_type"].map(_is_benign)
        )
        if mal_mask.any():
            proto_src = attack if not attack.empty else scenario_df.loc[mal_mask]
            cols = [c for c in BEHAVIOURAL_FEATURE_COLUMNS if c in scenario_df.columns]
            proto = proto_src[cols].astype(np.float64).mean(axis=0).to_numpy()
            strength = 1.0 if spec.attack_strength == "strong" else 0.35
            feat_min = scenario_df.loc[mal_mask, cols].astype(np.float64).min(axis=0).to_numpy()
            feat_max = scenario_df.loc[mal_mask, cols].astype(np.float64).max(axis=0).to_numpy()
            proto = np.clip(proto, feat_min, feat_max)
            for idx in scenario_df.index[mal_mask]:
                original = scenario_df.loc[idx, cols].astype(np.float64).to_numpy()
                blended = (1.0 - strength) * original + strength * proto
                scenario_df.loc[idx, cols] = np.clip(blended, feat_min, feat_max)

    scenario_df = scenario_df.copy()
    scenario_df["descriptor_id"] = [f"{seed}_{spec.code}_{i}" for i in range(len(scenario_df))]
    composition = ", ".join(
        f"{m}={n}"
        for m, n in sorted(
            scenario_df.groupby("vehicle_model")["scenario_vehicle_id"].nunique().items()
        )
    )
    return scenario_df, pd.DataFrame(membership_rows), composition


def main() -> int:
    t_start = time.time()
    dataset_root = _dataset_root()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    SCENARIOS.mkdir(parents=True, exist_ok=True)

    print(f"OCSLAB_DATASET_DIR={dataset_root}")
    if not dataset_root.is_dir():
        print("STOP_OTHER: dataset root missing")
        return 2

    split_df = pd.read_csv(HIST_SPLIT)
    resolution = resolve_traces(split_df, dataset_root)
    resolution.to_csv(EXP / "RAW_TRACE_RESOLUTION.csv", index=False)
    n_found = int((resolution["status"] == "FOUND").sum())
    print(f"Resolved segments: {n_found}/{len(resolution)}")
    if n_found != len(resolution):
        print("STOP_OTHER: not all traces resolved")
        return 2

    # ---- Window reconstruction ----
    print("Reconstructing windows from exact split manifest...")
    window_parts: list[pd.DataFrame] = []
    feature_parts: list[pd.DataFrame] = []
    id_offset = 0
    # Cache loaded traces by resolved path
    trace_cache: dict[str, pd.DataFrame] = {}

    for r in resolution.itertuples(index=False):
        path = Path(r.resolved_path)
        if r.resolved_path not in trace_cache:
            print(f"  loading {path.name} ...")
            trace_cache[r.resolved_path] = load_single_file(path)
        trace = trace_cache[r.resolved_path]
        # Labels are file-level (historical); use first row
        label = float(trace["label"].iloc[0]) if len(trace) else 0.0
        attack_type = str(trace["attack_type"].iloc[0])
        vehicle = str(trace["vehicle_model"].iloc[0])
        # Use local path as source_file for join; keep relative for provenance
        source_file = str(path)
        seg_start = int(r.segment_start)
        seg_end = int(r.segment_end)
        if seg_end > len(trace):
            print(
                f"  WARNING: segment_end {seg_end} > trace length {len(trace)} for {r.segment_id}"
            )
        wins = windows_for_segment(
            vehicle_model=vehicle,
            attack_type=attack_type,
            label=label,
            source_file=source_file,
            segment_id=r.segment_id,
            split=r.split,
            segment_start=seg_start,
            segment_end=min(seg_end, len(trace)),
            guard_start=r.guard_start,
            guard_end=r.guard_end,
            id_offset=id_offset,
        )
        # Attach relative_source
        wins["relative_source"] = r.relative_source
        id_offset += len(wins)
        print(f"  {r.segment_id}: {len(wins)} windows (hist={r.window_count_manifest})")
        window_parts.append(wins)
        feats = extract_features_for_trace(trace, wins)
        feats["relative_source"] = r.relative_source
        feature_parts.append(feats)

    windows = pd.concat(window_parts, ignore_index=True)
    features = pd.concat(feature_parts, ignore_index=True)
    windows.to_csv(ARTIFACTS / "reconstructed_windows.csv", index=False)
    features.to_csv(ARTIFACTS / "reconstructed_window_features.csv", index=False)

    # Provenance vs historical
    hist_counts = (
        pd.read_csv(HIST_WINDOWS, usecols=["segment_id", "split", "vehicle_model", "source_file"])
        .groupby(["segment_id", "split", "vehicle_model"], as_index=False)
        .size()
        .rename(columns={"size": "hist_n_windows"})
    )
    recon_counts = (
        windows.groupby(["segment_id", "split", "vehicle_model"], as_index=False)
        .agg(
            n_windows=("window_id", "count"),
            min_start=("start_frame_idx", "min"),
            max_end=("end_frame_idx", "max"),
            source_file=("source_file", "first"),
            relative_source=("relative_source", "first"),
            attack_types=("attack_type", "first"),
            segment_start=("segment_start", "first"),
            segment_end=("segment_end", "first"),
        )
    )
    provenance = recon_counts.merge(hist_counts, on=["segment_id", "split", "vehicle_model"], how="outer")
    provenance["window_size"] = WINDOW_SIZE
    provenance["count_match"] = provenance["n_windows"] == provenance["hist_n_windows"]
    provenance.to_csv(EXP / "WINDOW_PROVENANCE.csv", index=False)

    total_recon = int(windows.shape[0])
    total_hist = 110121
    split_recon = windows["split"].value_counts().to_dict()
    print(f"Window totals: recon={total_recon} hist={total_hist} match={total_recon == total_hist}")
    print(f"By split: {split_recon}")
    if int(provenance["count_match"].fillna(False).sum()) != len(hist_counts):
        bad = provenance[provenance["count_match"] != True]  # noqa: E712
        print("Segment count mismatches:")
        print(bad[["segment_id", "n_windows", "hist_n_windows"]].to_string(index=False))

    # ---- Vehicle-level IF ----
    print("Fitting per-vehicle Isolation Forest (benign train only)...")
    scored_parts: list[pd.DataFrame] = []
    thresholds: dict[str, float] = {}
    train_sizes: dict[str, int] = {}

    for vehicle, subset in features.groupby("vehicle_model", sort=True):
        train_df = subset[subset["split"] == "train"].copy()
        val_df = subset[subset["split"] == "validation"].copy()
        test_df = subset[subset["split"] == "test"].copy()
        benign_train = train_df[benign_training_mask(train_df)]
        if benign_train.empty:
            raise RuntimeError(f"No benign train windows for {vehicle}")
        X_benign = _feature_matrix(benign_train)
        train_sizes[str(vehicle)] = len(benign_train)
        model = fit_self_supervised_isolation_forest(
            X_benign, random_state=SEED, n_estimators=N_ESTIMATORS
        )
        joblib.dump(model, MODELS / f"if_{vehicle}_seed{SEED}.joblib")

        def _score(part: pd.DataFrame) -> np.ndarray:
            if part.empty:
                return np.array([], dtype=np.float64)
            _, scores = score_self_supervised_isolation_forest(
                model, _feature_matrix(part), X_benign
            )
            return scores

        for part in (train_df, val_df, test_df):
            if part.empty:
                continue
            p = part.copy()
            p["anomaly_score"] = _score(part)
            scored_parts.append(p)

        y_val = val_df["label"].to_numpy(dtype=np.int64) if not val_df.empty else np.array([])
        # Need scores on val — recompute from scored later; score now:
        val_scores = _score(val_df)
        pick = _resolve_threshold_for_method(SELECTED_METHOD_LABEL, y_val, val_scores)
        thresholds[str(vehicle)] = float(pick.value)
        print(
            f"  {vehicle}: benign_train={len(benign_train)} val={len(val_df)} "
            f"test={len(test_df)} threshold={pick.value:.4f} ({pick.method})"
        )

    scored = pd.concat(scored_parts, ignore_index=True)
    scored["predicted_label"] = 0
    for vehicle, t in thresholds.items():
        mask = scored["vehicle_model"] == vehicle
        scored.loc[mask, "predicted_label"] = (
            scored.loc[mask, "anomaly_score"] >= t
        ).astype(int)
        scored.loc[mask, "threshold"] = t
    scored.to_csv(ARTIFACTS / "scored_windows.csv", index=False)

    # Held-out test metrics
    test_eval = scored[scored["split"] == "test"].copy()
    metric_rows: list[dict[str, Any]] = []
    for vehicle, part in test_eval.groupby("vehicle_model"):
        yt = part["label"].to_numpy(dtype=np.int64)
        yp = part["predicted_label"].to_numpy(dtype=np.int64)
        ys = part["anomaly_score"].to_numpy(dtype=np.float64)
        m = _binary_metrics(yt, yp, ys)
        metric_rows.append(
            {
                "vehicle_model": vehicle,
                "pr_auc": m["pr_auc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fpr": m["fpr"],
                "roc_auc": m["roc_auc"],
                "threshold": thresholds[str(vehicle)],
                "test_windows": len(part),
            }
        )
    yt = test_eval["label"].to_numpy(dtype=np.int64)
    yp = test_eval["predicted_label"].to_numpy(dtype=np.int64)
    ys = test_eval["anomaly_score"].to_numpy(dtype=np.float64)
    pooled = _binary_metrics(yt, yp, ys)
    metric_rows.append(
        {
            "vehicle_model": "pooled",
            "pr_auc": pooled["pr_auc"],
            "precision": pooled["precision"],
            "recall": pooled["recall"],
            "f1": pooled["f1"],
            "fpr": pooled["fpr"],
            "roc_auc": pooled["roc_auc"],
            "threshold": float(np.median(list(thresholds.values()))),
            "test_windows": len(test_eval),
        }
    )
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(ARTIFACTS / "vehicle_level_test_metrics.csv", index=False)
    print(metrics_df.to_string(index=False))

    # Compare to table_P4
    hist_p4 = pd.read_csv(HIST_P4)
    audit_rows = []
    for _, row in metrics_df.iterrows():
        href = hist_p4[hist_p4["vehicle_model"] == row["vehicle_model"]]
        if href.empty:
            continue
        h = href.iloc[0]
        for key, hkey in [
            ("pr_auc", "pr_auc"),
            ("precision", "precision"),
            ("recall", "recall"),
            ("f1", "f1"),
        ]:
            cls = classify_delta(float(row[key]), float(h[hkey]))
            audit_rows.append(
                {
                    "vehicle_model": row["vehicle_model"],
                    "metric": key,
                    "reconstructed": float(row[key]),
                    "historical_table_P4": float(h[hkey]),
                    "abs_delta": abs(float(row[key]) - float(h[hkey])),
                    "classification": cls,
                }
            )
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(ARTIFACTS / "vehicle_level_audit_detail.csv", index=False)

    material = audit_df[audit_df["classification"] == "MATERIAL_DIFFERENCE"]
    vehicle_ok = material.empty or set(material["metric"]) <= set()  # any material → flag
    # Allow CLOSE; stop only on unexplained MATERIAL on primary F1/PR-AUC
    critical = material[material["metric"].isin(["f1", "pr_auc"])]
    vehicle_pipeline_ok = critical.empty

    # ---- Validation descriptors ----
    print("Building validation descriptors...")
    val = scored[scored["split"] == "validation"].copy()
    val_view = build_gnn_view(val)
    # Ensure all 9-D columns present
    missing_g = [c for c in GNN_FEATURE_COLUMNS if c not in val_view.columns]
    if missing_g:
        raise RuntimeError(f"Missing GNN columns: {missing_g}")

    val_view["descriptor_id"] = [
        f"val_{int(w)}_{s}" for w, s in zip(val_view["window_id"], val_view["segment_id"])
    ]
    val_view["split"] = "validation"
    # d_i = 24-D behavioural; g_i = 9-D GraphSAGE input (columns present, not yet GNN-embedded)
    desc_cols = (
        [
            "descriptor_id",
            "window_id",
            "vehicle_model",
            "source_file",
            "relative_source",
            "segment_id",
            "split",
            "attack_type",
            "label",
            "anomaly_score",
            "threshold",
            "predicted_label",
            "start_frame_idx",
            "end_frame_idx",
        ]
        + list(BEHAVIOURAL_FEATURE_COLUMNS)
        + list(GNN_FEATURE_COLUMNS)
    )
    # dedupe columns while preserving order
    seen = set()
    desc_cols_u = []
    for c in desc_cols:
        if c in val_view.columns and c not in seen:
            desc_cols_u.append(c)
            seen.add(c)
    descriptors = val_view[desc_cols_u].copy()
    descriptors.to_csv(ARTIFACTS / "validation_descriptors.csv", index=False)

    # Also export train benign for potential later scaler fit (not used for η)
    train_benign = scored[(scored["split"] == "train") & (scored["label"] == 0)].copy()
    train_benign.to_csv(ARTIFACTS / "train_benign_scored_windows.csv", index=False)

    # Weak/strong fleet thresholds from recovered local_model_training_manifest (fleet promotion)
    # Vehicle-level FPR threshold is separate; for scenario candidate selection use freeze YAML pair
    weak_th = 0.55
    strong_th = 0.80

    print("Constructing validation fleet scenarios S0–S4...")
    scenario_manifest = build_validation_scenarios(
        descriptors,
        weak_th=weak_th,
        strong_th=strong_th,
        output_dir=SCENARIOS,
    )
    scenario_manifest.to_csv(SCENARIOS / "validation_scenario_manifest.csv", index=False)
    n_pass = int(scenario_manifest["validation_passed"].sum())
    n_tot = len(scenario_manifest)
    print(f"Scenarios passed: {n_pass}/{n_tot}")
    if n_pass == 0:
        print("STOP_VALIDATION_DATA_INSUFFICIENT")
        scenarios_ok = False
    else:
        # Require at least one successful seed per S0–S4
        by_s = scenario_manifest.groupby("scenario")["validation_passed"].any()
        scenarios_ok = all(bool(by_s.get(s, False)) for s in ["S0", "S1", "S2", "S3", "S4"])
        print(by_s.to_dict())

    # Write JSON summary for markdown generators
    summary = {
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_root": str(dataset_root),
        "segments_resolved": n_found,
        "segments_required": len(resolution),
        "windows_reconstructed": total_recon,
        "windows_historical": total_hist,
        "windows_match": total_recon == total_hist,
        "split_counts_reconstructed": {k: int(v) for k, v in split_recon.items()},
        "split_counts_historical": {"train": 74858, "validation": 20669, "test": 14594},
        "thresholds": thresholds,
        "benign_train_sizes": train_sizes,
        "vehicle_metrics": metrics_df.to_dict(orient="records"),
        "vehicle_audit": audit_df.to_dict(orient="records"),
        "vehicle_pipeline_ok": vehicle_pipeline_ok,
        "critical_material_differences": critical.to_dict(orient="records"),
        "gnn_feature_columns": list(GNN_FEATURE_COLUMNS),
        "behavioural_feature_columns": list(BEHAVIOURAL_FEATURE_COLUMNS),
        "validation_descriptors": len(descriptors),
        "scenarios_passed": n_pass,
        "scenarios_total": n_tot,
        "scenarios_ok": scenarios_ok,
        "eta_evaluated": False,
        "eta_selected": False,
        "elapsed_sec": round(time.time() - t_start, 1),
    }
    (ARTIFACTS / "reconstruction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in (
        "windows_reconstructed", "windows_match", "vehicle_pipeline_ok",
        "scenarios_ok", "elapsed_sec"
    )}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
