"""Rebuild OCSLab publication test scenarios and compute vehicle-level IDS window metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]

SCENARIO_KEY_TO_ID = {
    "benign_fleet": "S0",
    "isolated_attack": "S1",
    "unrelated_incidents": "S2",
    "strong_campaign": "S3",
    "weak_campaign": "S4",
}


@dataclass(frozen=True)
class PublicationRunSpec:
    scenario_key: str
    scenario_id: str
    seed: int
    campaign_size: int
    attack_strength: str
    fleet_size: int


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return _safe_div(2.0 * precision * recall, precision + recall)


def compute_window_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "local_precision": precision,
        "local_recall": recall,
        "local_f1": f1_score(precision, recall),
        "local_fpr": _safe_div(fp, fp + tn),
        "window_tp": float(tp),
        "window_fp": float(fp),
        "window_fn": float(fn),
        "window_tn": float(tn),
        "n_windows": float(len(y_true)),
    }


def compute_window_metrics_from_alert_counts(
    *,
    malicious_windows: int,
    benign_windows: int,
    strong_alert_windows: int,
) -> dict[str, float]:
    """
    Derive window-level P/R/F1/FPR from per-window local_alert counts.

    ``strong_alert_windows`` is the publication-run ``strong_candidates`` field:
    number of scenario windows with ``anomaly_score >= theta_strong`` (local_alert=1).
    This is not descriptor promotion (``benign_incorrectly_promoted``).
    """
    mal = int(malicious_windows)
    ben = int(benign_windows)
    alerts = int(strong_alert_windows)

    if mal == 0:
        fp = alerts
        precision = np.nan
        recall = np.nan
        f1 = np.nan
        fpr = _safe_div(fp, ben) if ben else 0.0
        tp = 0
        fn = 0
    elif alerts >= mal:
        tp = mal
        fp = alerts - mal
        precision = _safe_div(tp, tp + fp)
        recall = 1.0
        f1 = f1_score(precision, recall)
        fpr = _safe_div(fp, ben) if ben else 0.0
        fn = 0
    else:
        tp = alerts
        fp = 0
        fn = mal - tp
        precision = 1.0 if tp else 0.0
        recall = _safe_div(tp, mal)
        f1 = f1_score(precision, recall)
        fpr = 0.0 if ben else 0.0

    return {
        "local_precision": float(precision) if not (isinstance(precision, float) and np.isnan(precision)) else np.nan,
        "local_recall": float(recall) if not (isinstance(recall, float) and np.isnan(recall)) else np.nan,
        "local_f1": float(f1) if not (isinstance(f1, float) and np.isnan(f1)) else np.nan,
        "local_fpr": float(fpr),
        "window_tp": float(tp),
        "window_fp": float(fp),
        "window_fn": float(fn),
        "window_tn": float(ben - fp) if ben else 0.0,
        "n_windows": float(mal + ben),
    }


def enumerate_publication_runs(campaign_metrics: pd.DataFrame) -> list[PublicationRunSpec]:
    specs: list[PublicationRunSpec] = []
    for _, row in campaign_metrics.iterrows():
        key = str(row.get("scenario_key", ""))
        if key not in SCENARIO_KEY_TO_ID:
            continue
        cs = int(row.get("campaign_size", 0))
        attack_strength = str(row.get("attack_strength", "n/a"))
        if attack_strength in ("", "n/a", "nan"):
            attack_strength = {
                "benign_fleet": "benign",
                "isolated_attack": "strong",
                "unrelated_incidents": "strong",
                "strong_campaign": "strong",
                "weak_campaign": "weak",
            }[key]
        specs.append(
            PublicationRunSpec(
                scenario_key=key,
                scenario_id=SCENARIO_KEY_TO_ID[key],
                seed=int(row["seed"]),
                campaign_size=cs,
                attack_strength=attack_strength,
                fleet_size=int(row.get("fleet_size", 20)),
            )
        )
    return specs


def _default_budget(fleet_size: int = 20):
    from src.experiments.campaign_analysis_corrected import DescriptorBudget

    return DescriptorBudget(
        descriptors_per_vehicle=10,
        malicious_per_attacked=5,
        benign_per_attacked=5,
        benign_per_benign=10,
        total_fleet_size=fleet_size,
    )


def _isolated_budget():
    from src.experiments.campaign_analysis_corrected import DescriptorBudget

    return DescriptorBudget(
        descriptors_per_vehicle=10,
        malicious_per_attacked=5,
        benign_per_attacked=5,
        benign_per_benign=10,
        total_fleet_size=2,
    )


def _append_chunk(
    rows: list[pd.DataFrame],
    chunk: pd.DataFrame,
    *,
    ground_truth_malicious: int,
) -> None:
    if chunk.empty:
        return
    part = chunk.copy()
    part["ground_truth_malicious"] = int(ground_truth_malicious)
    rows.append(part)


def build_publication_scenario_windows(
    spec: PublicationRunSpec,
    *,
    descriptors: pd.DataFrame,
    manifest: pd.DataFrame,
    catalog: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Reconstruct scenario windows (one row per window/event) for a publication run."""
    from src.experiments.campaign_analysis_corrected import generate_corrected_campaign_scenario
    from src.experiments.campaign_analysis_generator import (
        STRONG_ATTACK_DEFAULT,
        _attack_type_for_instance,
    )
    from src.experiments.data_splits import is_benign_attack_type
    from src.experiments.model_diversity_corrected.benign_fleet import BENIGN_FLEET_COMPOSITION
    from src.experiments.vehicle_instance_builder import (
        _build_attacked_vehicle_chunk,
        _segment_rows,
        _sample_n,
        select_instances_with_benign_composition,
    )

    local_cfg = config.get("local_ids", {})
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    rng = np.random.default_rng(spec.seed)

    if spec.scenario_key in ("strong_campaign", "weak_campaign"):
        attack_strength = "strong" if spec.scenario_key == "strong_campaign" else "weak"
        scenario_df, _, _, _, _ = generate_corrected_campaign_scenario(
            attack_strength=attack_strength,  # type: ignore[arg-type]
            seed=spec.seed,
            descriptors=descriptors,
            manifest=manifest,
            catalog=catalog,
            config=config,
            campaign_size=spec.campaign_size,
            coordination_strength=1.0,
            budget=_default_budget(),
        )
        return scenario_df

    if spec.scenario_key == "benign_fleet":
        budget = _default_budget()
        comp = BENIGN_FLEET_COMPOSITION
        benign_inst: list[dict[str, Any]] = []
        used: set[str] = set()
        for model, count in comp.items():
            pool = catalog[
                (catalog["vehicle_model"] == model)
                & catalog["attack_types_available"].eq("benign")
                & (catalog["benign_events"] >= budget.benign_per_benign)
                & ~catalog["scenario_vehicle_id"].isin(used)
            ]
            chosen = pool.sample(n=count, random_state=int(rng.integers(0, 2**31 - 1)))
            benign_inst.extend(chosen.to_dict("records"))
            used.update(chosen["scenario_vehicle_id"].astype(str).tolist())
        rows: list[pd.DataFrame] = []
        for inst in benign_inst:
            seg_df = _segment_rows(descriptors, manifest, inst)
            ben_pool = seg_df[seg_df["attack_type"].map(is_benign_attack_type)]
            chunk = _sample_n(ben_pool, budget.benign_per_benign, rng)
            _append_chunk(rows, chunk, ground_truth_malicious=0)
        return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])

    if spec.scenario_key == "isolated_attack":
        budget = _isolated_budget()
        comp = {"Hyundai": 1, "Kia": 0, "Chevrolet": 0}
        attacked_inst, benign_inst = select_instances_with_benign_composition(
            catalog,
            n_attacked=1,
            benign_model_composition={"Hyundai": 0, "Kia": 0, "Chevrolet": 1},
            attack_strength="strong",
            attack_type=STRONG_ATTACK_DEFAULT,
            model_composition=comp,
            seed=spec.seed,
            weak_threshold=weak_th,
            strong_threshold=strong_th,
            min_attack_events=budget.malicious_per_attacked,
            min_benign_on_benign=budget.benign_per_benign,
        )
        rows = []
        for inst in attacked_inst:
            seg_df = _segment_rows(descriptors, manifest, inst)
            atk_type = _attack_type_for_instance(inst, "strong")
            chunk = _build_attacked_vehicle_chunk(
                seg_df,
                attack_strength="strong",
                atk_type=atk_type,
                budget=budget,
                weak_th=weak_th,
                strong_th=strong_th,
                rng=rng,
            )
            rows.append(chunk)
        for inst in benign_inst:
            seg_df = _segment_rows(descriptors, manifest, inst)
            ben_pool = seg_df[seg_df["attack_type"].map(is_benign_attack_type)]
            chunk = _sample_n(ben_pool, budget.benign_per_benign, rng)
            _append_chunk(rows, chunk, ground_truth_malicious=0)
        return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])

    if spec.scenario_key == "unrelated_incidents":
        budget = _default_budget()
        comp = {"Hyundai": 3, "Kia": 2, "Chevrolet": 0}
        attacked_inst, benign_inst = select_instances_with_benign_composition(
            catalog,
            n_attacked=5,
            benign_model_composition=BENIGN_FLEET_COMPOSITION,
            attack_strength="strong",
            attack_type=STRONG_ATTACK_DEFAULT,
            model_composition=comp,
            seed=spec.seed + 17,
            weak_threshold=weak_th,
            strong_threshold=strong_th,
            min_attack_events=budget.malicious_per_attacked,
            min_benign_on_benign=budget.benign_per_benign,
        )
        rows = []
        for inst in attacked_inst:
            seg_df = _segment_rows(descriptors, manifest, inst)
            atk_type = _attack_type_for_instance(inst, "strong")
            chunk = _build_attacked_vehicle_chunk(
                seg_df,
                attack_strength="strong",
                atk_type=atk_type,
                budget=budget,
                weak_th=weak_th,
                strong_th=strong_th,
                rng=rng,
            )
            rows.append(chunk)
        for inst in benign_inst:
            seg_df = _segment_rows(descriptors, manifest, inst)
            ben_pool = seg_df[seg_df["attack_type"].map(is_benign_attack_type)]
            chunk = _sample_n(ben_pool, budget.benign_per_benign, rng)
            _append_chunk(rows, chunk, ground_truth_malicious=0)
        return pd.concat(rows, ignore_index=True).drop_duplicates(subset=["event_id"])

    raise ValueError(f"Unsupported scenario_key: {spec.scenario_key}")


def baseline_metrics_for_run(
    spec: PublicationRunSpec,
    campaign_row: pd.Series,
    *,
    descriptors: pd.DataFrame | None,
    manifest: pd.DataFrame | None,
    config: dict[str, Any],
    strong_threshold: float,
) -> dict[str, Any]:
    """Compute vehicle-level IDS window metrics for one publication scenario run."""
    if descriptors is not None and manifest is not None:
        from src.experiments.data_splits import is_benign_attack_type
        from src.experiments.vehicle_instance_builder import build_instance_catalog

        test_desc = descriptors[descriptors["split"] == "test"].copy()
        test_manifest = manifest[manifest["split"] == "test"].copy()
        catalog = build_instance_catalog(
            test_desc.drop(columns=["split"], errors="ignore"),
            test_manifest,
            weak_threshold=float(config.get("local_ids", {}).get("weak_threshold", 0.55)),
            strong_threshold=strong_threshold,
            min_windows_per_segment=10,
            target_split="test",
        )
        windows = build_publication_scenario_windows(
            spec,
            descriptors=test_desc,
            manifest=test_manifest,
            catalog=catalog,
            config=config,
        )
        if "ground_truth_malicious" not in windows.columns:
            windows["ground_truth_malicious"] = (
                ~windows["attack_type"].map(is_benign_attack_type)
            ).astype(int)
        scores = windows["anomaly_score"].astype(float).to_numpy()
        y_true = windows["ground_truth_malicious"].astype(int).to_numpy()
        y_pred = (scores >= strong_threshold).astype(int)
        metrics = compute_window_metrics_from_predictions(y_true, y_pred)
        source = "per_window_reconstruction"
    else:
        metrics = compute_window_metrics_from_alert_counts(
            malicious_windows=int(campaign_row["malicious_source_windows"]),
            benign_windows=int(campaign_row["benign_source_windows"]),
            strong_alert_windows=int(campaign_row["strong_candidates"]),
        )
        source = "per_window_local_alert_counts"

    metrics["metric_source"] = source
    metrics["local_alert_generated"] = bool(metrics.get("window_tp", 0) + metrics.get("window_fp", 0) > 0)
    return metrics


def load_publication_inputs(
    *,
    descriptors_path: Path | None,
    manifest_path: Path | None,
    git_ref: str = "origin/cursor/campaign-clustering",
    repo_root: Path,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    """Load descriptors and balanced window manifest if available."""
    import subprocess

    desc: pd.DataFrame | None = None
    manifest: pd.DataFrame | None = None
    notes: list[str] = []

    if descriptors_path and descriptors_path.exists():
        desc = pd.read_csv(descriptors_path)
        notes.append(f"descriptors: {descriptors_path}")
    else:
        notes.append("descriptors: not available (window reconstruction skipped)")

    if manifest_path and manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        notes.append(f"manifest: {manifest_path}")
    else:
        rel = "new_experiments/final_end_to_end_publication_run_balanced/manifests/balanced_window_manifest.csv"
        try:
            proc = subprocess.run(
                ["git", "show", f"{git_ref}:{rel}"],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            from io import StringIO

            manifest = pd.read_csv(StringIO(proc.stdout))
            notes.append(f"manifest: git show {git_ref}:{rel}")
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            notes.append(f"manifest: unavailable ({exc})")

    return desc, manifest, "; ".join(notes)
