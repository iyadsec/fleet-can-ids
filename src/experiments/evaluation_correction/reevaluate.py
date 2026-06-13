"""Re-evaluate corrected Phase 3 runs with fixed decision logic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.evaluation_correction.metrics import (
    aggregate_corrected_run_metrics,
    campaign_error_breakdown_row,
    compute_event_confusion_row,
    compute_vehicle_detailed_metrics,
)
from src.experiments.evaluation_correction.promotion import (
    PromotionConfig,
    apply_corrected_event_decisions,
    tune_promotion_threshold_validation,
)
from src.experiments.experiment_pipeline import run_graph_method
from src.experiments.method_local_ids import run_local_ids_method
from src.experiments.result_writer import ExperimentRunContext

PHASE3_ROOT = Path("new_experiments/final_validated_runs/results/campaign_size_corrected/runs")


def _load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    return {
        "event_predictions": pd.read_csv(run_dir / "event_predictions.csv"),
        "membership": pd.read_csv(run_dir / "vehicle_membership.csv"),
        "metrics": pd.read_csv(run_dir / "run_level_metrics.csv").iloc[0].to_dict(),
        "runtime": json.loads((run_dir / "runtime_memory.json").read_text(encoding="utf-8")),
        "scenario": pd.read_csv(run_dir / "selected_source_records.csv"),
    }


def reevaluate_from_saved_predictions(
    run_dir: Path,
    *,
    promotion_cfg: PromotionConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    arts = _load_run_artifacts(run_dir)
    m = arts["metrics"]
    method = str(m.get("method", ""))
    attack_strength = str(m.get("attack_strength", "strong"))
    raw = arts["event_predictions"]
    corrected = apply_corrected_event_decisions(
        raw,
        attack_strength=attack_strength,  # type: ignore[arg-type]
        method=method,
        cfg=promotion_cfg,
    )
    cluster_df = pd.DataFrame()
    if (run_dir / "cluster_df.csv").exists():
        cluster_df = pd.read_csv(run_dir / "cluster_df.csv")

    row = compute_event_confusion_row(
        corrected,
        run_id=run_dir.name,
        seed=int(m.get("seed", -1)),
        method=method,
        attack_strength=attack_strength,
        campaign_size=int(m.get("campaign_size", 0)),
    )
    veh_detail = compute_vehicle_detailed_metrics(corrected, arts["membership"])
    row.update(veh_detail)
    return corrected, row


def collect_all_confusion_rows(
    runs_root: Path,
    promotion_cfg: PromotionConfig,
) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "event_predictions.csv").exists():
            continue
        _, row = reevaluate_from_saved_predictions(run_dir, promotion_cfg=promotion_cfg)
        rows.append(row)
    return pd.DataFrame(rows)


def rerun_graph_method_if_needed(
    run_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Re-run M2–M4 on frozen scenario when cluster artifacts missing."""
    arts = _load_run_artifacts(run_dir)
    m = arts["metrics"]
    method = str(m["method"])
    if method not in ("descriptor_clustering", "standard_gnn", "fcgnn"):
        return arts["event_predictions"]
    ctx = ExperimentRunContext(
        run_id=run_dir.name,
        scenario_key=f"campaign_size_corrected_{m['attack_strength']}",
        method=method,
        seed=int(m["seed"]),
        campaign_size=int(m["campaign_size"]),
        coordination_strength=float(m.get("coordination_strength", 1.0)),
        created_at="",
        output_root=run_dir.parent.parent.parent,
        run_dir=run_dir,
    )
    out = run_graph_method(
        arts["scenario"],
        arts["membership"],
        config,
        int(m["seed"]),
        method,  # type: ignore[arg-type]
    )
    return out.event_predictions
