"""Publication-safe metric extraction with N/A for undefined campaign metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.final_gnn_fleet_decision_experiment import DECISION_COORDINATED
from src.experiments.campaign_evaluation import (
    compute_campaign_metrics,
    compute_event_metrics,
    compute_vehicle_metrics,
)
from src.experiments.scenario_registry import get_scenario

NA = "N/A"


def _fmt_mean_std(series: pd.Series, decimals: int = 3) -> str:
    s = series.dropna()
    if len(s) == 0:
        return NA
    if len(s) == 1:
        return f"{s.mean():.{decimals}f}"
    return f"{s.mean():.{decimals}f} $\\pm$ {s.std():.{decimals}f}"


def load_run_artifacts(run_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "metrics": pd.read_csv(run_dir / "run_level_metrics.csv"),
        "event": pd.read_csv(run_dir / "event_predictions.csv"),
        "vehicle": pd.read_csv(run_dir / "vehicle_predictions.csv"),
        "membership": pd.read_csv(run_dir / "scenario_membership.csv"),
        "campaign": pd.read_csv(run_dir / "campaign_predictions.csv"),
        "graph": pd.read_csv(run_dir / "graph_statistics.csv") if (run_dir / "graph_statistics.csv").exists() else pd.DataFrame(),
        "runtime": pd.read_json(run_dir / "runtime_memory.json", typ="series") if (run_dir / "runtime_memory.json").exists() else pd.Series(dtype=float),
    }


def recompute_run_metrics(run_dir: Path) -> dict[str, Any]:
    arts = load_run_artifacts(run_dir)
    scenario_key = str(arts["metrics"].iloc[0]["scenario_key"])
    spec = get_scenario(scenario_key)
    ev = compute_event_metrics(arts["event"])
    veh = compute_vehicle_metrics(arts["vehicle"])
    cluster_df = pd.DataFrame()
    if "cluster_id" in arts["event"].columns:
        for cid in arts["event"]["cluster_id"].unique():
            if int(cid) < 0:
                continue
            mask = arts["event"]["cluster_id"] == cid
            cluster_df = pd.concat(
                [
                    cluster_df,
                    pd.DataFrame(
                        [
                            {
                                "cluster_id": cid,
                                "is_qualifying_campaign_cluster": bool(
                                    (arts["event"].loc[mask, "final_decision"] == DECISION_COORDINATED).any()
                                    and arts["event"].loc[mask, "vehicle_model"].nunique() >= 2
                                ),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    camp = compute_campaign_metrics(
        arts["event"], arts["membership"], cluster_df, spec.expect_coordinated_campaign
    )

    out = {**ev, **veh, **camp, "method": arts["metrics"].iloc[0]["method"], "seed": int(arts["metrics"].iloc[0]["seed"])}
    if not spec.expect_coordinated_campaign:
        out["campaign_precision"] = np.nan
        out["campaign_recall"] = np.nan
        out["campaign_f1"] = np.nan
        out["campaign_detection_rate"] = np.nan
    if spec.scenario_id == "S0":
        out["campaign_detection_rate"] = np.nan
    return out


def aggregate_validated_metrics(validated: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in validated.iterrows():
        run_dir = Path(row["run_dir"])
        m = recompute_run_metrics(run_dir)
        m.update(
            {
                "run_id": row["run_id"],
                "scenario_key": row["scenario_key"],
                "scenario_id": row.get("scenario_id", ""),
                "campaign_size": row["campaign_size"],
                "coordination_strength": row["coordination_strength"],
            }
        )
        rt_path = run_dir / "runtime_memory.json"
        if rt_path.exists():
            rt = pd.read_json(rt_path, typ="series")
            m["runtime_total_sec"] = float(rt.get("runtime_total_sec", rt.get("total_sec", np.nan)))
        rows.append(m)
    return pd.DataFrame(rows)
