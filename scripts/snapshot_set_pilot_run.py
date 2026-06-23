#!/usr/bin/env python3
"""Save snapshot of set_pilot metrics for before/after comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.set_pilot import set_work_root


def collect_snapshot(root: Path, set_id: str = "set_01", label: str = "v1") -> dict:
    snap: dict = {"label": label, "set_id": set_id}

    marker = root / "manifests" / f"stage_set_pilot_{set_id}_complete.json"
    if marker.exists():
        snap["marker"] = json.loads(marker.read_text())

    nm = root / "manifests" / "normalization_manifest.csv"
    if nm.exists():
        ndf = pd.read_csv(nm)
        snap["files_processed"] = len(ndf)
        snap["rows_processed"] = int(ndf["row_count"].sum()) if "row_count" in ndf else None
        snap["subsets_in_manifest"] = sorted(ndf["subset_name"].unique().tolist())

    wm = root / "manifests" / "window_manifest.csv"
    if wm.exists():
        wdf = pd.read_csv(wm)
        snap["windows"] = len(wdf)
        snap["subsets_in_windows"] = sorted(wdf["subset_name"].unique().tolist())
        snap["vehicles_in_windows"] = sorted(wdf["vehicle_id"].unique().tolist())

    desc = root / "descriptors" / f"{set_id}_fleet_candidate_descriptors.csv"
    if not desc.exists():
        desc = root / "descriptors" / "fleet_candidate_descriptors.csv"
    if desc.exists():
        ddf = pd.read_csv(desc)
        snap["descriptors"] = len(ddf)
        snap["descriptor_vehicles"] = sorted(ddf["vehicle_id"].unique().tolist())
        snap["descriptor_vehicle_counts"] = ddf.groupby("vehicle_id").size().to_dict()

    gs = root / "graph" / f"{set_id}_graph_statistics.csv"
    if gs.exists():
        g = pd.read_csv(gs).iloc[0].to_dict()
        snap["graph_nodes"] = int(g.get("num_nodes", 0))
        snap["graph_edges"] = int(g.get("num_edges", 0))
        snap["cross_vehicle_edge_pct"] = float(g.get("cross_vehicle_edge_pct", 0))

    scen = root / "results" / "scenario_evaluation" / f"{set_id}_run_level_metrics.csv"
    if not scen.exists():
        scen = root / "results" / "scenario_evaluation" / "run_level_metrics.csv"
    if scen.exists():
        sdf = pd.read_csv(scen)
        snap["scenario_means"] = (
            sdf.groupby("scenario")[
                [
                    "local_or_incident_detected",
                    "fleet_campaign_detected",
                    "false_campaign",
                    "incorrect_merge_rate",
                    "campaign_f1",
                ]
            ]
            .mean()
            .round(4)
            .to_dict()
        )

    comm = root / "results" / "descriptor_transfer" / "communication_summary.csv"
    if comm.exists():
        snap["descriptor_rate"] = float(pd.read_csv(comm).iloc[0].get("candidate_transmission_rate", 0))

    return snap


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--set-id", default="set_01")
    parser.add_argument("--label", default="v1")
    parser.add_argument("--output-name", default=None)
    args = parser.parse_args()

    from src.ctt.constants import OUTPUT_ROOT

    root = set_work_root(OUTPUT_ROOT, args.set_id)
    snap = collect_snapshot(root, args.set_id, args.label)
    out = root / "validation" / (args.output_name or f"run_{args.label}_snapshot.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
