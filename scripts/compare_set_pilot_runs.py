#!/usr/bin/env python3
"""Compare two set_pilot run snapshots (e.g. v1 capped vs v2 rerun)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import OUTPUT_ROOT
from src.ctt.set_pilot import set_work_root


def _get(d: dict, *keys, default="n/a"):
    for k in keys:
        if k in d:
            return d[k]
    return default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set-id", default="set_01")
    parser.add_argument("--before", default="run_v1_snapshot.json")
    parser.add_argument("--after", default="run_v2_snapshot.json")
    args = parser.parse_args()

    val_dir = set_work_root(OUTPUT_ROOT, args.set_id) / "validation"
    before = json.loads((val_dir / args.before).read_text())
    after = json.loads((val_dir / args.after).read_text())

    lines = [
        f"# Set Pilot Comparison: {args.set_id}",
        "",
        f"**Before:** {before.get('label', 'v1')} | **After:** {after.get('label', 'v2')}",
        "",
        "| Metric | Before | After |",
        "|--------|--------|-------|",
        f"| Files processed | {_get(before, 'files_processed')} | {_get(after, 'files_processed')} |",
        f"| Rows processed | {_get(before, 'rows_processed')} | {_get(after, 'rows_processed')} |",
        f"| Windows | {_get(before, 'windows')} | {_get(after, 'windows')} |",
        f"| Descriptors | {_get(before, 'descriptors')} | {_get(after, 'descriptors')} |",
        f"| Subsets in windows | {', '.join(_get(before, 'subsets_in_windows', default=[]))} | {', '.join(_get(after, 'subsets_in_windows', default=[]))} |",
        f"| Vehicles in descriptors | {', '.join(_get(before, 'descriptor_vehicles', default=[]))} | {', '.join(_get(after, 'descriptor_vehicles', default=[]))} |",
        f"| Cross-vehicle edge % | {_get(before, 'cross_vehicle_edge_pct'):.4f} | {_get(after, 'cross_vehicle_edge_pct'):.4f} |",
        f"| Descriptor rate | {_get(before, 'descriptor_rate')} | {_get(after, 'descriptor_rate')} |",
        "",
        "## Scenario means",
        "",
    ]

    bsc = before.get("scenario_means", {})
    asc = after.get("scenario_means", {})
    for key in (
        "local_or_incident_detected",
        "fleet_campaign_detected",
        "false_campaign",
        "incorrect_merge_rate",
        "campaign_f1",
    ):
        lines.append(f"### {key}")
        lines.append("| Scenario | Before | After |")
        lines.append("|----------|--------|-------|")
        scenarios = sorted(set(list(bsc.get(key, {}).keys()) + list(asc.get(key, {}).keys())))
        for sc in scenarios:
            bv = bsc.get(key, {}).get(sc, "n/a")
            av = asc.get(key, {}).get(sc, "n/a")
            lines.append(f"| {sc} | {bv} | {av} |")
        lines.append("")

    out = val_dir / f"{args.set_id}_run_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
