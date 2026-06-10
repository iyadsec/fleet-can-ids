#!/usr/bin/env python3
"""Execute only missing scenario experiment combinations from publication manifest."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.experiment_runner import run_single_experiment
from src.experiments.result_writer import ExperimentRunContext, ProtectedOutputGuard, load_experiment_config
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.scenario_registry import get_scenario

MISSING_CSV = Path("new_experiments/publication_ready/data/missing_combinations.csv")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="new_experiments/configs/scenario_experiments.yaml")
    parser.add_argument("--missing", default=str(MISSING_CSV))
    parser.add_argument("--limit", type=int, default=0, help="Max runs (0=all)")
    args = parser.parse_args()

    missing_path = Path(args.missing)
    if not missing_path.exists():
        print(f"Missing file not found: {missing_path}", file=sys.stderr)
        return 1

    missing = pd.read_csv(missing_path)
    if args.limit > 0:
        missing = missing.head(args.limit)

    config = load_experiment_config(args.config)
    guard = ProtectedOutputGuard(_PROJECT_ROOT, config.get("general", {}).get("output_root", "new_experiments"))
    paths = config.get("paths", {})
    descriptors, features = load_descriptor_tables(
        _PROJECT_ROOT / paths.get("anomaly_descriptors", "data/processed/anomaly_descriptors.csv"),
        _PROJECT_ROOT / paths.get("window_features", "data/processed/window_features.csv"),
    )
    splits = config.get("splits", {})
    manifest = ensure_split_manifest(
        descriptors,
        features,
        output_path=Path("new_experiments/manifests/split_manifest.csv"),
        seed=11,
        train_ratio=float(splits.get("train", 0.7)),
        val_ratio=float(splits.get("validation", 0.15)),
        test_ratio=float(splits.get("test", 0.15)),
    )

    ok, fail = 0, 0
    for _, row in missing.iterrows():
        spec = get_scenario(row["scenario_key"])
        method_id = row["method"]
        try:
            ctx = ExperimentRunContext.create(
                guard=guard,
                scenario_key=row["scenario_key"],
                method=method_id,
                seed=int(row["seed"]),
                campaign_size=int(row["campaign_size"]),
                coordination_strength=float(row["coordination_strength"]),
                overwrite=False,
            )
            ctx.write_config_snapshot(config)
            run_single_experiment(ctx, spec, method_id, config, descriptors, manifest)
            ok += 1
            print(f"OK {row['scenario_key']} {method_id} seed={row['seed']} n={row['campaign_size']}")
        except FileExistsError:
            print(f"SKIP exists {row['scenario_key']} {method_id} seed={row['seed']}")
        except Exception as exc:
            fail += 1
            print(f"FAIL {row['scenario_key']} {method_id}: {exc}", file=sys.stderr)
            traceback.print_exc()

    print(f"Completed {ok}, failed {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
