#!/usr/bin/env python3
"""Regenerate FPR-controlled vehicle-level Isolation Forest metrics for Table I."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.vehicle_level_evaluation import EvaluationOutputs, run_vehicle_level_evaluation
from src.models.vehicle_ids import load_feature_dataset
from src.utils import get_logger, load_config
from src.utils.config import get_nested
from src.utils.paths import ProjectPaths

logger = get_logger("scripts.run_vehicle_level_fpr_controlled")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FPR-controlled OCSLab vehicle-level Isolation Forest evaluation (Table I)"
    )
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--features", default=None, help="Path to window_features.csv")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--tables-dir", default="tables")
    p.add_argument("--figures-dir", default="figures")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    paths = ProjectPaths.from_root(_ROOT)
    config = load_config(paths.root / args.config)

    features_path = Path(args.features) if args.features else paths.processed_dir / "window_features.csv"
    if not features_path.exists():
        logger.error(
            "Missing %s.\n"
            "Run the OCSLab preprocessing pipeline first:\n"
            "  python experiments/01_load_dataset.py --config %s\n"
            "  python experiments/02_generate_windows.py --config %s\n"
            "  python experiments/03_extract_features.py\n"
            "Set OCSLAB_DATASET_DIR if your dataset is not under Dataset/ocslab/",
            features_path,
            args.config,
            args.config,
        )
        return 1

    data_cfg = config.get("data", {})
    ids_cfg = config.get("vehicle_ids", {})
    eval_cfg = config.get("vehicle_level_evaluation", {})

    outputs = EvaluationOutputs(
        results_dir=paths.root / args.results_dir,
        tables_dir=paths.root / args.tables_dir,
        figures_dir=paths.root / args.figures_dir,
    )

    features = load_feature_dataset(features_path)
    written = run_vehicle_level_evaluation(
        features,
        train_ratio=float(eval_cfg.get("train_split", data_cfg.get("train_split", 0.7))),
        val_ratio=float(eval_cfg.get("val_split", data_cfg.get("val_split", 0.15))),
        test_ratio=float(eval_cfg.get("test_split", data_cfg.get("test_split", 0.15))),
        random_state=int(get_nested(config, "project", "seed", default=42)),
        n_estimators=int(eval_cfg.get("n_estimators", ids_cfg.get("n_estimators", 200))),
        outputs=outputs,
    )

    tex_path = written["table_vehicle_level_ids_fpr_controlled_tex"]
    print("\n=== Corrected LaTeX table (Table I) ===\n")
    print(tex_path.read_text(encoding="utf-8"))

    logger.info("Wrote:")
    for key, path in written.items():
        if "fpr_controlled" in key or "validation_report" in key:
            logger.info("  %s: %s", key, path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
