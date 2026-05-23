#!/usr/bin/env python3
"""Extract behavioural features per sliding window and generate EDA plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.feature_extractor import (
    extract_features,
    load_frames,
    load_windows,
    plot_feature_correlation_heatmap,
    plot_feature_distributions,
    print_feature_summary,
    save_window_features,
)
from src.utils import get_logger
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract behavioural CAN window features")
    parser.add_argument(
        "--frames",
        type=str,
        default="data/processed/clean_can_data.csv",
        help="Frame-level CAN CSV",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default="data/processed/window_metadata.csv",
        help="Window metadata CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/window_features.csv",
        help="Output feature CSV",
    )
    parser.add_argument(
        "--corr-plot",
        type=str,
        default="outputs/figures/feature_correlation_heatmap.png",
        help="Correlation heatmap output path",
    )
    parser.add_argument(
        "--dist-plot",
        type=str,
        default="outputs/figures/feature_distributions.png",
        help="Distribution plot output path",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating figures",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.03_extract_features")
    paths = ProjectPaths.from_root(_ROOT)

    frames_path = paths.root / args.frames
    windows_path = paths.root / args.windows
    output_path = paths.root / args.output
    corr_path = paths.root / args.corr_plot
    dist_path = paths.root / args.dist_plot

    logger.info("Frames:  %s", frames_path)
    logger.info("Windows: %s", windows_path)
    logger.info("Output:  %s", output_path)

    if not frames_path.exists():
        logger.error("Frames file not found: %s", frames_path)
        return 1
    if not windows_path.exists():
        logger.error("Windows file not found: %s (run 02_generate_windows.py)", windows_path)
        return 1

    frames = load_frames(frames_path)
    windows = load_windows(windows_path)
    logger.info("Loaded %d frames, %d windows", len(frames), len(windows))

    features = extract_features(frames, windows)
    if features.empty:
        logger.error("No features extracted.")
        return 1

    save_window_features(features, output_path)
    print_feature_summary(features)

    if not args.no_plots:
        plot_feature_correlation_heatmap(features, corr_path)
        plot_feature_distributions(features, dist_path)
        logger.info("Figures saved under %s", paths.figures_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
