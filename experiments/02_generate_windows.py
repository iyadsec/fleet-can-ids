#!/usr/bin/env python3
"""Generate sliding-window metadata from processed CAN frame data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.window_generator import (
    generate_windows,
    load_can_frames,
    print_window_statistics,
    resolve_window_params,
    save_window_metadata,
)
from src.utils import get_logger, load_config
from src.utils.paths import ProjectPaths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sliding windows over CAN traffic")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="YAML config for window_size / overlap / stride",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/clean_can_data.csv",
        help="Processed CAN frame CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/window_metadata.csv",
        help="Output window metadata CSV",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Frames per window (default: 100 from config)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help="Overlapping frames between consecutive windows",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Step between window starts (overridden if --overlap is set)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = get_logger("experiments.02_generate_windows")
    paths = ProjectPaths.from_root(_ROOT)

    config: dict = {}
    if args.config:
        try:
            config = load_config(paths.root / args.config)
        except FileNotFoundError:
            logger.warning("Config not found; using CLI defaults.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.is_absolute():
        input_path = paths.root / input_path
    if not output_path.is_absolute():
        output_path = paths.root / output_path

    size, overlap, stride = resolve_window_params(
        config,
        window_size=args.window_size,
        overlap=args.overlap,
        stride=args.stride,
    )
    logger.info("Input:  %s", input_path)
    logger.info("Output: %s", output_path)
    logger.info("window_size=%d, overlap=%d, stride=%d", size, overlap, stride)

    frames = load_can_frames(input_path)
    if frames.empty:
        logger.error("Input CAN data is empty.")
        return 1

    meta = generate_windows(
        frames,
        config,
        window_size=args.window_size,
        overlap=args.overlap,
        stride=args.stride,
    )
    if meta.empty:
        logger.error("No windows generated.")
        return 1

    save_window_metadata(meta, output_path)
    print_window_statistics(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
