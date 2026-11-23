"""Rebuild windows and features from raw labelled CAN traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.dataset_loader import load_and_merge, print_dataset_statistics, save_clean_dataset
from src.features.feature_extractor import extract_features, save_window_features
from src.features.window_generator import generate_windows, print_window_statistics, save_window_metadata
from src.utils.config import get_nested
from src.utils.logging import get_logger
from src.utils.paths import ProjectPaths

logger = get_logger(__name__)


def rebuild_windows_and_features(
    project_root: Path,
    out_root: Path,
    master_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CAN, window, extract features — all under experiment root."""
    paths = ProjectPaths.from_root(project_root)
    processed = out_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)

    clean_path = processed / "clean_can_data.csv"
    window_meta_path = out_root / "manifests" / "window_manifest.csv"
    features_path = processed / "window_features.csv"

    data_cfg = master_config.get("dataset", {})
    ext_dir = data_cfg.get("external_dataset_dir") or get_nested(
        master_config, "paths", "external_dataset_dir", default=None
    )
    if not ext_dir:
        ext_dir = str(paths.root / "data" / "raw")

    logger.info("Loading raw labelled CAN from %s", ext_dir)
    df = load_and_merge(
        external_root=Path(ext_dir),
        raw_root=paths.raw_dir,
    )
    print_dataset_statistics(df)
    save_clean_dataset(df, clean_path)

    feat_cfg = master_config.get("windowing", master_config.get("features", {}))
    window_size = int(feat_cfg.get("window_size", 100))
    overlap = int(feat_cfg.get("overlap", 50))
    stride = int(feat_cfg.get("stride", window_size - overlap))
    min_frames = int(feat_cfg.get("minimum_valid_frames", 10))

    logger.info("Generating windows (size=%d stride=%d)", window_size, stride)
    windows = generate_windows(
        df,
        config={"features": {"window_size": window_size, "overlap": overlap, "stride": stride}},
        window_size=window_size,
        overlap=overlap,
        stride=stride,
    )
    print_window_statistics(windows)
    save_window_metadata(windows, window_meta_path)

    logger.info("Extracting behavioural features")
    features = extract_features(df, windows)
    save_window_features(features, features_path)

    stats = {
        "n_traces": int(windows["source_file"].nunique()),
        "n_windows": int(len(windows)),
        "n_feature_rows": int(len(features)),
    }
    stats_path = out_root / "results" / "vehicle_level" / "window_statistics.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([stats]).to_csv(stats_path, index=False)

    return windows, features
