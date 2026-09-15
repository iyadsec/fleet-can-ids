"""Feature extraction from preprocessed CAN traces."""

from src.features.extraction import extract_window_features
from src.features.feature_extractor import (
    BEHAVIOURAL_FEATURE_COLUMNS,
    extract_features,
    extract_window_features as extract_behavioural_window_features,
    plot_feature_correlation_heatmap,
    plot_feature_distributions,
    save_window_features,
)
from src.features.window_generator import (
    generate_windows,
    print_window_statistics,
    resolve_window_params,
    save_window_metadata,
)

__all__ = [
    "BEHAVIOURAL_FEATURE_COLUMNS",
    "extract_window_features",
    "extract_behavioural_window_features",
    "extract_features",
    "plot_feature_correlation_heatmap",
    "plot_feature_distributions",
    "save_window_features",
    "generate_windows",
    "print_window_statistics",
    "resolve_window_params",
    "save_window_metadata",
]
