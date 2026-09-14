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

# descriptor_generator imports vehicle_ids, which imports this package — keep lazy.
def __getattr__(name: str):
    if name in {
        "generate_anomaly_descriptors",
        "print_descriptor_summary",
        "save_anomaly_descriptors",
    }:
        from src.features import descriptor_generator as _dg

        return getattr(_dg, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "generate_anomaly_descriptors",
    "print_descriptor_summary",
    "save_anomaly_descriptors",
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
