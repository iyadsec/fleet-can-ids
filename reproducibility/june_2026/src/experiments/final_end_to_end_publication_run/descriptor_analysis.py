"""Descriptor compactness and privacy analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.features.descriptor_generator import DESCRIPTOR_COLUMNS


def run_descriptor_analysis(
    descriptors: pd.DataFrame,
    master_config: dict,
    out_dir: Path,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    windowing = master_config.get("windowing", {})
    window_size = int(windowing.get("window_size", 100))
    raw_bytes = window_size * 16  # CAN frame proxy

    desc_cols = [c for c in DESCRIPTOR_COLUMNS if c in descriptors.columns]
    sample = descriptors[desc_cols].head(1000)
    desc_bytes = int(sample.memory_usage(deep=True).sum() / max(len(sample), 1))

    compression = raw_bytes / max(desc_bytes, 1)
    bandwidth_reduction = 1.0 - (desc_bytes / raw_bytes)

    security = pd.DataFrame(
        [
            {
                "metric": "raw_window_bytes",
                "value": raw_bytes,
            },
            {
                "metric": "descriptor_bytes",
                "value": desc_bytes,
            },
            {
                "metric": "compression_ratio",
                "value": compression,
            },
            {
                "metric": "bandwidth_reduction",
                "value": bandwidth_reduction,
            },
            {
                "metric": "payload_bytes_transmitted",
                "value": 0,
            },
            {
                "metric": "can_id_transmitted",
                "value": 0,
            },
            {
                "metric": "fleet_candidate_rate",
                "value": float(descriptors.get("fleet_candidate", pd.Series([0])).mean()),
            },
        ]
    )
    security.to_csv(out_dir / "descriptor_security_metrics.csv", index=False)

    fleet_sizes = master_config.get("descriptor_analysis", {}).get("fleet_sizes", [10, 50, 100, 500, 1000])
    scale_rows = []
    for n in fleet_sizes:
        scale_rows.append(
            {
                "fleet_size": n,
                "raw_can_mb_per_window_batch": raw_bytes * n * 10 / 1e6,
                "descriptor_mb_per_window_batch": desc_bytes * n * 10 / 1e6,
            }
        )
    pd.DataFrame(scale_rows).to_csv(out_dir / "descriptor_scalability.csv", index=False)

    comparison = pd.DataFrame(
        [
            {"channel": "raw_can", "bytes_per_window": raw_bytes, "privacy_sensitive_fields": "payload,can_id,timestamp"},
            {"channel": "descriptor", "bytes_per_window": desc_bytes, "privacy_sensitive_fields": "none_transmitted"},
        ]
    )
    comparison.to_csv(out_dir / "descriptor_comparison.csv", index=False)
    return security
