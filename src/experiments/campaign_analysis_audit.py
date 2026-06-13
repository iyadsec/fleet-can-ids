"""Data availability audit for campaign analysis experiments."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.experiments.data_splits import is_benign_attack_type
from src.experiments.scenario_generator import ensure_split_manifest, load_descriptor_tables
from src.experiments.vehicle_instance_builder import (
    VEHICLE_MODELS,
    build_instance_catalog,
    source_trace_name,
)


def run_data_availability_audit(
    config: dict[str, Any],
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit held-out test data and write audit artifacts."""
    paths = config.get("paths", {})
    splits = config.get("splits", {})
    local_cfg = config.get("local_ids", {})
    camp_cfg = config.get("campaign", {})

    descriptors, features = load_descriptor_tables(
        Path(paths["anomaly_descriptors"]),
        Path(paths["window_features"]),
    )
    manifest_path = output_root / "manifests" / "split_manifest.csv"
    manifest = ensure_split_manifest(
        descriptors,
        features,
        output_path=manifest_path,
        seed=int(splits.get("seed", 42)),
        train_ratio=float(splits.get("train", 0.70)),
        val_ratio=float(splits.get("validation", 0.15)),
        test_ratio=float(splits.get("test", 0.15)),
    )

    min_seg = int(camp_cfg.get("min_windows_per_segment", 15))
    catalog = build_instance_catalog(
        descriptors,
        manifest,
        weak_threshold=float(local_cfg.get("weak_threshold", 0.55)),
        strong_threshold=float(local_cfg.get("strong_threshold", 0.80)),
        min_windows_per_segment=min_seg,
    )

    catalog_path = output_root / "manifests" / "available_vehicle_instances.csv"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    export_cat = catalog.copy()
    if "window_ids" in export_cat.columns:
        export_cat["window_ids"] = export_cat["window_ids"].apply(
            lambda w: "|".join(str(int(x)) for x in w) if isinstance(w, list) else w
        )
    export_cat.to_csv(catalog_path, index=False)

    test = manifest[manifest["split"] == "test"]
    join_cols = ["window_id", "vehicle_model", "source_file"]
    desc_test = descriptors.merge(
        test[join_cols + ["split"]].drop_duplicates(),
        on=join_cols,
        how="inner",
    )
    weak_th = float(local_cfg.get("weak_threshold", 0.55))
    strong_th = float(local_cfg.get("strong_threshold", 0.80))
    mal = desc_test[~desc_test["attack_type"].map(is_benign_attack_type)]

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_windows": len(desc_test),
        "test_traces": desc_test["source_file"].nunique(),
        "catalog_instances": len(catalog),
        "strong_instances": int(catalog[catalog["strong_attack_events"] >= 3].shape[0]),
        "weak_instances": int(catalog[catalog["weak_attack_events"] >= 3].shape[0]),
        "benign_instances": int(catalog[catalog["benign_events"] >= 5].shape[0]),
        "per_model": {},
        "campaign_sizes_supported": {},
        "max_defensible_campaign_size": 0,
        "excluded_configurations": [],
    }

    for vm in VEHICLE_MODELS:
        vm_cat = catalog[catalog["vehicle_model"] == vm]
        vm_test = desc_test[desc_test["vehicle_model"] == vm]
        strong_mal = mal[(mal["vehicle_model"] == vm) & (mal["anomaly_score"] >= strong_th)]
        weak_mal = mal[
            (mal["vehicle_model"] == vm)
            & (mal["anomaly_score"] >= weak_th)
            & (mal["anomaly_score"] < strong_th)
        ]
        benign = desc_test[
            (desc_test["vehicle_model"] == vm) & desc_test["attack_type"].map(is_benign_attack_type)
        ]
        summary["per_model"][vm] = {
            "traces": int(vm_test["source_file"].nunique()),
            "segments": len(vm_cat),
            "strong_malicious_events": len(strong_mal),
            "weak_malicious_events": len(weak_mal),
            "benign_events": len(benign),
            "strong_by_attack": strong_mal.groupby("attack_type").size().to_dict(),
            "weak_by_attack": weak_mal.groupby("attack_type").size().to_dict(),
        }

    preferred_fleet = int(config.get("experiment_a", {}).get("preferred_total_fleet_size", 20))
    for cs in config.get("experiment_a", {}).get("campaign_sizes", [2, 5, 10]):
        n_benign = preferred_fleet - cs
        strong_pool = catalog[catalog["strong_attack_events"] >= 3]
        weak_pool = catalog[catalog["weak_attack_events"] >= 3]
        ben_pool = catalog[catalog["benign_events"] >= 5]
        supported = (
            len(strong_pool) >= cs
            and len(weak_pool) >= cs
            and len(ben_pool) >= n_benign
        )
        summary["campaign_sizes_supported"][cs] = supported
        if supported:
            summary["max_defensible_campaign_size"] = max(summary["max_defensible_campaign_size"], cs)

    if not summary["campaign_sizes_supported"].get(10, False):
        summary["excluded_configurations"].append(
            "Campaign size 10 may be limited if insufficient disjoint strong/weak attacked "
            "instances exist per seed; catalog shows segment-level capacity."
        )
    if summary["per_model"].get("Chevrolet", {}).get("weak_malicious_events", 0) == 0:
        summary["excluded_configurations"].append(
            "Chevrolet excluded from weak-attack experiments (no weak malicious events in test split)."
        )

    md = _render_audit_markdown(summary, catalog, desc_test, weak_th, strong_th)
    audit_path = output_root / "validation" / "data_availability_audit.md"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(md, encoding="utf-8")
    return catalog, summary


def _render_audit_markdown(
    summary: dict[str, Any],
    catalog: pd.DataFrame,
    desc_test: pd.DataFrame,
    weak_th: float,
    strong_th: float,
) -> str:
    lines = [
        "# Data Availability Audit — Campaign Analysis",
        "",
        f"**Generated:** {summary['generated_at']}",
        "",
        "## Test split overview",
        "",
        f"- Test windows: **{summary['test_windows']}**",
        f"- Independent source traces: **{summary['test_traces']}**",
        f"- Catalogued disjoint segments (min windows per segment): **{summary['catalog_instances']}**",
        f"- Strong-attack eligible instances: **{summary['strong_instances']}**",
        f"- Weak-attack eligible instances: **{summary['weak_instances']}**",
        f"- Benign eligible instances: **{summary['benign_instances']}**",
        "",
        "## Per vehicle model",
        "",
    ]
    for vm, info in summary["per_model"].items():
        lines.extend(
            [
                f"### {vm}",
                "",
                f"- Traces: {info['traces']}",
                f"- Disjoint segments: {info['segments']}",
                f"- Strong malicious events: {info['strong_malicious_events']}",
                f"- Weak malicious events: {info['weak_malicious_events']}",
                f"- Benign events: {info['benign_events']}",
                f"- Strong by attack type: `{info['strong_by_attack']}`",
                f"- Weak by attack type: `{info['weak_by_attack']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Campaign size support (fleet size 20)",
            "",
            "| Campaign size | Supported |",
            "| --- | --- |",
        ]
    )
    for cs, ok in summary["campaign_sizes_supported"].items():
        lines.append(f"| {cs} | {'yes' if ok else 'no'} |")
    lines.extend(
        [
            "",
            f"**Maximum defensible campaign size (catalog-level):** {summary['max_defensible_campaign_size']}",
            "",
            "## Construction method",
            "",
            "Each `scenario_vehicle_id` maps to one disjoint non-overlapping segment of a "
            "held-out test trace. Segments from the same trace must not overlap in `window_id` "
            "range. Descriptor values are preserved; rows are never duplicated or relabelled.",
            "",
            "## Excluded / limited configurations",
            "",
        ]
    )
    if summary["excluded_configurations"]:
        for item in summary["excluded_configurations"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    lines.extend(["", "## Trace inventory", ""])
    for sf, g in desc_test.groupby("source_file"):
        name = source_trace_name(str(sf))
        lines.append(
            f"- `{name}` ({g['vehicle_model'].iloc[0]}): {len(g)} windows, "
            f"window_id [{g['window_id'].min()}, {g['window_id'].max()}]"
        )
    return "\n".join(lines) + "\n"
