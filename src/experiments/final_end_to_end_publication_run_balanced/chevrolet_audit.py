"""Chevrolet source availability audit."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.data_splits import is_benign_attack_type


def audit_chevrolet_sources(
    window_manifest: pd.DataFrame,
    clean_can_path: Path | None,
    original_split_manifest: pd.DataFrame | None,
) -> str:
    chev = window_manifest[window_manifest["vehicle_model"] == "Chevrolet"].copy()
    row_counts: dict[str, int] = {}
    if clean_can_path and clean_can_path.exists():
        clean = pd.read_csv(clean_can_path, usecols=["source_file", "vehicle_model"])
        chev_clean = clean[clean["vehicle_model"] == "Chevrolet"]
        row_counts = chev_clean.groupby("source_file").size().to_dict()

    orig_map: dict[tuple[str, str], str] = {}
    if original_split_manifest is not None and not original_split_manifest.empty:
        for _, r in original_split_manifest[original_split_manifest.vehicle_model == "Chevrolet"].iterrows():
            orig_map[(str(r["source_file"]), str(r.get("attack_type", "")))] = str(r["split"])

    lines = [
        "# Chevrolet source availability audit",
        "",
        "| Source file | Attack type | Benign | Row count | Window count | Original split | Segmentable |",
        "|-------------|-------------|--------|-----------|--------------|----------------|-------------|",
    ]

    for src, grp in chev.groupby("source_file"):
        atk = str(grp["attack_type"].iloc[0])
        ben = is_benign_attack_type(atk)
        rows = row_counts.get(src, row_counts.get(str(src), "n/a"))
        wins = len(grp)
        orig = orig_map.get((src, atk), orig_map.get((src, ""), "n/a"))
        segmentable = "yes" if wins >= 9 else "no"
        name = Path(str(src)).name
        lines.append(
            f"| {name} | {atk} | {'yes' if ben else 'no'} | {rows} | {wins} | {orig} | {segmentable} |"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"- Benign source files: {chev[chev.attack_type.map(is_benign_attack_type)].source_file.nunique()}",
        f"- Malicious source files: {chev[~chev.attack_type.map(is_benign_attack_type)].source_file.nunique()}",
        "",
        "## Recommendation",
        "",
        "Chevrolet has a single benign trace (`Attack_free_CHEVROLET_Spark_train.csv`). "
        "Publication-safe validation coverage requires **contiguous segment-level splitting** "
        f"with **{100}-frame guard gaps** between train, validation, and test partitions.",
    ]
    return "\n".join(lines)
