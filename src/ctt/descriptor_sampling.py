"""Balanced descriptor sampling for fleet candidate selection."""

from __future__ import annotations

import pandas as pd


def balanced_sample_descriptors(
    candidates: pd.DataFrame,
    max_descriptors: int,
    group_cols: tuple[str, ...] = ("vehicle_id", "subset_name", "attack_type"),
) -> pd.DataFrame:
    """Sample weak candidates evenly across vehicles, subsets, and attack types."""
    if candidates.empty or max_descriptors is None or len(candidates) <= max_descriptors:
        return candidates.copy()

    working = candidates.copy()
    working["_group"] = working[list(group_cols)].astype(str).agg("|".join, axis=1)
    groups = [g for _, g in working.groupby("_group", sort=False)]
    n_groups = len(groups)
    if n_groups == 0:
        return working.nlargest(max_descriptors, "anomaly_score").drop(columns=["_group"], errors="ignore")

    base = max_descriptors // n_groups
    remainder = max_descriptors % n_groups
    selected_parts: list[pd.DataFrame] = []

    for i, grp in enumerate(groups):
        quota = base + (1 if i < remainder else 0)
        if quota <= 0:
            continue
        take = grp.nlargest(min(quota, len(grp)), "anomaly_score")
        selected_parts.append(take)

    sampled = pd.concat(selected_parts, ignore_index=True)
    if len(sampled) < max_descriptors:
        chosen_wids = set(sampled["window_id"]) if "window_id" in sampled.columns else set()
        remaining = working[~working["window_id"].isin(chosen_wids)] if chosen_wids else working
        extra = remaining.nlargest(max_descriptors - len(sampled), "anomaly_score")
        sampled = pd.concat([sampled, extra], ignore_index=True)

    return sampled.drop(columns=["_group"], errors="ignore").head(max_descriptors)
