"""Normalize fleet graph edge list column names."""

from __future__ import annotations

import pandas as pd


def edge_endpoints(edge_list: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if edge_list.empty:
        return pd.Series(dtype=str), pd.Series(dtype=str)
    if "source_event_id" in edge_list.columns:
        return edge_list["source_event_id"].astype(str), edge_list["target_event_id"].astype(str)
    return edge_list["source"].astype(str), edge_list["target"].astype(str)
