"""Represent vehicles / ECUs / CAN IDs as a graph for fleet-aware reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class FleetGraph:
    """Simple adjacency representation (upgrade to NetworkX / PyG later if needed)."""

    nodes: list[str] = field(default_factory=list)
    edges: list[tuple[str, str, float]] = field(default_factory=list)

    def add_edge(self, source: str, target: str, weight: float = 1.0) -> None:
        if source not in self.nodes:
            self.nodes.append(source)
        if target not in self.nodes:
            self.nodes.append(target)
        self.edges.append((source, target, weight))

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)


def build_fleet_graph(
    traces: list[pd.DataFrame],
    config: dict[str, Any],
) -> FleetGraph:
    """
    Construct a fleet graph from one or more vehicle traces.

    Edge semantics (co-occurrence, timing correlation, cross-vehicle similarity)
    will be defined when datasets are integrated.
    """
    graph_cfg = config.get("graph", {})
    min_weight = float(graph_cfg.get("min_edge_weight", 0.1))
    graph = FleetGraph()
    for idx, trace in enumerate(traces):
        vehicle_id = f"vehicle_{idx}"
        if "can_id" in trace.columns:
            for can_id in trace["can_id"].astype(str).unique():
                graph.add_edge(vehicle_id, f"id_{can_id}", weight=1.0)
    _ = min_weight  # reserved for filtering
    return graph
