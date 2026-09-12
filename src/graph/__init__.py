"""
Graph Construction Module (Publication Path)

Authoritative fleet graph builder for FLEET-GUARD publication results:
cross-vehicle constrained k-NN with separate same/cross caps.

Legacy radius/threshold graphs with a single ``max_neighbors`` cap have been
removed. Use :func:`build_scenario_graph` or
:func:`build_cross_vehicle_constrained_knn_edges`.
"""

from .fleet_graph_builder import (
    build_cross_vehicle_constrained_knn_edges,
    print_graph_statistics,
    save_fleet_graph,
)
from .scenario_graph import (
    build_scenario_graph,
    build_scenario_graph_from_features,
    resolve_publication_graph_config,
)

__all__ = [
    "build_cross_vehicle_constrained_knn_edges",
    "build_scenario_graph",
    "build_scenario_graph_from_features",
    "print_graph_statistics",
    "resolve_publication_graph_config",
    "save_fleet_graph",
]
