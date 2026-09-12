#!/usr/bin/env python3
"""Verify restored publication graph builder against algorithm contracts and archive stats."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.graph.fleet_graph_builder import build_cross_vehicle_constrained_knn_edges
from src.graph.scenario_graph import (
    build_scenario_graph,
    build_scenario_graph_from_features,
    resolve_publication_graph_config,
)

ARCHIVE_METRICS = (
    ROOT
    / "experimental-2026-06-23"
    / "01_primary_ocslab_balanced"
    / "results"
    / "campaign_metrics.csv"
)
ARCHIVE_P10 = (
    ROOT
    / "experimental-2026-06-23"
    / "01_primary_ocslab_balanced"
    / "tables"
    / "table_P10_edge_connectivity_performance.csv"
)


def _archive_strong_cs5_stats() -> dict[str, float]:
    rows = list(csv.DictReader(ARCHIVE_METRICS.open()))
    sub = [r for r in rows if r.get("attack_strength") == "strong" and r.get("campaign_size") == "5"]
    keys = [
        "graph_nodes",
        "unique_edges",
        "same_vehicle_edges",
        "cross_vehicle_edges",
        "cross_vehicle_edge_percentage",
        "average_degree",
    ]
    out: dict[str, float] = {"n_runs": float(len(sub))}
    for k in keys:
        vals = [float(r[k]) for r in sub if r.get(k) not in (None, "")]
        out[f"mean_{k}"] = float(statistics.mean(vals)) if vals else float("nan")
        out[f"min_{k}"] = float(min(vals)) if vals else float("nan")
        out[f"max_{k}"] = float(max(vals)) if vals else float("nan")
    return out


def _synthetic_campaign_descriptors(
    *,
    n_vehicles: int = 20,
    windows_per_vehicle: int = 10,
    campaign_vehicles: int = 5,
    dim: int = 24,
    seed: int = 11,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Synthetic 200-node fleet matching publication scenario shape (20x10).

    Campaign vehicles share a common latent direction so cross-vehicle edges
    survive the 0.95 cosine threshold; benign vehicles are near-orthogonal.
    """
    rng = np.random.default_rng(seed)
    assert dim == len(BEHAVIOURAL_FEATURE_COLUMNS)
    campaign_dir = rng.normal(size=dim)
    campaign_dir /= np.linalg.norm(campaign_dir) + 1e-12

    rows: list[dict] = []
    X: list[np.ndarray] = []
    vehicles: list[str] = []
    for v in range(n_vehicles):
        vid = f"V{v:02d}"
        if v < campaign_vehicles:
            base = campaign_dir + 0.02 * rng.normal(size=dim)
        else:
            base = rng.normal(size=dim)
            base /= np.linalg.norm(base) + 1e-12
        for w in range(windows_per_vehicle):
            vec = base + 0.01 * rng.normal(size=dim)
            X.append(vec.astype(np.float32))
            vehicles.append(vid)
            row = {
                "event_id": f"{vid}_w{w}",
                "window_id": w,
                "vehicle_model": vid,
                "source_file": f"{vid}.csv",
                "attack_type": "campaign" if v < campaign_vehicles else "benign",
                "anomaly_score": 0.9 if v < campaign_vehicles else 0.2,
                "evidence_level": "strong" if v < campaign_vehicles else "weak",
                "local_alert": 1 if v < campaign_vehicles else 0,
                "weak_signal": 0 if v < campaign_vehicles else 1,
            }
            for i, col in enumerate(BEHAVIOURAL_FEATURE_COLUMNS):
                row[col] = float(vec[i])
            rows.append(row)
    return np.asarray(X, dtype=np.float32), np.asarray(vehicles), pd.DataFrame(rows)


def _directed_neighbor_caps(
    X: np.ndarray,
    vehicles: np.ndarray,
    *,
    k_same: int,
    k_cross: int,
    threshold: float,
    seed: int,
) -> dict[str, object]:
    """Re-run builder and measure per-node same/cross neighbor counts on kept edges."""
    _b_ei, _b_w, ei, w, idx = build_cross_vehicle_constrained_knn_edges(
        X,
        vehicles,
        top_k_same_vehicle=k_same,
        top_k_cross_vehicle=k_cross,
        similarity_threshold=threshold,
        metric="cosine",
        seed=seed,
    )
    veh = vehicles[idx]
    same_deg: dict[int, int] = defaultdict(int)
    cross_deg: dict[int, int] = defaultdict(int)
    min_sim = float("inf")
    # ei is undirected with both directions duplicated for PyG in some builders;
    # here undirected pairs are stored once then expanded? Check shape usage.
    # build_cross_vehicle_constrained_knn_edges returns undirected edge_index
    # as pairs (not necessarily bidirectional). Treat unique undirected edges.
    seen: set[tuple[int, int]] = set()
    for k in range(ei.shape[1]):
        u, v = int(ei[0, k]), int(ei[1, k])
        a, b = (u, v) if u < v else (v, u)
        if a == b or (a, b) in seen:
            continue
        seen.add((a, b))
        sim = float(w[k])
        min_sim = min(min_sim, sim)
        if veh[u] == veh[v]:
            same_deg[u] += 1
            same_deg[v] += 1
        else:
            cross_deg[u] += 1
            cross_deg[v] += 1
    max_same = max(same_deg.values()) if same_deg else 0
    max_cross = max(cross_deg.values()) if cross_deg else 0
    n_same = sum(1 for a, b in seen if veh[a] == veh[b])
    n_cross = len(seen) - n_same
    return {
        "n_nodes": int(len(idx)),
        "n_undirected_edges": len(seen),
        "n_same_vehicle_edges": n_same,
        "n_cross_vehicle_edges": n_cross,
        "max_same_degree": max_same,
        "max_cross_degree": max_cross,
        "min_edge_similarity": min_sim if seen else float("nan"),
        "all_edges_above_threshold": (min_sim >= threshold - 1e-9) if seen else True,
        # Outbound selection is capped at k; undirected degree may exceed k if
        # many peers select a hub. Report both facts.
        "outbound_cap_same": k_same,
        "outbound_cap_cross": k_cross,
        "undirected_same_degree_le_cap": max_same <= k_same,
        "undirected_cross_degree_le_cap": max_cross <= k_cross,
    }


def main() -> int:
    print("=== Publication config resolution ===")
    cfg = resolve_publication_graph_config(
        {
            "graph": {
                "similarity_threshold": 0.95,
                "max_same_vehicle_neighbors": 2,
                "max_cross_vehicle_neighbors": 5,
            }
        }
    )
    print(
        f"  threshold={cfg.similarity_threshold} "
        f"same={cfg.max_same_vehicle_neighbors} "
        f"cross={cfg.max_cross_vehicle_neighbors}"
    )
    assert cfg.similarity_threshold == 0.95
    assert cfg.max_same_vehicle_neighbors == 2
    assert cfg.max_cross_vehicle_neighbors == 5

    sens = resolve_publication_graph_config(
        {"graph": {"max_cross_vehicle_neighbors": 3}},
    )
    print(f"  sensitivity cross override -> {sens.max_cross_vehicle_neighbors}")
    assert sens.max_cross_vehicle_neighbors == 3
    assert sens.similarity_threshold == 0.95
    assert sens.max_same_vehicle_neighbors == 2

    print("\n=== Synthetic campaign graph (20 vehicles x 10 windows = 200 nodes) ===")
    X, vehicles, desc = _synthetic_campaign_descriptors(seed=11)
    assert X.shape == (200, 24)
    caps = _directed_neighbor_caps(
        X, vehicles, k_same=2, k_cross=5, threshold=0.95, seed=11
    )
    for k, v in caps.items():
        print(f"  {k}: {v}")

    result = build_scenario_graph_from_features(
        X,
        vehicles,
        desc,
        similarity_threshold=0.95,
        max_same_vehicle_neighbors=2,
        max_cross_vehicle_neighbors=5,
        seed=11,
    )
    stats = result.stats.iloc[0].to_dict()
    print("\n=== build_scenario_graph_from_features stats ===")
    for k in [
        "nodes",
        "unique_undirected_edges",
        "same_vehicle_edges",
        "cross_vehicle_edges",
        "cross_vehicle_edge_percentage",
        "average_degree",
        "similarity_threshold",
        "max_same_vehicle_neighbors",
        "max_cross_vehicle_neighbors",
        "similarity_metric",
    ]:
        print(f"  {k}: {stats[k]}")

    assert stats["nodes"] == 200
    assert stats["similarity_metric"] == "cosine"
    assert stats["similarity_threshold"] == 0.95
    assert stats["max_same_vehicle_neighbors"] == 2
    assert stats["max_cross_vehicle_neighbors"] == 5
    assert stats["same_vehicle_edges"] > 0
    assert stats["cross_vehicle_edges"] > 0
    assert caps["all_edges_above_threshold"] is True

    # Descriptor path (24-D full_descriptor view)
    result2 = build_scenario_graph(
        desc,
        config={"graph": {"similarity_feature_view": "full_descriptor"}},
        similarity_threshold=0.95,
        max_same_vehicle_neighbors=2,
        max_cross_vehicle_neighbors=5,
        seed=11,
    )
    print("\n=== build_scenario_graph(descriptors) nodes/edges ===")
    print(result2.stats.iloc[0][["nodes", "unique_undirected_edges", "same_vehicle_edges", "cross_vehicle_edges"]].to_dict())
    assert int(result2.stats.iloc[0]["nodes"]) == 200
    assert int(result2.stats.iloc[0]["cross_vehicle_edges"]) > 0

    print("\n=== Archived publication graph statistics (strong, campaign_size=5) ===")
    arch = _archive_strong_cs5_stats()
    for k, v in arch.items():
        print(f"  {k}: {v}")

    print("\n=== Reconciliation notes ===")
    print(
        "  Exact node/edge reproduction requires the original scenario descriptor "
        "tables (not archived on main). Structural checks against the archive:"
    )
    print(f"  - archive mean graph_nodes={arch['mean_graph_nodes']:.1f} (expected 200); synthetic nodes={stats['nodes']}")
    print(
        f"  - archive mean unique_edges={arch['mean_unique_edges']:.1f} "
        f"range=[{arch['min_unique_edges']:.1f}, {arch['max_unique_edges']:.1f}]; "
        f"synthetic={stats['unique_undirected_edges']}"
    )
    print(
        f"  - archive mean same/cross="
        f"{arch['mean_same_vehicle_edges']:.1f}/{arch['mean_cross_vehicle_edges']:.1f}; "
        f"synthetic={stats['same_vehicle_edges']}/{stats['cross_vehicle_edges']}"
    )
    print(
        f"  - archive P10 configs include similarity_threshold=0.95 with "
        f"max_same_vehicle_neighbors=2 and max_cross_vehicle_neighbors in {{3,5,10,20}}"
    )
    assert ARCHIVE_P10.exists()
    p10 = pd.read_csv(ARCHIVE_P10)
    assert set(p10["max_same_vehicle_neighbors"].unique()) == {2.0}
    assert 5.0 in set(p10["max_cross_vehicle_neighbors"].unique())
    assert 3.0 in set(p10["max_cross_vehicle_neighbors"].unique())
    assert 0.95 in set(p10["similarity_threshold"].unique())
    print("  - P10 archive confirms publication + sensitivity neighbor/threshold settings: PASS")

    print("\n=== Cap semantics ===")
    print(
        "  Constrained-kNN applies outbound top-k same/cross selection, then undirected merge. "
        f"Undirected same-degree <= k_same? {caps['undirected_same_degree_le_cap']} "
        f"(max_same_degree={caps['max_same_degree']}, k_same={caps['outbound_cap_same']}). "
        f"Undirected cross-degree <= k_cross? {caps['undirected_cross_degree_le_cap']} "
        f"(max_cross_degree={caps['max_cross_degree']}, k_cross={caps['outbound_cap_cross']})."
    )
    print(
        "  Note: hub nodes can exceed the outbound cap in undirected degree; this matches the "
        "authoritative builder semantics (selection cap, not hard undirected degree cap)."
    )

    print("\nVERIFICATION SUMMARY: PASS (algorithm contracts + archive config reconciliation)")
    print("Exact edge-count bit-match: NOT POSSIBLE without original descriptor artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
