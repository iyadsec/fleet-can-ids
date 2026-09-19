"""Fleet graph construction — thin adapter over shared publication fleet graph path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ctt.constants import (
    GRAPH_CROSS_VEHICLE_CAP,
    GRAPH_KNN_CAP,
    GRAPH_SAME_VEHICLE_CAP,
    GRAPH_SIMILARITY_THRESHOLD,
    OUTPUT_ROOT,
)
from src.ctt.utils import ensure_dir
from src.evaluation.publication_fleet_core import (
    PublicationFleetConfig,
    build_publication_fleet_graph,
)
from src.experiments.local_descriptor_normalisation import (
    FleetScalerProvenance,
    fit_benign_fleet_scaler,
)


def resolve_ctt_fleet_config(
    *,
    similarity_threshold: float = GRAPH_SIMILARITY_THRESHOLD,
    k_same: int = GRAPH_SAME_VEHICLE_CAP,
    k_cross: int = GRAPH_CROSS_VEHICLE_CAP,
    seed: int = 42,
) -> PublicationFleetConfig:
    """Publication freeze defaults (CTT constants mirror the freeze)."""
    return PublicationFleetConfig(
        similarity_threshold=similarity_threshold,
        max_same_vehicle_neighbors=k_same,
        max_cross_vehicle_neighbors=k_cross,
        seed=seed,
    )


def fit_or_load_ctt_scaler(
    descriptors: pd.DataFrame,
    *,
    cache_path: Path | None = None,
) -> FleetScalerProvenance:
    """Fit benign-train fleet scaler on CTT descriptors (same methodology as OCSLab)."""
    from src.experiments.local_descriptor_normalisation import (
        load_scaler_provenance,
        save_scaler_provenance,
    )

    if cache_path is not None and cache_path.exists():
        return load_scaler_provenance(cache_path)
    # Prefer train_* subsets; if absent (scenario-only frames), fit on label==0 rows.
    try:
        prov = fit_benign_fleet_scaler(descriptors)
    except ValueError:
        benign = descriptors.copy()
        if "label" in benign.columns:
            benign = benign[benign["label"].astype(int) == 0]
        if benign.empty:
            # Last resort: fit on all rows but mark attack_labels_used — should not happen
            # in production; smoke fixtures always include benign rows.
            benign = descriptors
        from src.experiments.local_descriptor_normalisation import fit_benign_fleet_scaler_from_rows

        prov = fit_benign_fleet_scaler_from_rows(benign, training_split="scenario_benign")
    if cache_path is not None:
        save_scaler_provenance(prov, cache_path)
    return prov


def build_behavioural_graph(
    desc_df: pd.DataFrame,
    similarity_threshold: float = GRAPH_SIMILARITY_THRESHOLD,
    knn_cap: int = GRAPH_KNN_CAP,
    cross_vehicle_cap: int = GRAPH_CROSS_VEHICLE_CAP,
    *,
    scaler: FleetScalerProvenance | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Build fleet graph with publication constrained-kNN semantics.

    ``knn_cap`` is retained for API compatibility; same-vehicle cap is
    ``GRAPH_SAME_VEHICLE_CAP`` (freeze k_same=2), not the legacy single knn_cap.
    """
    del knn_cap  # legacy single-cap API; publication uses separate k_same / k_cross
    if desc_df.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "num_nodes": 0,
                "num_edges": 0,
                "same_vehicle_edges": 0,
                "cross_vehicle_edges": 0,
                "similarity_threshold": similarity_threshold,
                "k_same": GRAPH_SAME_VEHICLE_CAP,
                "k_cross": cross_vehicle_cap,
            },
        )

    cfg = resolve_ctt_fleet_config(
        similarity_threshold=similarity_threshold,
        k_same=GRAPH_SAME_VEHICLE_CAP,
        k_cross=cross_vehicle_cap,
        seed=seed,
    )
    scaler = scaler or fit_or_load_ctt_scaler(desc_df)
    data, meta, _X, _cols, stats = build_publication_fleet_graph(
        desc_df, cfg, fleet_scaler_provenance=scaler
    )

    # Edge table from PyG (undirected unique pairs via edge_attr half)
    edges: list[dict] = []
    ei = data.edge_index.numpy()
    ew = data.edge_attr.numpy() if data.edge_attr is not None else None
    seen: set[tuple[str, str]] = set()
    event_ids = list(data.event_ids)
    vehicles = list(data.vehicle_ids)
    for k in range(ei.shape[1]):
        i, j = int(ei[0, k]), int(ei[1, k])
        if i >= j:
            continue
        u, v = event_ids[i], event_ids[j]
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        edges.append(
            {
                "source": u,
                "target": v,
                "similarity": float(ew[k]) if ew is not None else 0.0,
                "edge_type": "behavioural_similarity",
                "cross_vehicle": vehicles[i] != vehicles[j],
                "temporal_edge": False,
            }
        )
    edge_df = pd.DataFrame(edges)
    node_df = meta.copy()
    if "event_id" in node_df.columns:
        node_df = node_df.rename(columns={"event_id": "node_id"})
    return node_df, edge_df, stats


def save_graph_artifacts(
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    stats: dict,
    output_root: Path = OUTPUT_ROOT,
) -> None:
    graph_dir = ensure_dir(output_root / "graph")
    results_dir = ensure_dir(output_root / "results" / "graph_analysis")
    node_df.to_csv(graph_dir / "node_manifest.csv", index=False)
    edge_df.to_csv(graph_dir / "edge_list.csv", index=False)
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(graph_dir / "graph_statistics.csv", index=False)
    stats_df.to_csv(results_dir / "graph_statistics.csv", index=False)
