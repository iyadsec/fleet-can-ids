"""Cross-model graph connectivity audit for heterogeneous campaigns."""

from __future__ import annotations

import networkx as nx
import pandas as pd

from src.experiments.model_diversity_final.edge_utils import edge_endpoints


def audit_cross_model_connectivity(
    scenario_df: pd.DataFrame,
    edge_list: pd.DataFrame,
    event_predictions: pd.DataFrame,
    *,
    run_id: str,
) -> dict:
    id_to_model = scenario_df.set_index("event_id")["vehicle_model"].astype(str).to_dict()
    same_model = cross_model = cross_mal = cross_ben = 0
    if not edge_list.empty:
        src, tgt = edge_endpoints(edge_list)
        for s, t in zip(src, tgt):
            m1, m2 = id_to_model.get(s), id_to_model.get(t)
            if not m1 or not m2:
                continue
            if m1 == m2:
                same_model += 1
            else:
                cross_model += 1
                gt = scenario_df.set_index("event_id")
                if s in gt.index and t in gt.index:
                    if gt.loc[s, "ground_truth_malicious"] == 1 and gt.loc[t, "ground_truth_malicious"] == 1:
                        cross_mal += 1
                    if gt.loc[s, "ground_truth_malicious"] == 0 and gt.loc[t, "ground_truth_malicious"] == 0:
                        cross_ben += 1
    total = same_model + cross_model
    G = nx.Graph()
    if not edge_list.empty:
        src, tgt = edge_endpoints(edge_list)
        for s, t in zip(src, tgt):
            G.add_edge(s, t)
    components = list(nx.connected_components(G)) if G.number_of_nodes() else []
    coord = event_predictions[event_predictions.get("final_decision", "") == "coordinated_attack"] if "final_decision" in event_predictions.columns else event_predictions.iloc[0:0]
    platforms_in_campaign = set()
    if not coord.empty and "vehicle_model" in coord.columns:
        platforms_in_campaign = set(coord["vehicle_model"].astype(str).unique())
    connected_platforms = 0
    if platforms_in_campaign and components:
        for comp in components:
            models = {id_to_model.get(n) for n in comp if id_to_model.get(n)}
            if platforms_in_campaign.issubset(models):
                connected_platforms = len(platforms_in_campaign)
                break
    return {
        "run_id": run_id,
        "same_model_edges": same_model,
        "cross_model_edges": cross_model,
        "cross_model_edge_percentage": 100.0 * cross_model / total if total else 0.0,
        "cross_model_malicious_edges": cross_mal,
        "cross_model_benign_edges": cross_ben,
        "malicious_cross_model_edge_purity": cross_mal / cross_model if cross_model else float("nan"),
        "benign_cross_model_false_edge_rate": cross_ben / cross_model if cross_model else float("nan"),
        "connected_components": len(components),
        "platforms_in_predicted_campaign": len(platforms_in_campaign),
        "connected_platform_count": connected_platforms,
        "graph_connected_across_platforms": connected_platforms >= len(platforms_in_campaign) if len(platforms_in_campaign) > 1 else True,
    }
