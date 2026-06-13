"""Write audit markdown reports for final Phase 4."""

from __future__ import annotations

from pathlib import Path


def write_similarity_audit(path: Path) -> None:
    text = """# Similarity metric audit

## Root cause of impossible values (~170) in diagnostic corrected Phase 4

`compute_descriptor_similarity_metrics()` in `campaign_analysis_runner.py` computed **raw dot products** on z-scored (not L2-normalized) feature vectors and reported them as similarity. Dot products of 7–8 dimensional z-scored vectors routinely exceed 1.0.

## Graph construction (unchanged)

`fleet_graph_builder.build_cross_vehicle_constrained_knn_edges()` uses sklearn `NearestNeighbors(metric="cosine")` with `similarity = 1 - cosine_distance`, clipped to [0, 1].

## Final Phase 4 reporting

`model_diversity_final/similarity.py` uses:
- **Metric name:** cosine_similarity_l2_normalized
- **Formula:** cos(u, v) = (u·v) / (||u|| ||v||) after fleet benign z-score
- **Input columns:** behavior graph candidate features from `behavior_only_vehicle_normalized` view
- **Aggregation:** mean over pairwise combinations within/cross masks
- **Validation:** all pairwise values must satisfy -1 - 1e-6 ≤ sim ≤ 1 + 1e-6
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_campaign_decision_audit(path: Path) -> None:
    text = """# Campaign decision logic audit

## Pipeline

1. Graph build (cosine kNN, cross-vehicle constrained)
2. C2: DBSCAN on descriptor features; C3: GraphSAGE embeddings then DBSCAN
3. `_qualify_clusters_ieee`: size ≥ min_cluster_size, vehicles ≥ min_vehicles, cohesion ≥ min_behavioral_cohesion
4. `assign_final_decisions`: **every event in a qualifying cluster → coordinated_attack**
5. `_decisions_to_predictions`: predicted_malicious = local_alert OR weak_signal OR coordinated

## Root cause: universal benign vehicle inclusion (diagnostic run)

1. **Descriptor pool:** only weak/strong windows become nodes → all 15 benign fleet vehicles have weak_signal=1.
2. **Cluster assignment:** large qualifying clusters span most fleet vehicles.
3. **No membership gate:** benign nodes in cluster automatically receive coordinated_attack.
4. **Vehicle metric conflation:** `predicted_attacked = max(predicted_malicious)` counts weak benign as attacked.

## Final Phase 4 fix

- **Campaign gate** (`campaign_gate.py`): clusters must pass anomalous-ratio, weak-only cap, cohesion, and cross-model edge requirements.
- **Event membership:** coordinated only if cluster gated AND (local_alert=1 OR embedding confidence ≥ threshold).
- **Metrics:** membership precision/recall use `fleet_campaign_member` vs `ground_truth_campaign_member` at vehicle level.
- **Same gate for C2 and C3** — only representation differs.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_false_campaign_analysis(path: Path) -> None:
    text = """# Final false campaign analysis

See `results/campaign_membership_error_breakdown.csv` for per-run membership errors.

Diagnostic corrected Phase 4: false campaign rate 1.0 with 15/15 benign vehicles included — **invalid for publication**.

Final Phase 4 applies validation-tuned campaign gate before coordinated membership assignment.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
