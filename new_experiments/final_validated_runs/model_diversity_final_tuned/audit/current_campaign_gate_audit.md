# Current campaign gate audit (provisional)

## CampaignGateConfig defaults

- min_distinct_vehicles: 3
- min_anomalous_member_ratio: 0.4
- max_weak_only_ratio: 0.5
- min_membership_confidence: 0.55
- min_cluster_cohesion: 0.15
- min_cross_vehicle_edges: 1
- require_cross_model_path: True
- max_benign_vehicle_inclusion: 3

## Decision flow (provisional)

1. DBSCAN clusters on similarity (C2) or GraphSAGE embeddings (C3).
2. `_qualify_clusters_ieee` marks multi-vehicle clusters as qualifying.
3. `gate_qualifying_clusters` filters clusters by vehicle count, anomalous ratio, cohesion, cross-model edges.
4. `_assign_gated_decisions` sets coordinated membership when cluster passes gate AND (local_strong OR score >= min_membership_confidence).

**Issue:** Campaign acceptance and member acceptance are combined in one step.

## Root cause of high provisional false campaign rate

Primary: **metric implementation** — legacy `false_campaign_alert_rate` equals 1.0 whenever any qualifying cluster exists.

Secondary contributors:
- Permissive campaign acceptance (min_distinct_vehicles=3, min_anomalous_ratio=0.40)
- Member promotion via weak_signal + coordinated decision path
- No separate benign-support ratio at campaign level

## Tuned pipeline changes

- Separate `accept_campaign_clusters` and `accept_cluster_members`
- Corrected false-campaign semantics (A–D)
- Validation-only grid search with constrained objectives