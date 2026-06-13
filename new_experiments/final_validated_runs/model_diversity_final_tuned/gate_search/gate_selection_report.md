# Gate selection report

**Selected candidate:** `gate_0864`
**Config hash:** `88061ce9aa6101a2`
**Feasible candidate found:** True
**Test data used for selection:** False

## Constraints

- V0 false campaign alert rate ≤ 0.05
- V1 campaign alert rate = 0
- V2 incorrect merging rate ≤ 0.05
- Mean benign vehicles included ≤ 1.0
- Membership precision ≥ 0.80

## Selected gate parameters

```yaml
max_benign_support_ratio: 0.1
max_weak_only_ratio: 0.5
min_anomalous_ratio: 0.4
min_cluster_cohesion: 0.47
min_connected_platforms: 1
min_cross_model_edges: 0
min_cross_vehicle_neighbors: 1
min_cross_vehicle_support: 1
min_distinct_vehicles: 2
min_membership_confidence: 0.32
require_cross_model_path: true

```

## Candidates evaluated: 51840
## Pareto-optimal count: 5328