# S3_S4_GRAPH_AUDIT.md

## Runtime graph parameters (verified)

- cosine τ = **0.95**
- k_same = **2**
- k_cross = **5**
- metric = cosine (`build_cross_vehicle_constrained_knn_edges`)
- Path: `publication_fleet_core.build_publication_fleet_graph` → fleet graph builder
- **No τ / k changes made.**

## Aggregate S3/S4 graph statistics

| Metric | Mean | Median | Min | Max |
|--------|-----:|-------:|----:|----:|
| nodes | 200 | 200 | 200 | 200 |
| undirected edges | 313.6 | 311.5 | 247 | 378 |
| same-vehicle edges | 60.1 | 58.0 | 34 | 83 |
| cross-vehicle edges | 253.5 | 253.0 | 200 | 304 |
| connected components | 72.6 | 72.5 | 58 | 86 |
| cross pairs pass τ | 362.2 | 355.5 | 212 | 504 |
| cross-edge density | 0.0133 | 0.0133 | 0.0105 | 0.0160 |
| same-GT cosine mean | 0.1692 | 0.1706 | 0.0809 | 0.2637 |
| unrelated cosine mean | 0.0275 | 0.0247 | 0.0067 | 0.0494 |

## Interpretation

Mean same-GT cross-vehicle cosine = **0.1692 ≪ τ=0.95**.

The reconstructed S3/S4 “coordination” (prototype blend of behavioural features) does **not** place GT-campaign descriptors near each other in the 9-D GNN feature space used for edges. Cross-vehicle edges that do form are therefore largely **not** GT↔GT campaign edges.

Fleet-scale DBSCAN mega-clusters still form (via GraphSAGE embedding geometry + eps=0.5), swallowing GT vehicles with many benign vehicles → low cohesion → β failure.

No graph parameters were changed.
