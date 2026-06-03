# Fleet Correlation Experiment — Summary

## Detection performance
- **Local-only detection score (F1):** 0.8456
- **Fleet-aware detection score (F1):** 0.8456
- **Detection gain (F1):** 0.00%
- **Recall gain:** 0.00%
- **F1 gain:** 0.00%
- **ROC-AUC gain:** 0.00%
- **Windows promoted by fleet correlation (not local_alert):** 0

## Fleet graph
- **Nodes:** 77233
- **Edges:** 814472
- **Average degree:** 21.09
- **Connected components:** 4
- **Largest component:** 72464
- **Edges before pruning (top-k union):** 814472
- **Density before pruning:** 0.00027309
- **Edges after pruning (sim ≥ 0.95):** 814472
- **Density after pruning:** 0.00027309
- **Components before pruning:** 4
- **Fleet cluster method:** connected_component

## Cluster quality
- **silhouette_score:** 0.1591
- **davies_bouldin_index:** 0.9433
- **adjusted_rand_index:** 0.0478
- **normalized_mutual_info:** 0.0124
- **mean_cluster_purity:** 0.8901
- **num_clusters:** 4.0000
- **mean_cluster_size:** 19308.2500
- **max_cluster_size:** 72464.0000

## False positives
- **Local FPR:** 0.4445
- **Fleet FPR:** 0.4445
- **FPR change (fleet − local):** +0.0000

## Interpretation

**Fleet-level graph reasoning did not improve F1 over local-only IDS** under the current thresholds; weak-signal promotions are listed in fleet_level_predictions.csv.

Fleet graph uses top-k cosine neighbours per descriptor (k=15) with similarity ≥ 0.95; suspicious groups are connected components with ≥ 3 vehicles and mean anomaly score ≥ 0.7 (behavioural clustering fallback if degenerate).

Parameters: top_k_neighbors=15, similarity_threshold=0.95, minimum_cluster_size=2, minimum_vehicle_count=3, fleet_cluster_score_threshold=0.7.
