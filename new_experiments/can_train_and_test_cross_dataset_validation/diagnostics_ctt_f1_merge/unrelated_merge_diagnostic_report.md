# Unrelated Incident Merge Diagnostic

## Root cause

Official CTT unrelated scenario yields **incorrect_merge_rate=1.0** on all seeds because:

1. DBSCAN collapses weak-candidate descriptors into **one large multi-vehicle cluster** (best_cluster=0, n_clusters≈1).
2. `evaluate_campaign()` sets `incorrect_merge_rate = raw_fleet_signal` when ≥2 vehicles share an attack-bearing cluster, even if `fleet_campaign_detected=0`.
3. Behaviour-only cosine graphs on ~40 scenario nodes (production: ~100k) connect unrelated attack families across vehicles at threshold 0.85 with cross_vehicle_cap=20.
4. GraphSAGE self-supervised embeddings do not separate attack families when edge density is high relative to node count.

## Evidence

set_id  seed attack_types                             vehicles  n_nodes  n_edges  cross_vehicle_edges  mean_cross_vehicle_similarity  cluster_count  largest_cluster_size attack_type_mixture_best_cluster         vehicle_mixture_best_cluster  dbscan_merged_all  incorrect_merge_rate  best_cluster              campaign_decision_reason  official_incorrect_merge_rate
set_01   101  dos|fuzzing chevrolet_impala|chevrolet_silverado       40       11                    0                            0.0              1                    40                      dos|fuzzing chevrolet_impala|chevrolet_silverado                  1                   1.0             0 multi_vehicle_attack_cluster_selected                            1.0
set_02   101  dos|fuzzing   chevrolet_traverse|subaru_forester       40       23                    0                            0.0              1                    40                      dos|fuzzing   chevrolet_traverse|subaru_forester                  1                   1.0             0 multi_vehicle_attack_cluster_selected                            1.0
set_03   101  dos|fuzzing  chevrolet_silverado|subaru_forester       40       43                    0                            0.0              1                    40                      dos|fuzzing  chevrolet_silverado|subaru_forester                  1                   1.0             0 multi_vehicle_attack_cluster_selected                            1.0
set_04   101  dos|fuzzing   chevrolet_traverse|subaru_forester       40       38                    0                            0.0              1                    40                      dos|fuzzing   chevrolet_traverse|subaru_forester                  1                   1.0             0 multi_vehicle_attack_cluster_selected                            1.0

**DIAGNOSTIC ONLY**