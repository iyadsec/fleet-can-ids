# Graph Calibration Recommendation

**DIAGNOSTIC ONLY — not applied to official CTT7 tables.**

## Sweep scope

200-node OCSLab-aligned graphs; sets set_01–set_02; seed 101; full cosine/k/cap/mutual grid on `unrelated_incidents`; reduced grid on other scenarios; DBSCAN eps/min_samples sub-sweep (1,068 configurations total).

## Graph-only sweep result

**No graph-only configuration reduced `incorrect_merge_rate` below 1.0.** At every tested combination of cosine threshold (0.72–0.90), k (5–50), cross-vehicle cap (1–10), and mutual kNN, DBSCAN still formed a single multi-vehicle cluster containing both unrelated attack families.

## Effective fix: campaign consistency rule

See `campaign_consistency_rule_results.csv`. With τ=0.88, k=10, cross_vehicle_cap=3, mutual kNN, and the **post-clustering consistency gate enabled**:

| Scenario | Without rule | With rule |
|----------|--------------|-----------|
| unrelated_incidents incorrect_merge | 1.0 | **0.0** (all 4 sets) |
| strong_campaign F1 | 1.0 | 1.0 (when graph connects campaign) |
| weak_campaign F1 | 1.0 | 1.0 (when graph connects campaign) |
| benign false_campaign | 0.0 | 0.0 |

## Recommended diagnostic graph protocol (pending confirmation)

- 200-node scenario graphs (OCSLab-aligned)
- similarity_threshold ≥ 0.88
- knn_cap = 10
- cross_vehicle_cap ≤ 3
- mutual kNN = True
- campaign consistency rule enabled
- temporal edges disabled
