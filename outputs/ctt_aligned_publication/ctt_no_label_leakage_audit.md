# CTT no-label-leakage audit

Confirmed separation:

1. **Scenario construction** (`src/ctt/scenarios.py`) may use `label` / `attack_type`
   only to build controlled evaluation scenarios and GT vehicle sets.
2. **Model inference** (`run_publication_fleet_pipeline`) uses only the 9-D GNN
   behavioural features + constrained-kNN graph structure.
3. **GraphSAGE training** uses `supervision='structure'`:
   `L_link` on connected embeddings + `λ L_score` vs min-max anomaly_score.
   Attack labels / types / campaign membership are not training targets.
4. **DBSCAN** clusters GraphSAGE embeddings after StandardScaler→PCA(8); no labels.
5. **Campaign gate** uses only `r_k` (distinct vehicles), `|C_k|`, and centroid
   behavioural cohesion `c_k`. Attack-type fields appear as `eval_*` columns only.

GNN feature columns (order): ['anomaly_score', 'message_rate', 'frame_count', 'burstiness', 'mean_inter_arrival_time', 'std_inter_arrival_time', 'can_id_entropy', 'most_common_can_id_ratio', 'payload_entropy']

Verdict: **PASS** — GT attack labels/types/campaign membership do not enter
GraphSAGE inputs, training targets, clustering, or campaign decisions.
