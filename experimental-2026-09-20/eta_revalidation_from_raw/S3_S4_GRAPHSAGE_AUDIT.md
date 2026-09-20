# S3_S4_GRAPHSAGE_AUDIT.md

## Runtime implementation path

- Trainer: `src.models.gnn_models.train_graphsage_fleet_correlation`
- Model: `GraphSAGEFleetCorrelator` (SAGEConv, mean aggregation)
- Called from the frozen η-validation / diagnostic path only
- **No alternate/legacy GNN path invoked**

## Runtime hyperparameters (frozen; unchanged)

| Field | Value |
|-------|-------|
| input_dim | 9 |
| feature_names | anomaly_score, message_rate, frame_count, burstiness, mean_inter_arrival_time, std_inter_arrival_time, can_id_entropy, most_common_can_id_ratio, payload_entropy |
| layers | 2 (9→64→32) |
| aggregation | mean |
| activation | ReLU after first SAGEConv |
| objective | structure (link reconstruction + λ·campaign MSE on anomaly_score) |
| epochs | 30 |
| optimizer | Adam |
| learning_rate | 0.01 |
| weight_decay | 0.0005 |
| λ | 0.25 |
| embedding_dim | 32 |

## Embedding separation (S3/S4 measured)

| Distance | Mean |
|----------|-----:|
| within-GT Euclidean | 6.5022 |
| GT↔unrelated Euclidean | 14.3086 |

Embeddings show **some** separation (within < between), but DBSCAN still produces fleet-scale clusters containing all 5 GT vehicles plus most benign vehicles. Separation is insufficient to yield compact, high-cohesion campaign clusters under frozen eps=0.5.

No retrain with different parameters was performed.
