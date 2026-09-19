# Fleet Model Transfer Policy

## Decision

Shared publication FLEET-GUARD fleet path (src.evaluation.publication_fleet_core) with frozen OCSLab config

## Rationale

CTT reuses the same GraphSAGE structure training, constrained-kNN graph, StandardScaler→PCA→euclidean DBSCAN, and centroid cohesion campaign gate as the authoritative balanced OCSLab publication experiment. Only dataset loading and controlled scenario construction remain CTT-specific.

## Temporal edges

None — all edges are behavioural similarity only.

## Label leakage

Attack labels/types/campaign membership are evaluation-only; not used as GraphSAGE inputs, training targets, clustering inputs, or campaign-gate features.
