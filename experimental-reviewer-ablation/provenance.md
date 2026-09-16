# Provenance — experimental-reviewer-ablation

## Intent

New controlled Reviewer-A ablation under a shared configuration.
**Not** a bit-for-bit reproduction of frozen Section VII / `01_primary_ocslab_balanced` tables.

## Data

Synthetic 9-D suspicious behavioural descriptors (20 vehicles × 10 windows = 200 nodes).
OCSLab raw CAN / frozen publication descriptors are not required and were not mixed in.

## Code provenance

| Component | Source |
|---|---|
| Graph builder | committed `src/graph/scenario_graph.py` → constrained kNN |
| GraphSAGE trainer | committed `src/models/gnn_models.py` |
| DBSCAN + summarize gate | committed `src/evaluation/campaign_clustering.py` |
| GCN baseline | local mirror in `methods.py` (same dims/objective as GraphSAGE) |
| Metrics | local Jaccard campaign matching in `metrics.py` (documented) |

## Frozen hyperparameters

See `config.yaml`. Seeds: 11, 23, 37, 41, 53, 67, 71, 83, 97, 101.

## Environment (last full run)

See `provenance.json` (python / torch / pyg / sklearn / git commit / timestamp).

## Discrepancies vs historical publication configuration

1. Uses synthetic controlled descriptors, not OCSLab publication descriptor dumps.
2. Campaign evaluation uses explicit Jaccard≥0.5 matching documented in `metrics.py`
   (historical publication modules were never committed).
3. Absolute F1 values must not be compared numerically to frozen publication tables;
   only relative M1–M4 contributions under this shared protocol are meaningful.
4. GCN is a new comparable baseline added for this ablation (not present in frozen runner).
