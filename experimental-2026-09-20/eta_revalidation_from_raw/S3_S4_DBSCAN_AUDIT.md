# S3_S4_DBSCAN_AUDIT.md

## Runtime clustering path

- `src.evaluation.campaign_clustering.run_dbscan` → `DbscanProjector`
- Confirmed pipeline: **StandardScaler → PCA(8) → DBSCAN(eps=0.5, min_samples=2, metric=euclidean)**

| Field | Value |
|-------|-------|
| StandardScaler | Yes (DbscanProjector) |
| PCA | 8 |
| eps | 0.5 |
| min_samples | 2 |
| metric | euclidean |

## Per-run cluster summary

| run_id | n_clusters | noise | sizes | vehicle composition (r_k list) |
|--------|-----------:|------:|-------|--------------------------------|
| val_S3_seed131 | 4 | 49 | [145, 2, 2, 2] | [20, 1, 1, 1] |
| val_S3_seed137 | 2 | 66 | [27, 107] | [7, 20] |
| val_S3_seed149 | 4 | 49 | [145, 2, 2, 2] | [20, 1, 1, 1] |
| val_S3_seed157 | 5 | 52 | [2, 140, 2, 2, 2] | [1, 20, 1, 1, 1] |
| val_S3_seed163 | 2 | 59 | [44, 97] | [8, 16] |
| val_S3_seed179 | 4 | 74 | [119, 2, 3, 2] | [20, 1, 1, 1] |
| val_S3_seed181 | 3 | 67 | [129, 2, 2] | [20, 1, 1] |
| val_S3_seed191 | 3 | 43 | [153, 2, 2] | [20, 1, 1] |
| val_S3_seed193 | 2 | 61 | [137, 2] | [20, 1] |
| val_S3_seed197 | 1 | 44 | [156] | [20] |
| val_S4_seed131 | 5 | 55 | [137, 2, 2, 2, 2] | [20, 1, 1, 1, 1] |
| val_S4_seed137 | 1 | 81 | [119] | [20] |
| val_S4_seed149 | 3 | 48 | [148, 2, 2] | [20, 1, 1] |
| val_S4_seed157 | 4 | 54 | [140, 2, 2, 2] | [20, 1, 1, 1] |
| val_S4_seed163 | 1 | 70 | [130] | [20] |
| val_S4_seed179 | 5 | 88 | [100, 5, 2, 3, 2] | [20, 3, 1, 1, 1] |
| val_S4_seed181 | 2 | 76 | [122, 2] | [20, 1] |
| val_S4_seed191 | 4 | 55 | [139, 2, 2, 2] | [20, 1, 1, 1] |
| val_S4_seed193 | 3 | 73 | [123, 2, 2] | [20, 1, 1] |
| val_S4_seed197 | 2 | 52 | [2, 146] | [1, 20] |

No clustering parameters were changed.

## Mega-cluster note

GT campaign vehicles almost always land in one very large cluster
(mean \|C_k\| ≈ 123, mean r_k ≈ 18.7) rather than a compact 5-vehicle campaign cluster.
This is the structural precursor to β failure (c_k ≪ 0.5) and Jaccard≈0.25.

No clustering parameters were changed.
