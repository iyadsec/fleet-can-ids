# TRACE_SPLIT_RECOVERY.md

## Recovery classification: **A. EXACT_SPLIT_RECOVERED** (balanced publication)

Unchanged from prior checkpoint. Authoritative assignments remain the committed
`balanced_split_manifest.csv` / recovered copy — **no new random split**.

### Sources

| Artifact | Path |
|----------|------|
| Segment/split assignments | `recovered_publication_pipeline/.../manifests/balanced_split_manifest.csv` |
| Per-window provenance (historical) | `.../manifests/balanced_window_manifest.csv` (110,121 windows) |
| Copy under this experiment | `recovered_balanced_split_manifest.csv` |

### Split statistics

| Split | Segments | Windows (historical + reconstructed) |
|-------|----------|----------------------------------------|
| train | 18 | **74,858** |
| validation | 8 | **20,669** |
| test | 8 | **14,594** |
| **Total** | **34** | **110,121** |

### Windowing (reconstructed from raw)

| Parameter | Value | Result |
|-----------|-------|--------|
| Window size | 100 | Exact |
| Stride | 50 | Exact |
| Scope | Within segment `[segment_start, segment_end)` | Exact |
| Cross-trace windows | Forbidden | None |

**Reconstructed total = 110,121 = historical total.** All 34 segment window counts match `WINDOW_PROVENANCE.csv`.

### Leakage checks (assignments)

| Check | Result |
|-------|--------|
| `segment_id` in >1 split | **0** |
| Same source file in >1 split | Only via disjoint contiguous segments + guard gaps |
| Altered split | **No** |
