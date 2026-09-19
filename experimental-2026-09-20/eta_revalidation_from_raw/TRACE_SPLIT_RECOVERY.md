# TRACE_SPLIT_RECOVERY.md

## Recovery classification: **A. EXACT_SPLIT_RECOVERED** (balanced publication)

The authoritative balanced FLEET-GUARD publication source-trace / segment assignments are recovered **exactly** from committed artifacts. No new random split was created.

### Sources

| Artifact | Path |
|----------|------|
| Segment/split assignments | `recovered_publication_pipeline/.../manifests/balanced_split_manifest.csv` |
| Per-window provenance | `.../manifests/balanced_window_manifest.csv` (110,121 windows) |
| Platform aggregates | `.../manifests/platform_split_summary.csv` |
| IF training provenance | `.../manifests/local_model_training_manifest.csv` |
| Split construction code | `recovered_publication_pipeline/src/experiments/final_end_to_end_publication_run_balanced/balanced_split.py` |
| Copy under this experiment | `recovered_balanced_split_manifest.csv` |

Master config (`balanced_master_experiment.yaml`) states intended ratios train/val/test **0.70 / 0.15 / 0.15** with `guard_frames: 100`. The balanced Chevrolet / long attack-free traces use **contiguous segments with guard gaps**; other attack traces are often assigned as **complete_trace** to a single split (to avoid mid-attack leakage).

---

## Split statistics (exact from manifest)

| Split | Segments | Windows (from window manifest) |
|-------|----------|--------------------------------|
| train | 18 | 74,858 |
| validation | 8 | 20,669 |
| test | 8 | 14,594 |
| **Total** | **34** | **110,121** |

### Leakage checks on recovered assignments

| Check | Result |
|-------|--------|
| `segment_id` appears in >1 split | **0** |
| Same `source_file` in >1 split | **6** files — **only** via disjoint contiguous segments (`train` / `seg_val` / `seg_test`) on long traces |
| Overlap across segment interiors | Forbidden by construction (`contiguous_segment` + guard gaps in `balanced_split.py`) |

### Vehicles × split (segment counts)

| Vehicle | train | validation | test |
|---------|-------|------------|------|
| Chevrolet | 4 | 4 | 4 |
| Hyundai | 7 | 2 | 2 |
| Kia | 7 | 2 | 2 |

### Split methods

| Method | Meaning |
|--------|---------|
| `contiguous_segment` | Long trace cut into train/val/test segments with 100-frame guards |
| `complete_trace` | Entire source file assigned to one split |

`validation_status` on all 34 rows: **`pass`**.

---

## Window construction (verified against recovered windows)

| Parameter | Config / code | Window manifest |
|-----------|---------------|-----------------|
| Window size | 100 frames | **100** only |
| Stride / overlap | 50 / 50 | Consecutive starts differ by **50** (sampled) |
| Cross-trace windows | Forbidden | Windows carry `source_file` + `segment_id` |

See `WINDOW_PROVENANCE.csv` (segment-level aggregation of the recovered window manifest).

---

## What is still blocked without raw files

Exact recovery of **assignments** does **not** restore raw CAN bytes. Without the challenge root on the VM we cannot:

- re-window from frames,
- re-fit Isolation Forests,
- regenerate descriptors,
- rebuild validation fleet scenarios from real observations.

Those steps remain pending dataset mount. The split itself must **not** be re-randomized; reuse `recovered_balanced_split_manifest.csv`.

---

## Classification note vs paper “70/15/15”

The paper states source-trace-level 70/15/15. The balanced publication implements that intent via:

1. contiguous 70/15/15-style cuts on long attack-free (and Chevrolet) traces, and  
2. whole-trace assignment of many attack files to a single split.

Window counts are **not** exactly 70/15/15 globally (train-heavy because many full attack traces are train-only). That is the **historical balanced design**, recovered exactly — not reinvented here.
