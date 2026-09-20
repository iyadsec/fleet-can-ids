# CHECKPOINT_VERDICT.md

# **READY_FOR_METRIC_PROTOCOL_REVIEW**

## Gate checklist

| Requirement | Status |
|-------------|--------|
| 34/34 traces resolved | **Yes** |
| Exact historical source split reconstructed | **Yes** (`EXACT_SPLIT_RECOVERED`) |
| Windowing reproduced (100/50 → 110,121) | **Yes** (exact match) |
| Vehicle-level reconstruction defensible | **Yes** (PR-AUC MATCH; F1 gap vs P4 explained by operating point) |
| Validation descriptors available | **Yes** (20,669; test not exposed) |
| Legitimate validation fleet scenarios | **Yes** (S0–S4, 50/50 seeds, budget 10×20) |
| Proposed metric protocol fully specified | **Yes** (`PROPOSED_REVISED_METRIC_PROTOCOL.md` v1, unused) |
| η completely unevaluated | **Yes** — candidates `{2,3,5,10}` predeclared only |
| CTT / manuscript / June artifacts | **Untouched** |

---

## What was done

1. Mounted OCSLab under `OCSLAB_DATASET_DIR` (prior download; gitignored).
2. Re-windowed all 34 segments from the exact balanced split.
3. Re-fit per-vehicle benign-train Isolation Forests; selected thresholds with validation **FPR≤5%** max-recall.
4. Audited vs immutable `table_P4` (scores MATCH; F1 policy divergence explained).
5. Exported validation descriptors (24-D dᵢ + 9-D gᵢ inputs).
6. Built controlled validation scenarios S0–S4 from validation data only.
7. Left metric protocol v1 unchanged (no V2 required by reconstruction findings).

## What was **not** done

- No η sweep / selection  
- No GraphSAGE training for η  
- No final OCSLab fleet evaluation  
- No CTT  
- No manuscript edits  
- No June artifact overwrites  
- Metric protocol **not** used for any scored η comparison  

---

## Stop conditions evaluated

| Code | Triggered? |
|------|------------|
| `STOP_VEHICLE_PIPELINE_MISMATCH` | **No** (difference explained) |
| `STOP_VALIDATION_DATA_INSUFFICIENT` | **No** |
| `STOP_RECONSTRUCTION_MISMATCH` | **No** (windows exact) |
| `STOP_OTHER` | **No** |

---

## Next step (human)

Review and approve `PROPOSED_REVISED_METRIC_PROTOCOL.md` **before** any η∈{2,3,5,10} evaluation.

## Verdict code

```text
READY_FOR_METRIC_PROTOCOL_REVIEW
```
