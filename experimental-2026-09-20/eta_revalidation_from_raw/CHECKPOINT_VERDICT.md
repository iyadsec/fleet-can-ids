# CHECKPOINT_VERDICT.md

# **METRIC_PROTOCOL_V2_READY**

Prior reconstruction gate (`READY_FOR_METRIC_PROTOCOL_REVIEW`) remains satisfied.  
Protocol **V1 was not approved unchanged**. **V2** is specified, implemented, and unit-tested.

## Gate checklist

| Requirement | Status |
|-------------|--------|
| 34/34 traces resolved | **Yes** |
| Exact historical source split reconstructed | **Yes** (`EXACT_SPLIT_RECOVERED`) |
| Windowing reproduced (100/50 → 110,121) | **Yes** (exact match) |
| Vehicle-level reconstruction defensible | **Yes** |
| Validation descriptors / S0–S4 scenarios | **Yes** |
| Metric protocol V2 fully specified | **Yes** (`PROPOSED_REVISED_METRIC_PROTOCOL_V2.md`) |
| V2 unit tests (synthetic vehicle ids) | **13/13 passed** |
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
7. Wrote `PROPOSED_REVISED_METRIC_PROTOCOL_V2.md` (V1 not approved unchanged).
8. Implemented `revised_campaign_metrics_v2` + synthetic vehicle-id unit tests (**all passed**).

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

## Metric protocol

| Item | Value |
|------|-------|
| Active proposal | `PROPOSED_REVISED_METRIC_PROTOCOL_V2.md` |
| Protocol id | `revised_jaccard_0.5_v2` |
| Implementation | `src/evaluation/revised_campaign_metrics_v2.py` |
| Unit tests | `tests/test_revised_campaign_metrics_v2.py` (**all passed**) |
| Approved for η use | **No** — awaiting human approval |
| Used to score η | **No** |

## Next step (human)

Review and approve `PROPOSED_REVISED_METRIC_PROTOCOL_V2.md` **before** any η∈{2,3,5,10} evaluation.

## Verdict code

```text
METRIC_PROTOCOL_V2_READY
```
