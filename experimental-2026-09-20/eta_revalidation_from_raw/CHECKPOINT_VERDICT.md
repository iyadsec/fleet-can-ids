# CHECKPOINT_VERDICT.md

# **METRIC_PROTOCOL_V3_READY**

Prior reconstruction remains satisfied. Protocol **V2 was corrected** with a final membership-aggregation change → **V3**.

## Gate checklist

| Requirement | Status |
|-------------|--------|
| 34/34 traces resolved | **Yes** |
| Exact split / windowing 110,121 | **Yes** |
| Vehicle-level reconstruction defensible | **Yes** |
| Validation descriptors / S0–S4 | **Yes** |
| Metric protocol V3 fully specified | **Yes** (`PROPOSED_REVISED_METRIC_PROTOCOL_V3.md`) |
| V3 unit tests (synthetic vehicle ids) | **17/17 passed** |
| η completely unevaluated | **Yes** — candidates `{2,3,5,10}` predeclared only |
| CTT / manuscript / June artifacts | **Untouched** |

---

## Metric protocol

| Item | Value |
|------|-------|
| Active proposal | `PROPOSED_REVISED_METRIC_PROTOCOL_V3.md` |
| Protocol id | `revised_jaccard_0.5_v3` |
| Change vs V2 | Unmatched predicted campaigns contribute \(\lvert V(P)\rvert\) to **primary** micro membership FP |
| Implementation | `src/evaluation/revised_campaign_metrics_v2.py` (`evaluate_protocol_v3`) |
| Unit tests | `tests/test_revised_campaign_metrics_v2.py` (**17/17 passed**) |
| Approved for η use | **No** — awaiting human approval |
| Used to score η | **No** |

## Next step (human)

Review and approve `PROPOSED_REVISED_METRIC_PROTOCOL_V3.md` **before** any η∈{2,3,5,10} evaluation.

## Verdict code

```text
METRIC_PROTOCOL_V3_READY
```
