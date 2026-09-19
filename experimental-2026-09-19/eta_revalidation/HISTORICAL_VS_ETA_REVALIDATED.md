# Historical vs η-revalidated results

**Status:** comparison **deferred** — `STOP_NO_VALIDATION_SPLIT`

No new final OCSLab evaluation was run. Historical publication artifacts were **not** modified.

## Historical reference (untouched)

Source: `experimental-2026-06-23/01_primary_ocslab_balanced/`

| Metric | Historical value |
|--------|------------------|
| Strong Campaign F1 (cs=2/5/10) | 0.533 / 0.733 / 1.000 |
| Weak Campaign F1 (cs=2/5/10) | 0.067 / 0.500 / 0.717 |
| Unrelated incorrect merging | 0.400 |

## New η-explicit results

| Metric | New value | Absolute difference | Relative difference |
|--------|-----------|---------------------|---------------------|
| Strong Campaign F1 | **NOT RUN** | — | — |
| Weak Campaign F1 | **NOT RUN** | — | — |
| Unrelated incorrect merging | **NOT RUN** | — | — |

## Notes

- No attempt was made to force reproduction of historical numbers.
- η was not selected; therefore no revalidated final tables exist under `final/`.
- A second independent blocker (`STOP_METRIC_IMPLEMENTATION_UNRECOVERED`) would also prevent a defensible numeric comparison even if validation data were restored, until the publication matcher/emitter is recovered or a separately proposed explicit evaluation protocol is ratified.
