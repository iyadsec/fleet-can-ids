# ETA decision impact

**Status:** deferred — `STOP_NO_VALIDATION_SPLIT`

η was not selected and no OCSLab fleet scenarios were executed in this experiment. Therefore the following quantities were **not** calculated (calculation requires runnable scenarios after a frozen η):

| Quantity | Value |
|----------|-------|
| DBSCAN candidate clusters before η | NOT COMPUTED |
| Rejected only because `\|C_k\| < η` | NOT COMPUTED |
| Retained after η | NOT COMPUTED |
| Decisions changed specifically because of η | NOT COMPUTED |
| Strong / weak breakdown | NOT COMPUTED |
| Campaign-size breakdown | NOT COMPUTED |

## Why this matters

Without these counts we cannot yet classify η as:

- **A.** an active decision criterion that changes predictions; or  
- **B.** formally explicit but redundant for these experiments.

That classification must be **calculated**, not inferred. It remains pending until a legitimate validation→freeze→final pipeline can run.

## What is already established in unit tests

Independence of η from DBSCAN and the semantic split `|C_k|` (nodes) vs `r_k` (vehicles) are covered by `tests/test_ctt_aligned_publication_fleet.py`. Those tests do **not** substitute for scenario-level impact analysis on OCSLab data.
