# False campaign metric definition

Four error types are reported separately. They must not be combined into one ambiguous rate.

## A. False campaign alert

A predicted multi-vehicle campaign exists when no ground-truth campaign exists (`n_gt == 0` and `n_accepted > 0`).

## B. Benign membership contamination

A real campaign is detected, but benign vehicles are incorrectly included in fleet campaign membership.

## C. Extra false campaign cluster

A valid campaign is detected, but one or more unrelated campaign clusters are also generated: `max(n_accepted - n_gt, 0)`.

## D. Incorrect merging

Unrelated incidents are merged into one campaign (`n_gt > 1` and `n_accepted == 1`).

## Legacy metric bug (provisional Phase 4)

In `compute_campaign_metrics`, when `spec_expects_campaign=True`:

```
false_campaign_alert_rate = n_detected / max(n_detected, 1)
```

Therefore any run with at least one qualifying DBSCAN cluster reports `false_campaign_alert_rate ≈ 1.0`,
regardless of ground-truth campaign presence. This is a **metric semantics** issue, not purely gate failure.

Example when n_detected=2, expect_campaign=True: {'n_detected': 2, 'expect_campaign': True, 'legacy_false_campaign_alert_rate': 1.0, 'legacy_always_one_when_detected': True}