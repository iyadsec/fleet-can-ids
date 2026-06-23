# Local Onboarding Protocol

## Approach

Per-set, per-known-vehicle benign-only Isolation Forest training. Weak (90th pct) and strong (97.5th pct) thresholds calibrated on held-out benign validation windows. Unknown vehicles without benign training data are evaluated via descriptor/fleet layer only.

## Models trained

- **set_01/chevrolet_impala**: benign-only Isolation Forest; weak=-0.0068, strong=0.0421
