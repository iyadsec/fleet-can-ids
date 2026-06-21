# CTT Label Consistency Report

Generated: 2026-06-21T16:11:37.890213+00:00

## Safest evaluation ground-truth rule

1. **Primary:** use `label==1` when the source file labels are reliable.
2. **Fallback (evaluation only):** if `attack_type != 'benign'` but `label==0`, treat as positive for metric computation.
3. **Never** pass `attack_type` into the model, graph features, or threshold calibration.

Total label/attack_type mismatches across manifests: 1,553,365.

## Silverado DoS check

                              subset  n_windows  mismatch_count label_values_present
test_02_unknown_vehicle_known_attack      18998           18998                  0.0
  test_01_known_vehicle_known_attack      13966           13966                  0.0

## Impact

Local F1 uses label-only → attack traffic with label=0 counts as benign → recall collapse on affected vehicles. Fleet scenarios correctly use `_is_attack_window()`.