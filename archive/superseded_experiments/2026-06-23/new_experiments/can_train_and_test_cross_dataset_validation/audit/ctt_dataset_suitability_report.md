# CTT Dataset Suitability Report

## Summary

- Files: **236**
- Total samples: **193,240,845**
- Vehicles found: **4** (chevrolet_impala, chevrolet_silverado, chevrolet_traverse, subaru_forester)
- Attack types found: **9** (combined_spoofing, dos, fuzzing, gear_spoofing, interval, rpm_spoofing, speed_spoofing, standstill, systematic)
- Attack families: **9** (combined_spoofing, dos, fuzzing, gear_spoofing, interval, rpm_spoofing, speed_spoofing, standstill, systematic)

## Per-vehicle benign onboarding

**Supported.** Each vehicle has attack-free captures in train_01 subsets across the four sets (known-vehicle training splits).

## Unknown-vehicle testing

**Supported** via `test_02` and `test_04` subsets.

## Unknown-attack testing

**Supported** via `test_03` and `test_04` subsets.

## Controlled fleet campaign simulation

**Supported.** Four vehicles (Chevrolet Impala, Silverado, Traverse; Subaru Forester) enable cross-vehicle scenario construction with behaviourally related attack families.

## Real synchronized fleet campaigns

**Not present.** Attacks were conducted per-vehicle on rural roads; the dataset does not contain naturally synchronized multi-vehicle campaigns.

## Schema normalizability

**Yes.** Source CSV schema (`timestamp`, `arbitration_id`, `data_field`, `attack`) maps cleanly to the framework normalized schema.

## Set-vehicle policy

- **set_01**: known=Chevrolet Impala, unknown=Chevrolet Silverado
- **set_02**: known=Chevrolet Traverse, unknown=Subaru Forester
- **set_03**: known=Chevrolet Silverado, unknown=Subaru Forester
- **set_04**: known=Subaru Forester, unknown=Chevrolet Traverse
