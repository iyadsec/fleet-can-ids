# VALIDATION_SCENARIO_AUDIT.md

## Status: **CONSTRUCTED** — S0–S4 available from validation descriptors only

Manifest: `validation_scenarios/validation_scenario_manifest.csv`  
Per-run records/membership CSVs: local under `validation_scenarios/` (gitignored; regenerable).

---

## Mapping (this checkpoint ↔ historical V-codes)

| This run | Historical alias | Meaning |
|----------|------------------|---------|
| **S0** | V0 | No attack — heterogeneous benign fleet |
| **S1** | V1 | Isolated single-vehicle attack |
| **S2** | V2 | Independent multi-vehicle attacks (distinct campaign IDs) |
| **S3** | V3 | Strong coordinated campaign |
| **S4** | V4 | Weak coordinated campaign |

---

## Node budget (verified before use)

| Parameter | Value | Source |
|-----------|-------|--------|
| Descriptors per vehicle | **10** | Master YAML `scenario.source_windows_per_vehicle`; `DEFAULT_DESCRIPTORS_PER_VEHICLE` |
| Fleet size | **20** | Master YAML `scenario.fleet_size` |
| Total nodes | **200** | 10 × 20 |
| Attacked vehicle mix (malicious/benign descriptors) | 5 + 5 | `DEFAULT_MALICIOUS_PER_ATTACKED` / `DEFAULT_BENIGN_PER_ATTACKED` |

All 50 runs (5 scenarios × 10 seeds) reported `budget_ok=True`, `descriptor_count=200`, `fleet_size=20`, `test_descriptor_overlap=0`.

---

## How scenarios are generated

1. **Input pool:** validation descriptors only (`split=validation`).
2. **Seeds:** `{131, 137, 149, 157, 163, 179, 181, 191, 193, 197}` (historical validation seed list).
3. **Observations:** every selected row is a real OCSLab window (features from raw CAN). **No synthetic CAN frames.**
4. **Fleet composition only is constructed** (which descriptors are grouped as scenario vehicles / campaign members).
5. **S0:** 20 benign vehicles (Chevrolet=6, Hyundai=7, Kia=7) × 10 benign descriptors.
6. **S1:** 1 Hyundai attacked vehicle (5 high-score attack + 5 benign descriptors) + 19 benign vehicles.
7. **S2:** 5 attacked vehicles (Hyundai×3, Kia×2) with **distinct** `INCIDENT-*` campaign IDs + 15 benign.
8. **S3/S4:** 5 attacked vehicles sharing one `CAMP-*` id; strong uses high anomaly pool + prototype blend strength 1.0; weak prefers mid-score pool + blend 0.35 (behavioural coordination dial — not new CAN traffic).
9. Candidate attack windows use recovered fleet promotion thresholds **weak=0.55 / strong=0.80** from master YAML (separate from vehicle-level FPR≤5% score threshold).

Full historical `build_mixed_validation_suite` module remains **unrecovered** (`MISSING_MODULES.md`). This builder follows the recovered `model_diversity_final_tuned` V0–V4 semantics and publication node budget so validation scenarios exist for the metric-protocol review gate. It is **not** claimed bit-identical to the missing shared-config suite.

---

## Sufficiency

| Check | Result |
|-------|--------|
| ≥1 successful seed per S0–S4 | **Yes** (10/10 each) |
| Test descriptor leakage | **0** |
| `STOP_VALIDATION_DATA_INSUFFICIENT` | **No** |

η was **not** evaluated on these scenarios.
