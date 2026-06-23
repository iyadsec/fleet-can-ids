# Set Pilot Comparison: set_01

**Before:** v1 | **After:** v2

| Metric | Before | After |
|--------|--------|-------|
| Files processed | 28 | 42 |
| Rows processed | 20406189 | 26911803 |
| Windows | 400000 | 530098 |
| Descriptors | 50000 | 100000 |
| Subsets in windows | test_01_known_vehicle_known_attack, test_02_unknown_vehicle_known_attack, test_03_known_vehicle_unknown_attack, train_01 | test_01_known_vehicle_known_attack, test_02_unknown_vehicle_known_attack, test_03_known_vehicle_unknown_attack, test_04_unknown_vehicle_unknown_attack, train_01 |
| Vehicles in descriptors | chevrolet_impala, chevrolet_silverado | chevrolet_impala, chevrolet_silverado |
| Cross-vehicle edge % | 0.0004 | 0.2771 |
| Descriptor rate | 0.1388881172882373 | 0.2040399918384003 |

## Scenario means

### campaign_detected
| Scenario | Before | After |
|----------|--------|-------|
| benign_fleet_control | 1.0 | 0.0 |
| isolated_attack | 1.0 | 1.0 |
| strong_campaign | 0.0 | 1.0 |
| unrelated_incidents | 0.0 | 1.0 |
| weak_campaign | 0.0 | 1.0 |

### false_campaign
| Scenario | Before | After |
|----------|--------|-------|
| benign_fleet_control | 1.0 | 0.0 |
| isolated_attack | 1.0 | 0.0 |
| strong_campaign | 0.0 | 0.0 |
| unrelated_incidents | 0.0 | 0.0 |
| weak_campaign | 0.0 | 0.0 |

### campaign_f1
| Scenario | Before | After |
|----------|--------|-------|
| benign_fleet_control | 0.0 | 0.0 |
| isolated_attack | 0.0 | 0.0 |
| strong_campaign | 0.0 | 1.0 |
| unrelated_incidents | 0.0 | 0.0 |
| weak_campaign | 0.0 | 1.0 |
