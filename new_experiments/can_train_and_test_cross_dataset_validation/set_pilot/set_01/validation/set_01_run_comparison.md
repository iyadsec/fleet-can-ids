# Set Pilot Comparison: set_01

**Before:** v2 | **After:** v3_metrics

| Metric | Before | After |
|--------|--------|-------|
| Files processed | 42 | 42 |
| Rows processed | 26911803 | 26911803 |
| Windows | 530098 | 530098 |
| Descriptors | 100000 | 100000 |
| Subsets in windows | test_01_known_vehicle_known_attack, test_02_unknown_vehicle_known_attack, test_03_known_vehicle_unknown_attack, test_04_unknown_vehicle_unknown_attack, train_01 | test_01_known_vehicle_known_attack, test_02_unknown_vehicle_known_attack, test_03_known_vehicle_unknown_attack, test_04_unknown_vehicle_unknown_attack, train_01 |
| Vehicles in descriptors | chevrolet_impala, chevrolet_silverado | chevrolet_impala, chevrolet_silverado |
| Cross-vehicle edge % | 0.2771 | 0.2771 |
| Descriptor rate | 0.2040399918384003 | 0.2040399918384003 |

## Scenario means

### local_or_incident_detected
| Scenario | Before | After |
|----------|--------|-------|
| benign_fleet_control | n/a | 0.0 |
| isolated_attack | n/a | 1.0 |
| strong_campaign | n/a | 1.0 |
| unrelated_incidents | n/a | 1.0 |
| weak_campaign | n/a | 1.0 |

### fleet_campaign_detected
| Scenario | Before | After |
|----------|--------|-------|
| benign_fleet_control | n/a | 0.0 |
| isolated_attack | n/a | 0.0 |
| strong_campaign | n/a | 1.0 |
| unrelated_incidents | n/a | 0.0 |
| weak_campaign | n/a | 1.0 |

### false_campaign
| Scenario | Before | After |
|----------|--------|-------|
| benign_fleet_control | 0.0 | 0.0 |
| isolated_attack | 0.0 | 0.0 |
| strong_campaign | 0.0 | 0.0 |
| unrelated_incidents | 0.0 | 0.0 |
| weak_campaign | 0.0 | 0.0 |

### incorrect_merge_rate
| Scenario | Before | After |
|----------|--------|-------|
| benign_fleet_control | n/a | 0.0 |
| isolated_attack | n/a | 0.0 |
| strong_campaign | n/a | 0.0 |
| unrelated_incidents | n/a | 1.0 |
| weak_campaign | n/a | 0.0 |

### campaign_f1
| Scenario | Before | After |
|----------|--------|-------|
| benign_fleet_control | 0.0 | 0.0 |
| isolated_attack | 0.0 | 0.0 |
| strong_campaign | 1.0 | 1.0 |
| unrelated_incidents | 0.0 | 0.0 |
| weak_campaign | 1.0 | 1.0 |
