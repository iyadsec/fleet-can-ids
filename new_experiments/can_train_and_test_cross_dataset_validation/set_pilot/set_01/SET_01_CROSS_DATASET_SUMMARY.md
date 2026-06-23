# SET_01 Cross-Dataset Set Pilot Summary

## 1. Files processed

42 files for set_01 (inventory: ?)

## 2. Rows and windows

Rows read (manifest sum): 26911803; windows: 530,098

## 3. Vehicles

chevrolet_impala, chevrolet_silverado

## 4. Attack types

combined_spoofing, dos, fuzzing, gear_spoofing, interval, rpm_spoofing, speed_spoofing, standstill, systematic

## 5. Local detection by subset

precision  recall      f1  roc_auc
subset_name                                                               
test_01_known_vehicle_known_attack         0.0555     1.0  0.1052   0.9944
test_02_unknown_vehicle_known_attack       0.0000     0.0  0.0000      NaN
test_03_known_vehicle_unknown_attack       0.0000     0.0  0.0000      NaN
test_04_unknown_vehicle_unknown_attack     0.0000     0.0  0.0000      NaN

## 6. Local detection by attack type

Empty DataFrame
Columns: [precision, recall, f1]
Index: []

## 7. Descriptor transmission rate

0.2040

## 8. Graph statistics

{
  "num_nodes": 100000.0,
  "num_edges": 710147.0,
  "cross_vehicle_edges": 1968.0,
  "cross_vehicle_edge_pct": 0.2771257218575872,
  "cross_manufacturer_edges": 0.0,
  "average_degree": 14.222567142656866,
  "isolated_node_rate": 0.00138,
  "similarity_threshold": 0.85,
  "cross_vehicle_threshold": 0.77,
  "knn_cap": 10.0,
  "cross_vehicle_cap": 20.0,
  "temporal_edges": 0.0,
  "mean_similarity": 0.9486186681905442,
  "connected_components": 172.0,
  "largest_component": 99405.0
}

## 9. Benign-fleet campaign-free

0.0

## 10. Isolated attacks

0.0

## 11. Unrelated incidents separate

1.0

## 12. Strong campaigns detected

1.0

## 13. Weak campaigns detected

1.0

## 14. Campaign sizes supported

[1, 2]

## 15. Edge-connectivity trend

nan

## 16. Runtime and memory

1831.3s, peak 2701.8 MB

## 17. Ready for full four-set run

YES

## Safety caps applied

{
  "max_rows_per_file": 475000,
  "max_windows": 800000,
  "max_descriptors": 100000
}

## Tables

table_SET01_1_dataset_summary, table_SET01_2_local_detection_by_subset, table_SET01_3_local_detection_by_attack, table_SET01_4_descriptor_compactness, table_SET01_5_graph_statistics, table_SET01_6_scenario_results, table_SET01_7_campaign_size_sensitivity, table_SET01_8_edge_sensitivity

## Figures

figure_SET01_1_local_score_distribution, figure_SET01_2_detection_by_subset, figure_SET01_4_graph_statistics, figure_SET01_5_campaign_f1_by_scenario, figure_SET01_6_campaign_size_sensitivity, figure_SET01_7_edge_count_vs_campaign_f1

## Scenario summary

local_or_incident_detected  fleet_campaign_detected  false_campaign  incorrect_merge_rate  campaign_f1
scenario                                                                                                                    
benign_fleet_control                         0.0                      0.0             0.0                   0.0          0.0
isolated_attack                              1.0                      0.0             0.0                   0.0          0.0
strong_campaign                              1.0                      1.0             0.0                   0.0          1.0
unrelated_incidents                          1.0                      0.0             0.0                   1.0          0.0
weak_campaign                                1.0                      1.0             0.0                   0.0          1.0
