# SET_01 Cross-Dataset Set Pilot Summary

## 1. Files processed

28 files for set_01 (inventory: 52)

## 2. Rows and windows

Rows read (manifest sum): 20406189; windows: 400,000

## 3. Vehicles

chevrolet_impala, chevrolet_silverado

## 4. Attack types

combined_spoofing, dos, fuzzing, gear_spoofing, interval, rpm_spoofing, speed_spoofing, standstill, systematic

## 5. Local detection by subset

precision  recall      f1  roc_auc
subset_name                                                             
test_01_known_vehicle_known_attack       0.0555     1.0  0.1052   0.9944
test_02_unknown_vehicle_known_attack     0.0000     0.0  0.0000      NaN
test_03_known_vehicle_unknown_attack     0.0000     0.0  0.0000      NaN

## 6. Local detection by attack type

Empty DataFrame
Columns: [precision, recall, f1]
Index: []

## 7. Descriptor transmission rate

0.1389

## 8. Graph statistics

{
  "num_nodes": 50000.0,
  "num_edges": 241758.0,
  "cross_vehicle_edges": 1.0,
  "cross_vehicle_edge_pct": 0.0004136367772731,
  "cross_manufacturer_edges": 0.0,
  "average_degree": 9.73515613989168,
  "isolated_node_rate": 0.00666,
  "similarity_threshold": 0.85,
  "knn_cap": 10.0,
  "temporal_edges": 0.0,
  "mean_similarity": 0.927394650143477,
  "connected_components": 391.0,
  "largest_component": 37647.0
}

## 9. Benign-fleet campaign-free

1.0

## 10. Isolated attacks

1.0

## 11. Unrelated incidents separate

19.0

## 12. Strong campaigns detected

0.0

## 13. Weak campaigns detected

0.0

## 14. Campaign sizes supported

[1, 2]

## 15. Edge-connectivity trend

nan

## 16. Runtime and memory

0.0s, peak 0.0 MB

## 17. Ready for full four-set run

NO — review caps/validation first

## Safety caps applied

{
  "max_rows_per_file": 1000000,
  "max_windows": 400000,
  "max_descriptors": 50000
}

## Tables

table_SET01_1_dataset_summary, table_SET01_2_local_detection_by_subset, table_SET01_3_local_detection_by_attack, table_SET01_4_descriptor_compactness, table_SET01_5_graph_statistics, table_SET01_6_scenario_results, table_SET01_7_campaign_size_sensitivity, table_SET01_8_edge_sensitivity

## Figures

figure_SET01_1_local_score_distribution, figure_SET01_2_detection_by_subset, figure_SET01_4_graph_statistics, figure_SET01_5_campaign_f1_by_scenario, figure_SET01_6_campaign_size_sensitivity, figure_SET01_7_edge_count_vs_campaign_f1

## Scenario summary

campaign_detected  campaign_f1  false_campaign
scenario                                                            
benign_fleet_control                1.0          0.0             1.0
isolated_attack                     1.0          0.0             1.0
strong_campaign                     0.0          0.0             0.0
unrelated_incidents                 0.0          0.0             0.0
weak_campaign                       0.0          0.0             0.0
