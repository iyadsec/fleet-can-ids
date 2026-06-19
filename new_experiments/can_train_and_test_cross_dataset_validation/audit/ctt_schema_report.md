# CTT Schema Report

## Source schema

Columns: `timestamp, arbitration_id, data_field, attack`

Sample rows from `DoS-1.csv`:

```
   timestamp arbitration_id       data_field  attack
1.672531e+09            1E5 460196E001FE6701       0
1.672531e+09            1E9 0010000C00580000       0
1.672531e+09            0F9 01B34003167D33FA       0
```

## Target normalized schema

timestamp, can_id, dlc, byte_0..byte_7, label, is_attack, attack_type, vehicle_id, manufacturer, dataset_set, subset_name, source_file, source_row_index

## CAN ID format

Hex arbitration_id without 0x prefix (e.g., `0C1`, `1E5`)

## Payload format

Contiguous hex string in `data_field`; DLC derived from byte length

## Label format

Column `attack`: 0=benign, 1=attack frame

## Timestamp

Present as float epoch seconds in all inspected files
