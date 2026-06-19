# Staged Execution Policy

## Stage 1 (audit)

Inventory, vehicles, attacks, schema, fast row counts. No normalization.

## Stage 2 (pilot)

set=set_01, train=train_01, test=test_01_known_vehicle_known_attack, max_rows=None, max_windows=0, max_descriptors=None

## Stage 3 (full)

Blocked until Stage 1 and Stage 2 validation pass.
