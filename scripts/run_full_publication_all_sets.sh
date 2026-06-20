#!/usr/bin/env bash
# Run full publication stage for all CTT sets with v2 safety caps.
set -euo pipefail
cd /workspace

CAPS=(
  --max-rows-per-file 475000
  --max-windows 800000
  --max-descriptors 100000
  --resume
  --skip-existing
  --confirm-large-run
)

for SET in set_01 set_02 set_03 set_04; do
  echo "========== FULL STAGE: ${SET} =========="
  python3 scripts/run_can_train_and_test_cross_dataset.py \
    --stage full --set-id "${SET}" "${CAPS[@]}"
done

echo "========== POOLED AGGREGATION =========="
python3 - <<'PY'
from src.ctt.constants import OUTPUT_ROOT, SETS
from src.ctt.full_stage import generate_pooled_outputs
generate_pooled_outputs(OUTPUT_ROOT, list(SETS))
print("Pooled outputs generated")
PY

python3 scripts/validate_can_train_and_test_full.py
echo "ALL FULL STAGE RUNS COMPLETE"
