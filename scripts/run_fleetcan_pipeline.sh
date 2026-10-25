#!/usr/bin/env bash
# Run the FLEET-GUARD experimental pipeline on a restored fleetcan-sim-scale-1.0 tree.
#
# Usage:
#   scripts/run_fleetcan_pipeline.sh /Users/iyadatieh/fleetcan_data
#   scripts/run_fleetcan_pipeline.sh /Users/iyadatieh/fleetcan_data/fleetcan-sim-scale-1.0
#   RELEASE_ROOT=... scripts/run_fleetcan_pipeline.sh
#
# Default pilot: N=100 S0–S3 flooding (see src/data/fleetcan_adapter.default_pilot_sessions).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT="${1:-${RELEASE_ROOT:-}}"
if [ -z "$INPUT" ]; then
  echo "Usage: $0 /path/to/fleetcan_data[/fleetcan-sim-scale-1.0]" >&2
  exit 2
fi

# Accept either the parent folder or the release folder itself.
if [ -d "$INPUT/fleetcan-sim-scale-1.0" ]; then
  RELEASE="$INPUT/fleetcan-sim-scale-1.0"
elif [ -d "$INPUT/data/raw" ] || [ -d "$INPUT/raw" ]; then
  RELEASE="$INPUT"
else
  echo "FAIL: cannot find fleetcan-sim-scale-1.0 under: $INPUT" >&2
  echo "Expected .../fleetcan_data/fleetcan-sim-scale-1.0/data/raw/..." >&2
  exit 1
fi

RELEASE="$(cd "$RELEASE" && pwd)"
echo "Release root: $RELEASE"

python experiments/01c_validate_fleetcan_dataset.py --pilot --release-root "$RELEASE"
python experiments/01_load_fleetcan_dataset.py --pilot --release-root "$RELEASE" \
  --output data/processed/clean_can_data.csv
python experiments/01b_validate_clean_dataset.py --schema-only --config configs/fleetcan_sim.yaml
python experiments/02_generate_windows.py --config configs/fleetcan_sim.yaml
python experiments/03_extract_features.py
python experiments/04_train_vehicle_ids.py --config configs/fleetcan_sim.yaml --no-autoencoder
python experiments/05_generate_descriptors.py --config configs/fleetcan_sim.yaml \
  --regenerate-predictions --no-autoencoder --primary-model isolation_forest
python experiments/06_build_graph.py --config configs/fleetcan_sim.yaml

echo "SUCCESS: fleetcan pipeline finished (N=100 pilot)."
echo "  clean:       data/processed/clean_can_data.csv"
echo "  features:    data/processed/window_features.csv"
echo "  descriptors: data/processed/anomaly_descriptors.csv"
echo "  graph:       data/processed/fleet_graph.pt"
echo "  metrics:     outputs/metrics/"
