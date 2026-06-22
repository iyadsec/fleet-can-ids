#!/usr/bin/env bash
# Pull OVERLEAF_CROSS_DATASET_ARTIFACTS from GitHub into your local CodeRepo clone (macOS).
# Run from anywhere: bash scripts/sync_overleaf_artifacts_mac.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ARTIFACT_DIR="$REPO_ROOT/OVERLEAF_CROSS_DATASET_ARTIFACTS"
FOLDER_NAME="OVERLEAF_CROSS_DATASET_ARTIFACTS"

cd "$REPO_ROOT"

echo "==> Repo: $REPO_ROOT"
echo "==> Fetching from GitHub..."
git fetch origin main

echo "==> Updating local main branch..."
git checkout main 2>/dev/null || git checkout -b main origin/main
git pull origin main

if [[ ! -d "$ARTIFACT_DIR" ]]; then
  echo "ERROR: $FOLDER_NAME not found after pull."
  echo "       Check you are in fleet-can-ids / CodeRepo and remote is github.com/iyadsec/fleet-can-ids"
  exit 1
fi

FILE_COUNT="$(find "$ARTIFACT_DIR" -type f | wc -l | tr -d ' ')"
echo "==> SUCCESS: $FOLDER_NAME exists with $FILE_COUNT files"
echo ""
echo "Absolute path:"
echo "  $ARTIFACT_DIR"
echo ""
echo "Opening in Finder..."
open "$ARTIFACT_DIR"
