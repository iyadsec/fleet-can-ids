#!/usr/bin/env bash
# Enable auto-push hook for this repository (run once after clone)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p .git/hooks
cp .githooks/post-commit .git/hooks/post-commit
chmod +x .git/hooks/post-commit
echo "Installed post-commit hook: pushes to origin after each commit on main."
