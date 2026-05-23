#!/usr/bin/env bash
# Push local changes to github.com/iyadsec/fleet-can-ids
set -euo pipefail

REPO_NAME="${GITHUB_REPO_NAME:-fleet-can-ids}"
GITHUB_USER="iyadsec"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .git ]]; then
  echo "Not a git repo. Run: git init && gh repo create ${GITHUB_USER}/${REPO_NAME} --public --source=. --remote=origin --push"
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
fi

git add -A
if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git commit -m "${1:-chore: sync local changes}"
fi

git push origin main
echo "Synced: https://github.com/${GITHUB_USER}/${REPO_NAME}"
