#!/usr/bin/env python3
"""Verify local dataset placement for FLEET-GUARD experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import DEFAULT_CTT_DATASET_ROOT
from src.data.dataset_loader import DEFAULT_EXTERNAL_DATASET
from src.utils.paths import resolve_project_root


def _check_ocslab(root: Path) -> list[str]:
    issues: list[str] = []
    if not root.exists():
        issues.append(f"OCSLab root missing: {root}")
        issues.append("  Download from https://ocslab.hksecurity.net/Datasets/car-hacking-dataset")
        issues.append("  Place under Dataset/ocslab/ or set OCSLAB_DATASET_DIR")
        return issues
    files = list(root.rglob("*.csv")) + list(root.rglob("*.txt")) + list(root.rglob("*.log"))
    if not files:
        issues.append(f"No CAN trace files found under {root}")
    else:
        issues.append(f"OCSLab: found {len(files)} trace files under {root}")
    return issues


def _check_ctt(root: Path) -> list[str]:
    issues: list[str] = []
    if not root.exists():
        issues.append(f"can-train-and-test root missing: {root}")
        issues.append("  Download DOI 10.11583/DTU.24805533")
        issues.append("  Place under Dataset/can-train-and-test/ or set CTT_DATASET_ROOT")
        return issues
    sets = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("set_")]
    if not sets:
        issues.append(f"No set_XX folders found under {root}")
    else:
        issues.append(f"CTT: found {len(sets)} dataset sets under {root}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Check FLEET-GUARD dataset placement")
    parser.add_argument("--check", action="store_true", help="Check dataset folders (default)")
    parser.add_argument("--ocslab-only", action="store_true")
    parser.add_argument("--ctt-only", action="store_true")
    args = parser.parse_args()

    root = resolve_project_root()
    ocs_root = Path(os.environ.get("OCSLAB_DATASET_DIR", DEFAULT_EXTERNAL_DATASET))
    ctt_root = Path(os.environ.get("CTT_DATASET_ROOT", DEFAULT_CTT_DATASET_ROOT))

    print("FLEET-GUARD dataset check")
    print(f"Repository root: {root}")
    print()

    messages: list[str] = []
    ok = True

    if not args.ctt_only:
        for line in _check_ocslab(ocs_root):
            print(line)
            if line.startswith("OCSLab root missing") or line.startswith("No CAN"):
                ok = False

    if not args.ocslab_only:
        print()
        for line in _check_ctt(ctt_root):
            print(line)
            if line.startswith("can-train-and-test root missing") or line.startswith("No set_"):
                ok = False

    print()
    if ok:
        print("Dataset check: PASS (at least one dataset appears present)")
        return 0
    print("Dataset check: datasets missing — see docs/datasets.md for setup instructions")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
