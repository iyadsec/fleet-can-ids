#!/usr/bin/env python3
"""Run balanced-split end-to-end publication experiment."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.final_end_to_end_publication_run_balanced.runner import run_balanced_publication


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="Balanced-split end-to-end publication experiment")
    p.add_argument("--dry-run", action="store_true", help="Single seed smoke test")
    p.add_argument("--skip-edge", action="store_true", help="Skip edge-connectivity sweep")
    args = p.parse_args(argv)

    result = run_balanced_publication(dry_run=args.dry_run, skip_edge=args.skip_edge)
    print(f"Complete: {result['out_root']} runs={result['n_runs']} hash={result['master_hash']}")
    return 0 if result.get("split_passed") and result.get("n_runs", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
