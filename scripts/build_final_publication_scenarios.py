#!/usr/bin/env python3
"""Build authoritative publication package for S0–S4, campaign size, and edge sensitivity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.experiments.final_publication_scenarios.runner import run_publication_package


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-edge", action="store_true", help="Skip edge sensitivity experiment")
    p.add_argument("--edge-dry-only", action="store_true", help="Dry audit edge grid only")
    args = p.parse_args(argv)
    summary = run_publication_package(skip_edge=args.skip_edge, edge_dry_only=args.edge_dry_only)
    print("Publication package built:", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
