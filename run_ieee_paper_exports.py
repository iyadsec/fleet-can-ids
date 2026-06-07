#!/usr/bin/env python3
"""
Generate publication-quality IEEE Experimental Evaluation exports (H1–H4).

Runs final GNN fleet decision evaluation first if H4 artifacts are missing.

Usage:
  python3 run_ieee_paper_exports.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.ieee_paper_exports import run_ieee_paper_exports
from src.utils import get_logger

logger = get_logger("run_ieee_paper_exports")


def main() -> int:
    written = run_ieee_paper_exports(repo_root=_ROOT)
    for key, val in written.items():
        if isinstance(val, dict):
            for fmt, path in val.items():
                logger.info("%s.%s -> %s", key, fmt, path)
        else:
            logger.info("%s -> %s", key, val)
    print("IEEE paper exports completed → paper/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
