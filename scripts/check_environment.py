#!/usr/bin/env python3
"""Check Python version and required dependencies for FLEET-GUARD."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("polars", "polars"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("torch_geometric", "torch-geometric"),
    ("networkx", "networkx"),
    ("matplotlib", "matplotlib"),
    ("yaml", "pyyaml"),
    ("scipy", "scipy"),
]

OPTIONAL = [
    ("seaborn", "seaborn"),
    ("tqdm", "tqdm"),
    ("pytest", "pytest"),
]


def check_module(import_name: str, pip_name: str) -> tuple[bool, str]:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        return True, str(version)
    except ImportError as exc:
        return False, f"missing ({pip_name}): {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="FLEET-GUARD environment check")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    print(f"Project root: {root}")
    print(f"Python:       {sys.version.split()[0]} ({sys.executable})")

    if sys.version_info < (3, 10):
        print("FAIL: Python 3.10+ required")
        ok = False
    else:
        print("PASS: Python version")
        ok = True

    for import_name, pip_name in REQUIRED:
        passed, detail = check_module(import_name, pip_name)
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {pip_name} — {detail}")
        ok = ok and passed

    if args.include_optional:
        for import_name, pip_name in OPTIONAL:
            passed, detail = check_module(import_name, pip_name)
            status = "PASS" if passed else "WARN"
            print(f"{status}: {pip_name} — {detail}")

    cfg = root / "configs" / "default.yaml"
    if cfg.exists():
        print(f"PASS: config found — {cfg}")
    else:
        print(f"FAIL: config missing — {cfg}")
        ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
