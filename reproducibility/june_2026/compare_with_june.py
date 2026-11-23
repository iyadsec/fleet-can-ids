#!/usr/bin/env python3
"""Compare a reproduced June 2026 run with the archived June outputs.

Committed outputs (TREE/_june_reference/, moved there by reproduce.py) are compared file by file. Outputs that
were never committed are compared with a local June copy (--june-root) when available, otherwise with the
SHA-256 values in june_local_artifact_hashes.csv.

Absolute dataset paths are written into several outputs. When the reproduction reads OCSLab from a different
location, pass --path-map REPRO_PREFIX=JUNE_PREFIX: reproduced text is rewritten with the June prefix before an
exact comparison, and the master-configuration hash is mapped only if the rewritten master configuration
hashes to the June value. CSVs that still differ are compared cell by cell; wall-clock columns are reported
separately.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
BAL = "new_experiments/final_end_to_end_publication_run_balanced"
MASTER = f"{BAL}/configs/balanced_master_experiment.yaml"
WALL_CLOCK = re.compile(r"(_time$|latency|peak_memory|runtime|^run_id$|seconds|_ms$|timestamp)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def remap(text: str, maps: list[tuple[str, str]]) -> str:
    for old, new in maps:
        text = text.replace(old, new)
    return text


def text_identical(ref: Path, new: Path, maps: list[tuple[str, str]]) -> bool:
    with ref.open(encoding="utf-8", newline="") as a, new.open(encoding="utf-8", newline="") as b:
        while True:
            x, y = a.readline(), b.readline()
            if remap(y, maps) != x:
                return False
            if not x:
                return True


def compare_csv(ref: Path, new: Path, maps: list[tuple[str, str]]) -> tuple[str, str]:
    x = pd.read_csv(ref)
    y = pd.read_csv(new)
    for c in y.columns:
        if y[c].dtype == object:
            y[c] = y[c].map(lambda v: remap(v, maps) if isinstance(v, str) else v)
    if list(x.columns) != list(y.columns) or len(x) != len(y):
        return "DIFFERENT", f"shape/columns {x.shape} vs {y.shape}"
    diff_cols, wall_cols = [], []
    for c in x.columns:
        u, v = x[c], y[c]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
            same = np.allclose(u.to_numpy(float), v.to_numpy(float), rtol=0, atol=1e-9, equal_nan=True)
        else:
            same = u.astype(str).equals(v.astype(str))
        if not same:
            (wall_cols if WALL_CLOCK.search(c) else diff_cols).append(c)
    if diff_cols:
        return "DIFFERENT", "columns: " + ";".join(diff_cols)
    if wall_cols:
        return "equal_except_wall_clock", "wall-clock columns: " + ";".join(wall_cols)
    return "equal_values", "numeric formatting only"


def compare(ref: Path, new: Path, maps: list[tuple[str, str]]) -> tuple[str, str]:
    if not new.exists():
        return "MISSING", ""
    if sha256(ref) == sha256(new):
        return "identical", ""
    if maps and text_identical(ref, new, maps):
        return "identical_after_path_map", ""
    if ref.suffix == ".csv":
        return compare_csv(ref, new, maps)
    if ref.suffix == ".yaml":
        same = yaml.safe_load(remap(new.read_text(encoding="utf-8"), maps)) == yaml.safe_load(ref.read_text(encoding="utf-8"))
        return ("equal_values", "YAML line folding only") if same else ("DIFFERENT", "")
    return "DIFFERENT", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--june-root", type=Path, help="local June project root holding the uncommitted outputs")
    ap.add_argument("--path-map", action="append", default=[], help="REPRO_PREFIX=JUNE_PREFIX")
    ap.add_argument("--out", type=Path, default=HERE / "reproduction_check")
    args = ap.parse_args()
    tree = args.tree.resolve()
    ref_root = tree / "_june_reference"
    maps = [tuple(m.split("=", 1)) for m in args.path_map]

    notes = []
    if maps:
        mapped = remap((tree / MASTER).read_text(encoding="utf-8"), maps)
        june_text = (ref_root / MASTER).read_text(encoding="utf-8")
        june_hash = (ref_root / BAL / "configs/balanced_master_experiment.sha256").read_text().strip()
        repro_hash = (tree / BAL / "configs/balanced_master_experiment.sha256").read_text().strip()
        if hashlib.sha256(mapped.encode()).hexdigest() == june_hash:
            maps.append((repro_hash, june_hash))
            notes.append(f"master config is byte-identical to June after path map (hash {june_hash})")
        elif yaml.safe_load(mapped) == yaml.safe_load(june_text):
            maps.append((repro_hash, june_hash))
            notes.append("master config equals June after path map (YAML line folding of the longer path differs); "
                         f"hash {repro_hash} mapped to {june_hash}")
        else:
            notes.append("master config does NOT equal June after path map")

    rows = []
    for ref in sorted((ref_root / BAL).rglob("*")):
        if not ref.is_file() or ref.suffix not in {".csv", ".yaml", ".json", ".sha256"} or "/runs/" in str(ref):
            continue
        rel = ref.relative_to(ref_root)
        result, detail = compare(ref, tree / rel, maps)
        rows.append(dict(artifact=str(rel), reference="committed", result=result, detail=detail))

    for r in pd.read_csv(HERE / "june_local_artifact_hashes.csv").itertuples():
        new = tree / r.path
        local = args.june_root / r.path if args.june_root else None
        if local is not None and local.exists() and sha256(local) == r.sha256:
            result, detail = compare(local, new, maps)
            rows.append(dict(artifact=r.path, reference="local_june_copy", result=result, detail=detail))
        elif not new.exists():
            rows.append(dict(artifact=r.path, reference="recorded_sha256", result="MISSING", detail=""))
        else:
            same = sha256(new) == r.sha256
            rows.append(dict(artifact=r.path, reference="recorded_sha256", result="identical" if same else "DIFFERENT",
                             detail="" if same else "sha256 differs"))

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "artifact_comparison.csv", index=False)
    (args.out / "comparison_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    print("\n".join(notes))
    print(df.groupby(["reference", "result"]).size().to_string())
    bad = df[df.result.isin(["DIFFERENT", "MISSING"])]
    if not bad.empty:
        print(bad.to_string(index=False))
    return 1 if not bad.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
