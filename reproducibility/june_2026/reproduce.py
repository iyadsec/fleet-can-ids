#!/usr/bin/env python3
"""Reproduce the June 2026 balanced OCSLab run from raw data on a clean export of commit 61a8203.

Stages (all executed with cwd = TREE, because resolve_project_root() uses the working directory):
  overlay         copy this snapshot's src/ into TREE (refuses to replace differing files) and move the
                  committed June outputs to TREE/_june_reference/ for comparison
  preprocess      raw OCSLab -> clean CAN, windows, features (final_end_to_end_publication_run.preprocessing)
  archived_split  install the committed June split manifests and build the balanced window_features.csv
                  with the merge used by scripts/build_balanced_split.py (default)
  split           scripts/build_balanced_split.py; NOT deterministic, because the split is seeded with
                  hash(vehicle_model), which Python randomises per process (see split_hash_diagnostic.py)
  run             scripts/run_balanced_publication_run.py (local IDS, descriptors, shared configuration
                  search, 90 scenario runs, edge sweep, tables)
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIG = "new_experiments/final_end_to_end_publication_run"
BAL = "new_experiments/final_end_to_end_publication_run_balanced"
SPLIT_MANIFESTS = ("balanced_window_manifest.csv", "balanced_split_manifest.csv", "platform_split_summary.csv")

PREPROCESS = """
from pathlib import Path
from src.experiments.final_end_to_end_publication_run.guard import FinalEndToEndPublicationGuard
from src.experiments.final_end_to_end_publication_run.runner import _load_master_config, _write_master_config
from src.experiments.final_end_to_end_publication_run.preprocessing import rebuild_windows_and_features
root = Path.cwd()
out = FinalEndToEndPublicationGuard(root).output_root
for sub in ("configs", "manifests", "processed", "results/vehicle_level"):
    (out / sub).mkdir(parents=True, exist_ok=True)
_write_master_config(out, root)
rebuild_windows_and_features(root, out, _load_master_config(out, root))
"""

# Same merge as scripts/build_balanced_split.py at 61a8203, applied to the archived June manifests.
MERGE_FEATURES = f"""
import shutil
from pathlib import Path
import pandas as pd
src_features = Path("{ORIG}/processed/window_features.csv")
balanced_wm = pd.read_csv("{BAL}/manifests/balanced_window_manifest.csv")
proc = Path("{BAL}/processed")
proc.mkdir(parents=True, exist_ok=True)
feat = pd.read_csv(src_features)
join_cols = ["window_id", "vehicle_model", "source_file"]
feat = feat.drop(columns=["split"], errors="ignore").merge(
    balanced_wm[join_cols + ["split", "segment_id"]].drop_duplicates(),
    on=join_cols,
    how="inner",
)
feat.to_csv(proc / "window_features.csv", index=False)
shutil.copy2("{ORIG}/processed/clean_can_data.csv", proc / "clean_can_data.csv")
"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlay(tree: Path, dataset_dir: str | None) -> None:
    for src in sorted((HERE / "src").rglob("*.py")):
        dst = tree / src.relative_to(HERE)
        if dst.exists() and sha256(dst) != sha256(src):
            raise SystemExit(f"Refusing to replace differing file: {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    ref = tree / "_june_reference"
    for rel in (ORIG, BAL):
        d = tree / rel
        if d.exists() and not (ref / rel).exists():
            (ref / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(ref / rel))
    if dataset_dir:
        cfg = tree / "configs/fleet_ids.yaml"
        lines = cfg.read_text(encoding="utf-8").splitlines(keepends=True)
        lines = [f"  external_dataset_dir: {dataset_dir}\n" if ln.startswith("  external_dataset_dir:") else ln for ln in lines]
        cfg.write_text("".join(lines), encoding="utf-8")


def archived_split(tree: Path) -> None:
    import yaml

    ref = tree / "_june_reference" / BAL
    june_prefix = yaml.safe_load((ref / "configs/balanced_master_experiment.yaml").read_text())["dataset"]["external_dataset_dir"]
    # dataset_loader records resolved paths (e.g. /tmp -> /private/tmp on macOS)
    new_prefix = str(Path(yaml.safe_load((tree / "configs/fleet_ids.yaml").read_text())["data"]["external_dataset_dir"]).resolve())
    (tree / BAL / "manifests").mkdir(parents=True, exist_ok=True)
    for name in SPLIT_MANIFESTS:
        text = (ref / "manifests" / name).read_text(encoding="utf-8")
        (tree / BAL / "manifests" / name).write_text(text.replace(june_prefix, new_prefix), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True, help="clean export of commit 61a8203")
    ap.add_argument("--dataset-dir", help="directory containing the OCSLab folder (sets data.external_dataset_dir)")
    stages = ["overlay", "preprocess", "archived_split", "split", "run"]
    ap.add_argument("--stages", nargs="+", default=["overlay", "preprocess", "archived_split", "run"], choices=stages)
    args = ap.parse_args()
    if {"archived_split", "split"} <= set(args.stages):
        raise SystemExit("choose either archived_split or split")
    tree = args.tree.resolve()
    if "overlay" in args.stages:
        overlay(tree, args.dataset_dir)
    commands = {
        "preprocess": [sys.executable, "-c", PREPROCESS],
        "archived_split": [sys.executable, "-c", MERGE_FEATURES],
        "split": [sys.executable, "scripts/build_balanced_split.py"],
        "run": [sys.executable, "scripts/run_balanced_publication_run.py"],
    }
    for stage in ("preprocess", "archived_split", "split", "run"):
        if stage in args.stages:
            print(f"== {stage}", flush=True)
            if stage == "archived_split":
                archived_split(tree)
            subprocess.run(commands[stage], cwd=tree, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
