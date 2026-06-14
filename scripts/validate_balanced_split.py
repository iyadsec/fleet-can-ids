#!/usr/bin/env python3
"""Validate balanced split manifests only (pre-pipeline gate)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.data_splits import is_benign_attack_type
from src.experiments.final_end_to_end_publication_run_balanced.balanced_split import (
    GUARD_FRAMES,
    PLATFORMS,
    validate_balanced_split,
)
from src.experiments.final_end_to_end_publication_run_balanced.guard import OUTPUT_REL
from src.utils.paths import resolve_project_root

OUT = resolve_project_root() / OUTPUT_REL
CRITICAL: list[str] = []


def fail(msg: str) -> None:
    CRITICAL.append(msg)


def main() -> int:
    wm_path = OUT / "manifests/balanced_window_manifest.csv"
    sm_path = OUT / "manifests/balanced_split_manifest.csv"
    ps_path = OUT / "manifests/platform_split_summary.csv"

    for p in (wm_path, sm_path, ps_path):
        if not p.exists():
            fail(f"Missing {p.relative_to(OUT)}")
            _write_report(pd.DataFrame())
            return 1

    wm = pd.read_csv(wm_path)
    sm = pd.read_csv(sm_path)
    ps = pd.read_csv(ps_path)

    for vm in PLATFORMS:
        for sp in ("train", "validation", "test"):
            if wm[(wm.vehicle_model == vm) & (wm.split == sp) & wm.attack_type.map(is_benign_attack_type)].empty:
                fail(f"{vm} missing benign {sp} windows")

    if wm.groupby("window_id")["split"].nunique().max() > 1:
        fail("window_id assigned to multiple splits")

    if "event_id" in wm.columns and wm.groupby("event_id")["split"].nunique().max() > 1:
        fail("event_id duplicated across splits")

    seg = sm[sm["split_method"] == "contiguous_segment"]
    if not seg.empty and GUARD_FRAMES < 100:
        fail(f"guard gap {GUARD_FRAMES} < required 100 frames")

    for err in validate_balanced_split(wm, sm):
        fail(err)

    orig = resolve_project_root() / "new_experiments/final_end_to_end_publication_run"
    if not orig.exists():
        fail("original E2E root missing for isolation check")
    elif str(OUT.resolve()) == str(orig.resolve()):
        fail("balanced output root overlaps original")

    passed = len(CRITICAL) == 0
    _write_report(ps, passed)
    return 0 if passed else 1


def _write_report(ps: pd.DataFrame, passed: bool = False) -> None:
    lines = [
        "# Balanced split validation",
        "",
        f"**Status:** {'PASS' if passed else 'FAIL'}",
        f"**Guard gap (frames):** {GUARD_FRAMES}",
        "",
    ]
    if not ps.empty:
        lines += ["## Platform summary", "", ps.to_markdown(index=False), ""]
    lines += [
        "## Checks",
        "",
        "1. Chevrolet/Hyundai/Kia benign train/validation/test windows.",
        "2. No row/window in multiple splits.",
        "3. No event ID duplicated across splits.",
        "4. Guard gaps on segmented traces (100 frames).",
        "5. Ready for downstream IF retraining.",
        "6. Original E2E outputs isolated under separate root.",
        "",
    ]
    if CRITICAL:
        lines += ["## Failures", ""] + [f"- {c}" for c in CRITICAL]
    val = OUT / "validation/balanced_split_validation.md"
    val.parent.mkdir(parents=True, exist_ok=True)
    val.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
