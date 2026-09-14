#!/usr/bin/env python3
"""Build and validate balanced dataset split (split-only phase)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

from src.experiments.final_end_to_end_publication_run_balanced.balanced_split import (
    GUARD_FRAMES,
    build_balanced_split,
    validate_balanced_split,
)
from src.experiments.final_end_to_end_publication_run_balanced.chevrolet_audit import audit_chevrolet_sources
from src.experiments.final_end_to_end_publication_run_balanced.guard import BalancedPublicationGuard

ORIG_ROOT = _PROJECT_ROOT / "new_experiments/final_end_to_end_publication_run"


def main() -> int:
    guard = BalancedPublicationGuard(_PROJECT_ROOT)
    out = guard.ensure_directory_tree()

    src_wm = ORIG_ROOT / "manifests/window_manifest.csv"
    src_features = ORIG_ROOT / "processed/window_features.csv"
    src_clean = ORIG_ROOT / "processed/clean_can_data.csv"
    if not src_wm.exists():
        print(f"Missing source window manifest: {src_wm}")
        return 1

    window_manifest = pd.read_csv(src_wm)
    orig_trace = pd.read_csv(ORIG_ROOT / "manifests/final_split_manifest.csv") if (ORIG_ROOT / "manifests/final_split_manifest.csv").exists() else None

    audit_md = audit_chevrolet_sources(window_manifest, src_clean if src_clean.exists() else None, orig_trace)
    (out / "audit/chevrolet_source_availability.md").write_text(audit_md, encoding="utf-8")
    print("Wrote audit/chevrolet_source_availability.md")

    balanced_wm, split_manifest, platform_summary, errors = build_balanced_split(window_manifest, seed=42)
    errors.extend(validate_balanced_split(balanced_wm, split_manifest))

    balanced_wm.to_csv(out / "manifests/balanced_window_manifest.csv", index=False)
    split_manifest.to_csv(out / "manifests/balanced_split_manifest.csv", index=False)
    platform_summary.to_csv(out / "manifests/platform_split_summary.csv", index=False)

    passed = len(errors) == 0
    val_lines = [
        "# Balanced split validation",
        "",
        f"**Status:** {'PASS' if passed else 'FAIL'}",
        f"**Guard gap (frames):** {GUARD_FRAMES}",
        "",
        "## Platform summary",
        "",
        platform_summary.to_markdown(index=False),
        "",
    ]
    if errors:
        val_lines += ["## Errors", ""] + [f"- {e}" for e in errors]
    else:
        val_lines += [
            "## Checks",
            "",
            "1. Chevrolet/Hyundai/Kia benign train/validation/test windows present.",
            "2. Segment guard gaps applied where single-trace splitting required.",
            "3. No window_id assigned to multiple splits.",
            "4. Ready for downstream regeneration.",
        ]
    (out / "validation/balanced_split_validation.md").write_text("\n".join(val_lines), encoding="utf-8")

    orig_vs = _comparison_report(platform_summary, orig_trace, passed)
    (out / "audit/original_vs_balanced_split.md").write_text(orig_vs, encoding="utf-8")

    proc = out / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    if src_features.exists():
        feat = pd.read_csv(src_features)
        join_cols = ["window_id", "vehicle_model", "source_file"]
        feat = feat.drop(columns=["split"], errors="ignore").merge(
            balanced_wm[join_cols + ["split", "segment_id"]].drop_duplicates(),
            on=join_cols,
            how="inner",
        )
        feat.to_csv(proc / "window_features.csv", index=False)
    if src_clean.exists():
        shutil.copy2(src_clean, proc / "clean_can_data.csv")

    print(platform_summary.to_string(index=False))
    print(f"\nValidation: {'PASS' if passed else 'FAIL'} ({len(errors)} issues)")
    for e in errors[:10]:
        print(f"  - {e}")
    return 0 if passed else 1


def _comparison_report(platform_summary: pd.DataFrame, orig_trace: pd.DataFrame | None, passed: bool) -> str:
    lines = [
        "# Original vs balanced split",
        "",
        "## Chevrolet validation coverage",
        "",
        "| Metric | Original (trace-level) | Balanced |",
        "|--------|------------------------|----------|",
    ]
    orig_val_traces = 0
    if orig_trace is not None:
        orig_val_traces = len(
            orig_trace[
                (orig_trace.vehicle_model == "Chevrolet")
                & (orig_trace.split == "validation")
                & (orig_trace.ground_truth_malicious == 0)
            ]
        )
    bal_val = platform_summary[
        (platform_summary.vehicle_model == "Chevrolet") & (platform_summary.split == "validation")
    ]
    bal_val_seg = int(bal_val["source_segment_count"].iloc[0]) if not bal_val.empty else 0
    bal_val_win = int(bal_val["benign_window_count"].iloc[0]) if not bal_val.empty else 0
    lines.append(f"| Benign validation traces | {orig_val_traces} | {bal_val_seg} segment(s) |")
    lines.append(f"| Benign validation windows | (within-trace only) | {bal_val_win} |")
    lines += [
        "",
        "## Window counts by platform (balanced)",
        "",
        platform_summary.to_markdown(index=False),
        "",
        f"## Conclusion",
        "",
        f"Balanced split validation: **{'PASS' if passed else 'FAIL'}**. "
        "Chevrolet validation coverage is restored via disjoint contiguous segments. "
        "Downstream metrics must be regenerated before comparing scenario conclusions.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
