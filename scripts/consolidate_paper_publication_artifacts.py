#!/usr/bin/env python3
"""Consolidate all paper figures/tables into paper/publication_artifacts/."""

from __future__ import annotations

import csv
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper/publication_artifacts"

# (git ref, repo path prefix, destination subfolder label)
BUNDLES = [
    ("origin/cursor/corrected-ctt-publication-8f28",
     "new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt",
     "01_corrected_ctt"),
    ("origin/cursor/corrected-ctt-publication-8f28",
     "new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected",
     "02_cross_dataset_ocslab_vs_ctt_corrected"),
    ("origin/cursor/local-ids-metric-comparison-8f28",
     "new_experiments/local_ids_ocslab_vs_ctt_metric_comparison",
     "03_local_ids_ocslab_vs_ctt"),
    ("origin/cursor/fleet-corrected-comparison-summary-8f28",
     "new_experiments/fleet_level_corrected_comparison_summary",
     "04_fleet_level_corrected_summary"),
    ("origin/cursor/ctt-f1-merge-diagnostics-8f28",
     "new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge",
     "05_ctt_f1_merge_diagnostics"),
]

INCLUDE = {".csv", ".md", ".tex", ".png", ".pdf", ".json"}


def git_files(ref: str, prefix: str) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", ref, "--", prefix],
        cwd=REPO, check=True, capture_output=True, text=True,
    )
    out = []
    for line in proc.stdout.splitlines():
        p = line.strip()
        if p and Path(p).suffix.lower() in INCLUDE:
            out.append(p)
    return out


def git_export(ref: str, repo_path: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = subprocess.run(
        ["git", "show", f"{ref}:{repo_path}"],
        cwd=REPO, check=True, capture_output=True,
    ).stdout
    dest.write_bytes(data)


def bucket(rel: Path) -> str:
    parts = rel.parts
    if "figures" in parts:
        return "figures"
    if "tables" in parts:
        return "tables"
    return "reports"


def dest_for(root: Path, rel: Path) -> Path:
    b = bucket(rel)
    if b in ("figures", "tables"):
        return root / b / rel.name
    return root / rel


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    manifest = []

    for ref, prefix, label in BUNDLES:
        try:
            files = git_files(ref, prefix)
        except subprocess.CalledProcessError:
            print(f"SKIP missing: {prefix} @ {ref}")
            continue
        dest_root = OUT / label
        for repo_path in files:
            rel = Path(repo_path).relative_to(prefix)
            dest = dest_for(dest_root, rel)
            git_export(ref, repo_path, dest)
            manifest.append({
                "bundle": label,
                "filename": rel.name,
                "category": bucket(rel),
                "dest_path": str(dest.relative_to(REPO)),
                "source_git_ref": ref,
                "source_repo_path": repo_path,
            })
        print(f"  {len(files):3d} files -> {label}")

    with (OUT / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()) if manifest else [])
        if manifest:
            w.writeheader()
            w.writerows(manifest)

    index = f"""# Paper Publication Artifacts — Single Folder

**Location (use this folder only for paper writing):**

```
paper/publication_artifacts/
```

**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}  
**Total files:** {len(manifest)}

This folder consolidates **all latest figures and tables** from corrected CTT, cross-dataset, local IDS, fleet, and diagnostic outputs. Original experiment roots are unchanged on their source branches; these are copies for convenience.

## Subfolders

| Folder | What it contains |
|--------|------------------|
| `01_corrected_ctt/` | CTT_CORR1–7 tables & figures, corrected local/scenario results |
| `02_cross_dataset_ocslab_vs_ctt_corrected/` | Fair OCSLab vs corrected CTT comparison |
| `03_local_ids_ocslab_vs_ctt/` | LOCAL_COMP1–6 local IDS pooled/per-vehicle comparison |
| `04_fleet_level_corrected_summary/` | FLEET_CORR1–6 fleet scenario & consistency-rule summary |
| `05_ctt_f1_merge_diagnostics/` | Diagnostic figures explaining low F1 & unrelated merge |

Each subfolder has `figures/` and `tables/` (plus reports/validation/results as needed).

## Original scattered locations (before consolidation)

| Content | Original path |
|---------|---------------|
| Corrected CTT | `new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/` |
| Cross-dataset (corrected) | `new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/` |
| Local IDS comparison | `new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/` |
| Fleet summary | `new_experiments/fleet_level_corrected_comparison_summary/` |
| F1/merge diagnostics | `new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge/` |

## Recommended for main paper

| Section | Table | Figure |
|---------|-------|--------|
| Local IDS | `03_local_ids_ocslab_vs_ctt/tables/LOCAL_COMP1_pooled_ocslab_vs_ctt.csv` | `03_local_ids_ocslab_vs_ctt/figures/figure_LOCAL_COMP1_pooled_comparison.png` |
| Fleet scenarios | `04_fleet_level_corrected_summary/tables/FLEET_CORR1_corrected_ctt_fleet_summary.csv` | `04_fleet_level_corrected_summary/figures/figure_FLEET_CORR2_unrelated_merge_before_after.png` |
| Cross-dataset | `02_cross_dataset_ocslab_vs_ctt_corrected/tables/table_CUR_COMP3_fleet_scenario_comparison.csv` | `02_cross_dataset_ocslab_vs_ctt_corrected/figures/figure_CUR_COMP3_scenario_outcomes.png` |

See `MANIFEST.csv` for full source traceability (git ref + original path per file).

## Regenerate this folder

```bash
python3 scripts/consolidate_paper_publication_artifacts.py
```
"""
    (OUT / "INDEX.md").write_text(index, encoding="utf-8")
    n_fig = len(list(OUT.rglob("figures/*")))
    n_tab = len(list(OUT.rglob("tables/*")))
    print(f"\nDone: {OUT.relative_to(REPO)}")
    print(f"  Figures: {n_fig}  Tables: {n_tab}  Total: {len(manifest)}")


if __name__ == "__main__":
    main()
