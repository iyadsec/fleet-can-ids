#!/usr/bin/env python3
"""Consolidate canonical experiment artifacts and archive superseded local folders.

Creates:
  experimental-2026-06-19/     — draft-aligned + new results (canonical)
  archive/superseded_experiments/ — moved redundant workspace copies

Does not delete git history on remote branches.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TODAY = date.today().isoformat()  # 2026-06-19 when run on paper date
OUT = REPO / f"experimental-{TODAY}"
ARCHIVE = REPO / "archive" / "superseded_experiments"

CAMPAIGN_CLUSTERING = "origin/cursor/campaign-clustering"

# (git_ref, repo_relative_path, dest_under_OUT)
GIT_COPIES: list[tuple[str, str, str]] = [
    # --- 01 Primary OCSLab balanced (authoritative Section VII) ---
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/BALANCED_PUBLICATION_SUMMARY.md", "01_primary_ocslab_balanced/BALANCED_PUBLICATION_SUMMARY.md"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/audit/original_vs_balanced_split.md", "01_primary_ocslab_balanced/audit/original_vs_balanced_split.md"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P4_vehicle_level_results.csv", "01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P5_descriptor_compactness_and_privacy.csv", "01_primary_ocslab_balanced/tables/table_P5_descriptor_compactness_and_privacy.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P6_benign_isolated_unrelated_results.csv", "01_primary_ocslab_balanced/tables/table_P6_benign_isolated_unrelated_results.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P7_strong_campaign_results.csv", "01_primary_ocslab_balanced/tables/table_P7_strong_campaign_results.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P8_weak_campaign_results.csv", "01_primary_ocslab_balanced/tables/table_P8_weak_campaign_results.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P9_campaign_size_graph_and_cost.csv", "01_primary_ocslab_balanced/tables/table_P9_campaign_size_graph_and_cost.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P10_edge_connectivity_performance.csv", "01_primary_ocslab_balanced/tables/table_P10_edge_connectivity_performance.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/tables/table_P12_primary_statistical_tests.csv", "01_primary_ocslab_balanced/tables/table_P12_primary_statistical_tests.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/figures/figure_P4_strong_vs_weak_campaign_F1.pdf", "01_primary_ocslab_balanced/figures/figure_P4_strong_vs_weak_campaign_F1.pdf"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/figures/figure_P5_campaign_precision_recall_vs_size.pdf", "01_primary_ocslab_balanced/figures/figure_P5_campaign_precision_recall_vs_size.pdf"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/figures/figure_P8_campaign_F1_vs_unique_edges.pdf", "01_primary_ocslab_balanced/figures/figure_P8_campaign_F1_vs_unique_edges.pdf"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/results/scenario_evaluation/campaign_metrics.csv", "01_primary_ocslab_balanced/results/campaign_metrics.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/results/scenario_evaluation/safety_metrics.csv", "01_primary_ocslab_balanced/results/safety_metrics.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/results/campaign_size/strong_summary.csv", "01_primary_ocslab_balanced/results/strong_summary.csv"),
    (CAMPAIGN_CLUSTERING, "new_experiments/final_end_to_end_publication_run_balanced/results/campaign_size/weak_summary.csv", "01_primary_ocslab_balanced/results/weak_summary.csv"),
    # Framework ablation reference (M1/M2 descriptive)
    (CAMPAIGN_CLUSTERING, "new_experiments/publication_ready/tables/table_06_method_ablation.csv", "01_primary_ocslab_balanced/reference_framework_ablation/table_06_method_ablation.csv"),
]

LOCAL_COPIES: list[tuple[str, str]] = [
    ("new_experiments/baseline_ablation_comparison", "02_baseline_ablation"),
    ("OVERLEAF_CROSS_DATASET_ARTIFACTS", "03_cross_dataset_ctt"),
]

# Workspace paths to move into archive (superseded by consolidated bundle)
ARCHIVE_MOVES: list[str] = [
    "new_experiments/can_train_and_test_cross_dataset_validation",
    "outputs",
]

DRAFT_TABLE_MAP = """# Draft table → canonical source mapping

| Draft table | Metric (example) | Draft value | Canonical source | Canonical value | Action |
|-------------|------------------|-------------|------------------|-----------------|--------|
| Table I | Pooled local F1 | 0.868 | `01_primary_ocslab_balanced/tables/table_P4_vehicle_level_results.csv` | ~0.886 pooled | Align wording to balanced P4 |
| Table II | Bandwidth reduction | 53.69% | `table_P5_descriptor_compactness_and_privacy.csv` | verify from P5 | Use P5 |
| Table III | Unrelated incorrect merge | **0.400** | `table_P6_benign_isolated_unrelated_results.csv` | **0.400** | **Keep** (matches balanced) |
| Table IV | Strong F1, cs=5 | **0.867** | `table_P7_strong_campaign_results.csv` | **0.733** | **Update draft** (was original split) |
| Table IV | Strong F1, cs=10 | **0.933** | `table_P7` | **1.000** | **Update draft** |
| Table VI | Weak F1, cs=5 | 0.533 | `table_P8_weak_campaign_results.csv` | 0.500 | Minor update |
| Baseline ablation | M3 strong F1 | 0.406 (old) | `02_baseline_ablation/` (balanced M3) | 0.733 | Use new bundle |
| Baseline ablation | M3 unrelated merge | 0.000 (old) | `02_baseline_ablation/` | 0.400 | Use new bundle |

See `01_primary_ocslab_balanced/audit/original_vs_balanced_split.md` for original→balanced diffs.
"""


@dataclass
class ManifestRow:
    dest_path: str
    source_type: str
    source_ref: str
    copied: bool
    notes: str = ""


def git_show_bytes(ref: str, repo_path: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{repo_path}"],
        cwd=REPO,
        capture_output=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def copy_from_git(ref: str, repo_path: str, dest: Path) -> bool:
    data = git_show_bytes(ref, repo_path)
    if data is None:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def copy_tree(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    return True


def archive_path(src_rel: str) -> Path:
    return ARCHIVE / TODAY / src_rel


def main() -> int:
    subprocess.run(["git", "fetch", "origin", "cursor/campaign-clustering"], cwd=REPO, check=False)

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    manifest: list[ManifestRow] = []

    for ref, repo_path, dest_rel in GIT_COPIES:
        dest = OUT / dest_rel
        ok = copy_from_git(ref, repo_path, dest)
        manifest.append(ManifestRow(dest_rel, "git", f"{ref}:{repo_path}", ok, "" if ok else "MISSING"))

    for src_rel, dest_rel in LOCAL_COPIES:
        src = REPO / src_rel
        dest = OUT / dest_rel
        ok = copy_tree(src, dest)
        manifest.append(ManifestRow(dest_rel, "local", src_rel, ok, "" if ok else "MISSING"))

    (OUT / "DRAFT_TABLE_MAPPING.md").write_text(DRAFT_TABLE_MAP, encoding="utf-8")
    (OUT / "README.md").write_text(
        f"""# Experimental results bundle ({TODAY})

Canonical artifacts for the FLEET-GUARD paper. **Do not cite superseded folders in `archive/`.**

## Layout

| Folder | Contents | Use in paper |
|--------|----------|--------------|
| `01_primary_ocslab_balanced/` | Balanced end-to-end publication run (Tables P4–P12) | **Section VII primary OCSLab results** |
| `02_baseline_ablation/` | M1/M2/M3 ablation (see `RESULT_PROVENANCE.md`) | Baseline & ablation subsection |
| `03_cross_dataset_ctt/` | CTT cross-dataset Overleaf bundle | Cross-dataset validation section |
| `DRAFT_TABLE_MAPPING.md` | Draft Table I–IX → file mapping | Manuscript revision |

## Rebuild

```bash
python scripts/build_baseline_ablation_comparison.py
python scripts/consolidate_experimental_results.py
```

## Authoritative vs draft

- Draft Table III (unrelated merge **0.400**) matches the balanced publication run.
- Draft Table IV strong cs=5 F1 (**0.867**) is from the **superseded original split**; update to **0.733** from `table_P7`.
""",
        encoding="utf-8",
    )

    # Archive superseded local folders
    archived: list[str] = []
    for src_rel in ARCHIVE_MOVES:
        src = REPO / src_rel
        if not src.exists():
            continue
        dest = archive_path(src_rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(src), str(dest))
        archived.append(src_rel)
        manifest.append(ManifestRow(f"archive/{TODAY}/{src_rel}", "archived", src_rel, True, "moved to archive"))

    # Write manifests
    manifest_path = OUT / "MANIFEST.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dest_path", "source_type", "source_ref", "copied", "notes"])
        for row in manifest:
            w.writerow([row.dest_path, row.source_type, row.source_ref, row.copied, row.notes])

    (ARCHIVE / TODAY / "ARCHIVE_README.md").write_text(
        f"""# Archived superseded experiment outputs ({TODAY})

These workspace copies were moved here because canonical results now live in
`experimental-{TODAY}/`.

| Archived path | Superseded by |
|---------------|---------------|
| `can_train_and_test_cross_dataset_validation/` | `experimental-{TODAY}/03_cross_dataset_ctt/` (corrected Overleaf bundle) |
| `outputs/` | `01_primary_ocslab_balanced/` and git branch artifacts |

Original git branches are unchanged; restore from git history if needed.
""",
        encoding="utf-8",
    )

    # Regenerate P4 with canonical balanced F1 (overwrites git-copied legacy figure)
    p4_script = REPO / "scripts" / "build_figure_P4_strong_vs_weak_campaign_f1.py"
    if p4_script.is_file():
        import subprocess as sp

        sp.run(["python3", str(p4_script)], cwd=REPO, check=True)
        manifest.append(
            ManifestRow(
                "01_primary_ocslab_balanced/figures/figure_P4_strong_vs_weak_campaign_F1.pdf",
                "generated",
                str(p4_script),
                True,
                "canonical balanced F1 from campaign_metrics.csv",
            )
        )

    missing = [r for r in manifest if not r.copied and r.source_type != "archived"]
    print(f"Bundle: {OUT}")
    print(f"Archived: {archived}")
    if missing:
        print("MISSING:")
        for r in missing:
            print(f"  - {r.source_ref}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
