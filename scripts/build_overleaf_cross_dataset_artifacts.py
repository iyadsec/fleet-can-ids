#!/usr/bin/env python3
"""Copy corrected cross-dataset artifacts into OVERLEAF_CROSS_DATASET_ARTIFACTS/."""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "OVERLEAF_CROSS_DATASET_ARTIFACTS"

# git ref -> list of source directory prefixes to scan
SOURCE_ROOTS: dict[str, list[str]] = {
    "origin/cursor/local-ids-metric-comparison-8f28": [
        "new_experiments/local_ids_ocslab_vs_ctt_metric_comparison",
    ],
    "origin/cursor/corrected-ctt-publication-8f28": [
        "new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt",
        "new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected",
    ],
    "origin/cursor/fleet-corrected-comparison-summary-8f28": [
        "new_experiments/fleet_level_corrected_comparison_summary",
    ],
    "origin/cursor/ctt-f1-merge-diagnostics-8f28": [
        "new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge",
    ],
}

# Explicit required copies: (source_prefix, relative_path, dest_subfolder)
REQUIRED = [
    # Local IDS tables
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/tables", "LOCAL_COMP1_pooled_ocslab_vs_ctt", "table"),
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/tables", "LOCAL_COMP2_per_vehicle_ocslab_vs_ctt", "table"),
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/tables", "LOCAL_COMP3_ctt_by_subset", "table"),
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/tables", "LOCAL_COMP4_ctt_by_attack_type", "table"),
    # Local IDS figures
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/figures", "figure_LOCAL_COMP1_pooled_comparison", "figure"),
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/figures", "figure_LOCAL_COMP2_per_vehicle_pr_auc", "figure"),
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/figures", "figure_LOCAL_COMP3_per_vehicle_prf", "figure"),
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/figures", "figure_LOCAL_COMP4_per_vehicle_fpr", "figure"),
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/figures", "figure_LOCAL_COMP5_ctt_by_subset", "figure"),
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison/figures", "figure_LOCAL_COMP6_ctt_by_attack_f1", "figure"),
    ("new_experiments/local_ids_ocslab_vs_ctt_metric_comparison", "LOCAL_IDS_COMPARISON_INTERPRETATION.md", "report"),
    # CTT corrected
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/tables", "CTT_CORR1_local_threshold_policy_comparison", "table"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/tables", "CTT_CORR2_local_by_subset", "table"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/tables", "CTT_CORR3_local_by_attack_type", "table"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/tables", "CTT_CORR4_local_by_vehicle", "table"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/tables", "CTT_CORR5_campaign_consistency_ablation", "table"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/tables", "CTT_CORR6_corrected_scenario_results", "table"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/tables", "CTT_CORR7_corrected_edge_sensitivity", "table"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/figures", "figure_CTT_CORR1_f1_vs_threshold", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/figures", "figure_CTT_CORR2_precision_recall_vs_threshold", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/figures", "figure_CTT_CORR3_corrected_local_by_subset", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/figures", "figure_CTT_CORR4_corrected_scenario_outcomes", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/figures", "figure_CTT_CORR5_unrelated_merge_before_after_rule", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/figures", "figure_CTT_CORR6_edge_count_vs_campaign_f1", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt/figures", "figure_CTT_CORR7_edge_count_vs_incorrect_merge_rate", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt", "CTT_CORRECTED_PUBLICATION_SUMMARY.md", "report"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/corrected_publication_ctt", "LOCAL_THRESHOLD_POLICY_RECOMMENDATION.md", "report"),
    # Fleet
    ("new_experiments/fleet_level_corrected_comparison_summary/tables", "FLEET_CORR1_corrected_ctt_fleet_summary", "table"),
    ("new_experiments/fleet_level_corrected_comparison_summary/tables", "FLEET_CORR2_consistency_rule_ablation", "table"),
    ("new_experiments/fleet_level_corrected_comparison_summary/tables", "FLEET_CORR3_ocslab_vs_ctt_corrected_fleet_comparison", "table"),
    ("new_experiments/fleet_level_corrected_comparison_summary/figures", "figure_FLEET_CORR1_corrected_scenario_outcomes", "figure"),
    ("new_experiments/fleet_level_corrected_comparison_summary/figures", "figure_FLEET_CORR2_unrelated_merge_before_after", "figure"),
    ("new_experiments/fleet_level_corrected_comparison_summary/figures", "figure_FLEET_CORR3_consistency_rule_ablation", "figure"),
    ("new_experiments/fleet_level_corrected_comparison_summary/figures", "figure_FLEET_CORR4_ocslab_vs_corrected_ctt_scenario_comparison", "figure"),
    ("new_experiments/fleet_level_corrected_comparison_summary/figures", "figure_FLEET_CORR5_edge_count_vs_campaign_f1", "figure"),
    ("new_experiments/fleet_level_corrected_comparison_summary/figures", "figure_FLEET_CORR6_edge_count_vs_unrelated_merge", "figure"),
    ("new_experiments/fleet_level_corrected_comparison_summary", "FLEET_CORRECTED_COMPARISON_SUMMARY.md", "report"),
    ("new_experiments/fleet_level_corrected_comparison_summary", "FLEET_CORRECTED_COMPARISON_PAPER_WORDING.md", "report"),
    # Cross-dataset corrected (priority)
    ("new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/tables", "table_CUR_COMP3_fleet_scenario_comparison", "table"),
    ("new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/tables", "table_CUR_COMP1_dataset_role_coverage", "table"),
    ("new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/tables", "table_CUR_COMP5_combined_evidence_limitations", "table"),
    ("new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/tables", "table_CUR_COMP2_headline_local_descriptor", "table"),
    ("new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/tables", "table_CUR_COMP4_fleet_graph_edge_comparison", "table"),
    ("new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/figures", "figure_CUR_COMP3_scenario_outcomes", "figure"),
    ("new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/figures", "figure_CUR_COMP1_dataset_coverage", "figure"),
    ("new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/figures", "figure_CUR_COMP2_descriptor_bandwidth", "figure"),
    ("new_experiments/cross_dataset_comparison_ocslab_vs_ctt_corrected/audit", "recommended_curated_comparison_wording.md", "report"),
    # Diagnostics (selected)
    ("new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge/figures", "local_f1_vs_threshold", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge/figures", "fpr_tpr_vs_threshold", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge/figures", "unrelated_merge_vs_edge_threshold", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge/figures", "campaign_f1_vs_graph_edge_count", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge/figures", "incorrect_merge_vs_graph_edge_count", "figure"),
    ("new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge", "CTT_F1_AND_MERGE_DIAGNOSTIC_SUMMARY.md", "report"),
]

REF_FOR_PATH: dict[str, str] = {}
for ref, roots in SOURCE_ROOTS.items():
    for root in roots:
        REF_FOR_PATH[root] = ref


@dataclass
class CopyRow:
    copied_file: str
    original_source_path: str
    file_type: str
    copied_successfully: bool
    notes: str = ""


def git_show(ref: str, repo_path: str) -> bytes | None:
    proc = subprocess.run(
        ["git", "show", f"{ref}:{repo_path}"],
        cwd=REPO,
        capture_output=True,
    )
    return proc.stdout if proc.returncode == 0 else None


def resolve_source(full_path: str) -> tuple[Path | None, str | None]:
    """Return (local Path, git_ref) for a repo-relative path."""
    local = REPO / full_path
    if local.is_file():
        return local, None
    prefix = full_path.split("/")[0]
    for root, ref in REF_FOR_PATH.items():
        if full_path.startswith(root + "/") or full_path == root:
            return None, ref
    # walk prefixes
    for root, ref in REF_FOR_PATH.items():
        if full_path.startswith(root):
            return None, ref
    return None, None


def copy_file(repo_rel: str, dest: Path, ref: str | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    local = REPO / repo_rel
    if local.is_file():
        shutil.copy2(local, dest)
        return True
    if ref:
        data = git_show(ref, repo_rel)
        if data is not None:
            dest.write_bytes(data)
            return True
    # try any ref
    for r in SOURCE_ROOTS:
        data = git_show(r, repo_rel)
        if data is not None:
            dest.write_bytes(data)
            return True
    return False


def ref_for(repo_rel: str) -> str | None:
    for root, ref in REF_FOR_PATH.items():
        if repo_rel.startswith(root):
            return ref
    for ref in SOURCE_ROOTS:
        if git_show(ref, repo_rel) is not None:
            return ref
    return None


def dest_folder(kind: str, ext: str) -> Path:
    if kind == "figure":
        return OUT / ("figures_pdf" if ext == ".pdf" else "figures_png")
    if kind == "table":
        if ext == ".tex":
            return OUT / "tables_tex"
        if ext == ".csv":
            return OUT / "tables_csv"
        return OUT / "tables_md"
    return OUT / "reports"


def main() -> int:
    subprocess.run(["git", "fetch", "origin"], cwd=REPO, capture_output=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    for sub in ("figures_pdf", "figures_png", "tables_tex", "tables_csv", "tables_md", "reports", "source_manifest"):
        (OUT / sub).mkdir(parents=True)

    manifest: list[CopyRow] = []
    missing: list[str] = []

    for dir_prefix, base_name, kind in REQUIRED:
        exts = [".pdf", ".png"] if kind == "figure" else [".tex", ".csv", ".md"] if kind == "table" else [""]
        for ext in exts:
            if kind == "report":
                repo_rel = f"{dir_prefix}/{base_name}"
            elif kind == "figure":
                repo_rel = f"{dir_prefix}/{base_name}{ext}"
            else:
                repo_rel = f"{dir_prefix}/{base_name}{ext}"

            ref = ref_for(repo_rel)
            dest = dest_folder(kind, ext) / (Path(base_name).name + ext if ext else base_name)
            ok = copy_file(repo_rel, dest, ref)
            manifest.append(
                CopyRow(
                    copied_file=str(dest.relative_to(OUT)),
                    original_source_path=repo_rel,
                    file_type=ext.lstrip(".") or "md",
                    copied_successfully=ok,
                    notes="" if ok else "not found",
                )
            )
            if not ok and kind != "table":  # tables may miss tex on some branches
                missing.append(repo_rel)
            elif not ok and ext == ".csv":
                missing.append(repo_rel)

    # Write source manifest
    sm_path = OUT / "source_manifest" / "source_manifest.csv"
    with sm_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["copied_file", "original_source_path", "file_type", "copied_successfully", "notes"])
        w.writeheader()
        for row in manifest:
            w.writerow(row.__dict__)

    # Artifact index
    index_rows = build_artifact_index(manifest)
    with (OUT / "ARTIFACT_INDEX.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "file_name", "artifact_type", "topic", "recommended_use",
                "source_path", "overleaf_folder", "caption_suggestion", "include_in_main_paper",
            ],
        )
        w.writeheader()
        w.writerows(index_rows)

    write_readme()

    # Validate counts
    counts = {
        "figures_pdf": len(list((OUT / "figures_pdf").glob("*.pdf"))),
        "figures_png": len(list((OUT / "figures_png").glob("*.png"))),
        "tables_tex": len(list((OUT / "tables_tex").glob("*.tex"))),
        "tables_csv": len(list((OUT / "tables_csv").glob("*.csv"))),
        "tables_md": len(list((OUT / "tables_md").glob("*.md"))),
        "reports": len(list((OUT / "reports").glob("*"))),
    }
    total = sum(counts.values()) + 2  # README + ARTIFACT_INDEX

    print("VALIDATION")
    print(f"  output exists: {OUT.exists()}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  README.md: {(OUT / 'README.md').exists()}")
    print(f"  ARTIFACT_INDEX.csv: {(OUT / 'ARTIFACT_INDEX.csv').exists()}")
    print(f"  source_manifest.csv: {sm_path.exists()}")
    print(f"  total copied (+readme/index): {total}")
    print(f"\nABSOLUTE_PATH={OUT.resolve()}")
    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for m in missing[:20]:
            print(f"  - {m}")

    # open in Finder (Mac); no-op on Linux
    subprocess.run(["open", str(OUT.resolve())], capture_output=True)
    return 0


def build_artifact_index(manifest: list[CopyRow]) -> list[dict]:
    yes_files = {
        "LOCAL_COMP1_pooled_ocslab_vs_ctt.csv",
        "figure_LOCAL_COMP1_pooled_comparison.pdf",
        "FLEET_CORR1_corrected_ctt_fleet_summary.csv",
        "figure_FLEET_CORR2_unrelated_merge_before_after.pdf",
        "figure_CTT_CORR4_corrected_scenario_outcomes.pdf",
    }
    supplement_keywords = ["LOCAL_COMP2", "LOCAL_COMP3", "LOCAL_COMP4", "CTT_CORR2", "CTT_CORR3", "CTT_CORR4", "CTT_CORR5", "CTT_CORR7", "FLEET_CORR3", "diagnostic", "local_f1", "fpr_tpr", "incorrect_merge", "campaign_f1_vs"]

    rows = []
    for m in manifest:
        if not m.copied_successfully:
            continue
        fname = Path(m.copied_file).name
        if fname.endswith(".pdf"):
            atype = "figure_pdf"
        elif fname.endswith(".png"):
            atype = "figure_png"
        elif fname.endswith(".tex"):
            atype = "table_tex"
        elif fname.endswith(".csv"):
            atype = "table_csv"
        elif fname.endswith(".md"):
            atype = "report" if "reports" in m.copied_file else "table_md"
        else:
            atype = "other"

        topic = "local_ids" if "LOCAL" in fname or "CTT_CORR1" in fname else \
                "fleet" if "FLEET" in fname or "CTT_CORR4" in fname or "CTT_CORR5" in fname or "CTT_CORR6" in fname else \
                "cross_dataset" if "CUR_COMP" in fname else \
                "diagnostic" if any(x in fname for x in ("local_f1", "merge", "threshold", "campaign_f1")) else "ctt_corrected"

        if fname in yes_files or fname.replace(".png", ".pdf") in yes_files:
            incl = "YES"
        elif any(k in fname for k in supplement_keywords):
            incl = "SUPPLEMENT"
        elif fname.endswith(".md") and "WORDING" in fname or "SUMMARY" in fname or "INTERPRETATION" in fname:
            incl = "SUPPLEMENT"
        else:
            incl = "NO"

        rows.append({
            "file_name": fname,
            "artifact_type": atype,
            "topic": topic,
            "recommended_use": "Main paper" if incl == "YES" else "Supplement" if incl == "SUPPLEMENT" else "Reference only",
            "source_path": m.original_source_path,
            "overleaf_folder": str(Path(m.copied_file).parent.name),
            "caption_suggestion": caption_for(fname),
            "include_in_main_paper": incl,
        })
    return rows


def caption_for(fname: str) -> str:
    caps = {
        "figure_LOCAL_COMP1_pooled_comparison": "Pooled local IDS comparison: OCSLab vs corrected CTT (FPR≤5%).",
        "figure_FLEET_CORR2_unrelated_merge_before_after": "Unrelated incident incorrect merge rate before and after campaign consistency rule.",
        "figure_CTT_CORR4_corrected_scenario_outcomes": "Corrected CTT fleet scenario outcomes (200-node graphs).",
        "FLEET_CORR1_corrected_ctt_fleet_summary": "Corrected CTT fleet scenario summary table.",
        "LOCAL_COMP1_pooled_ocslab_vs_ctt": "Pooled local IDS metrics: OCSLab vs corrected CTT.",
    }
    stem = Path(fname).stem
    for k, v in caps.items():
        if stem.startswith(k) or stem == k:
            return v
    return stem.replace("_", " ")


def write_readme() -> None:
    text = """# Overleaf Cross-Dataset Validation Artifacts

## Purpose

This folder contains **copies** of all corrected figures and tables for the cross-dataset validation section (OCSLab vs can-train-and-test). Upload directly to Overleaf. Original experiment outputs are unchanged.

## Folder structure

```
OVERLEAF_CROSS_DATASET_ARTIFACTS/
├── figures_pdf/      ← upload these for LaTeX \\includegraphics
├── figures_png/      ← preview / Word
├── tables_tex/       ← \\input{} in LaTeX
├── tables_csv/       ← data inspection
├── tables_md/        ← human-readable tables
├── reports/          ← summaries and paper wording
└── source_manifest/  ← traceability to original paths
```

## Upload to Overleaf first

1. `figures_pdf/figure_LOCAL_COMP1_pooled_comparison.pdf`
2. `figures_pdf/figure_FLEET_CORR2_unrelated_merge_before_after.pdf`
3. `figures_pdf/figure_CTT_CORR4_corrected_scenario_outcomes.pdf`
4. `tables_tex/LOCAL_COMP1_pooled_ocslab_vs_ctt.tex`
5. `tables_tex/FLEET_CORR1_corrected_ctt_fleet_summary.tex`

## Recommended main-paper tables

- `LOCAL_COMP1_pooled_ocslab_vs_ctt` — local IDS pooled comparison
- `FLEET_CORR1_corrected_ctt_fleet_summary` — corrected fleet scenarios
- `table_CUR_COMP3_fleet_scenario_comparison` — OCSLab vs CTT scenarios (supplement if space tight)

## Recommended main-paper figures

- `figure_LOCAL_COMP1_pooled_comparison`
- `figure_FLEET_CORR2_unrelated_merge_before_after`
- `figure_CTT_CORR4_corrected_scenario_outcomes`

## Supplementary

Per-vehicle, per-subset, per-attack local tables (LOCAL_COMP2–4), CTT_CORR2–7, FLEET_CORR3–6, diagnostic figures.

## Corrected CTT evaluation notes

- **Ground truth:** `eval_attack = (label==1) OR (attack_type!='benign')` — evaluation only
- **Local policy:** FPR ≤ 5% (F1-optimal diagnostic only)
- **Fleet graphs:** OCSLab-aligned 200-node scenario graphs
- **Consistency rule:** post-clustering; unrelated merge 1.0→0.0
- **No temporal edges**
- **Labels/attack types:** evaluation and diagnostics only, not model inputs

## LaTeX examples

```latex
\\begin{figure}[t]
  \\centering
  \\includegraphics[width=\\linewidth]{figures/figure_LOCAL_COMP1_pooled_comparison.pdf}
  \\caption{Pooled local IDS comparison under FPR$\\leq$5\\%.}
\\end{figure}

\\begin{figure}[t]
  \\centering
  \\includegraphics[width=0.75\\linewidth]{figures/figure_FLEET_CORR2_unrelated_merge_before_after.pdf}
  \\caption{Unrelated incident merge rate before and after the campaign consistency rule.}
\\end{figure}

\\input{tables/FLEET_CORR1_corrected_ctt_fleet_summary.tex}
```

See `ARTIFACT_INDEX.csv` for full file list and `include_in_main_paper` column (YES / SUPPLEMENT / NO).
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
