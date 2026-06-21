# Overleaf Cross-Dataset Validation Artifacts

## Purpose

This folder contains **copies** of all corrected figures and tables for the cross-dataset validation section (OCSLab vs can-train-and-test). Upload directly to Overleaf. Original experiment outputs are unchanged.

## Folder structure

```
OVERLEAF_CROSS_DATASET_ARTIFACTS/
├── figures_pdf/      ← upload these for LaTeX \includegraphics
├── figures_png/      ← preview / Word
├── tables_tex/       ← \input{} in LaTeX
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
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/figure_LOCAL_COMP1_pooled_comparison.pdf}
  \caption{Pooled local IDS comparison under FPR$\leq$5\%.}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.75\linewidth]{figures/figure_FLEET_CORR2_unrelated_merge_before_after.pdf}
  \caption{Unrelated incident merge rate before and after the campaign consistency rule.}
\end{figure}

\input{tables/FLEET_CORR1_corrected_ctt_fleet_summary.tex}
```

See `ARTIFACT_INDEX.csv` for full file list and `include_in_main_paper` column (YES / SUPPLEMENT / NO).
