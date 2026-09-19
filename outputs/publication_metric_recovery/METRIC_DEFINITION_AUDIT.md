# METRIC_DEFINITION_AUDIT.md

**Scope:** Authoritative balanced OCSLab publication P7/P8 metric field definitions.  
**Artifacts:** `experimental-2026-06-23/01_primary_ocslab_balanced/`  
**Master config hash:** `72dbfc1760126e799a8a1670f1aa81391bd69a65edeab7beecb61846aa946d9e`

Evidence grades used below: `VERIFIED_FROM_EXECUTED_CODE` | `VERIFIED_FROM_SOURCE` | `VERIFIED_FROM_PUBLICATION_ARTIFACT` | `INFERRED` | `UNRECOVERABLE`.

---

## 1. Call chain that produced P7/P8 (orchestration only)

```
scripts/run_balanced_publication_run.py
  → final_end_to_end_publication_run_balanced.runner.run_balanced_publication
       → for each run: _run_single_test(..., shared=shared)     [MISSING MODULE]
            └─ historically run_refinement_fcgnn                 [MISSING MODULE]
       → rows collected → written as results/campaign_metrics.csv (same all_df)
       → generate_tables (balanced wrapper → base generate_tables) [BASE MISSING]
            → tables/table_P7_strong_campaign_results.csv
            → tables/table_P8_weak_campaign_results.csv
```

**Source:** `recovered_publication_pipeline/src/experiments/final_end_to_end_publication_run_balanced/runner.py` (imports + write sites).  
**Grade:** `VERIFIED_FROM_SOURCE` for the call graph; **`UNRECOVERABLE`** for the functions that *computed* the metric fields.

The recovered balanced `tables.py` only wraps missing base `generate_tables`. It does **not** compute campaign metrics.

Named missing metric emitter (import site only):

| Symbol | Declared module | Role |
|--------|-----------------|------|
| `extract_run_metrics` | `src.experiments.final_shared_configuration.metrics` | Per-run metric extraction |
| `safety_row` | same | Safety/P6 row builder |
| `run_refinement_fcgnn` | `src.experiments.coordinated_campaign_refinement.refinement_pipeline` | Per-scenario FCGNN pipeline |
| `_run_single_test` | `src.experiments.final_end_to_end_publication_run.runner` | Single-test orchestration |

**Grade:** `VERIFIED_FROM_SOURCE` that these are the intended emitters; **`UNRECOVERABLE`** as git blobs (confirmed `MISSING_MODULES.md`, `RECOVERY_REPORT.md`, full-history search).

---

## 2. How P7/P8 relate to `campaign_metrics.csv`

For strong/weak rows, grouping `campaign_metrics.csv` by `campaign_size` and taking the numeric mean **exactly reproduces** `table_P7` / `table_P8`, including:

- Strong F1: `0.533… / 0.733… / 1.0`
- Weak F1: `0.0666… / 0.5 / 0.7166…`

**Grade:** `VERIFIED_FROM_PUBLICATION_ARTIFACT` (+ verification script output).  
**Implication:** recovering field semantics on `campaign_metrics.csv` *is* recovering P7/P8 semantics at the aggregate layer. It does **not** recover the per-run computation code.

---

## 3. Per-field status

| Field | Recovered definition? | Best evidence | Grade |
|-------|----------------------|---------------|-------|
| `campaign_precision` | Algebraic identity on saved rows only | `TP/n_pred` with `TP=min(n_pred,n_true)` | Artifact identity `VERIFIED`; emitter `UNRECOVERABLE` |
| `campaign_recall` | Same | `TP/n_true`; equals `campaign_detection_rate` and `completeness` | same |
| `campaign_f1` | Same | `2PR/(P+R)` (0 if P=R=0) | same |
| `false_campaign_cluster_count` | Same | `max(n_pred−n_true, 0)` | same |
| `missed_campaign_count` | Same | `max(n_true−n_pred, 0)` | same |
| `membership_precision` | Vehicle-level identity | `attacked_ok / predicted_campaign_size` | Artifact `VERIFIED`; code `UNRECOVERABLE` |
| `membership_recall` | Vehicle-level identity | `attacked_ok / true_campaign_size` | same |
| `membership_f1` | Derived | `2PR/(P+R)` on membership | Artifact `VERIFIED` |
| `membership_purity` | Alias | **Identical to** `membership_precision` on all strong/weak rows | Artifact `VERIFIED` |
| `completeness` | Alias of campaign recall | **Identical to** `campaign_recall` (not membership recall) | Artifact `VERIFIED` |
| `fragmentation_rate` | Binary identity | `1` iff `n_pred > n_true` else `0` | Artifact `VERIFIED` |
| `fragments_per_true_campaign` | Count identity | **Equals** `n_predicted_campaign_clusters` | Artifact `VERIFIED` |
| `incorrect_merging_rate` | Scenario binary mean | Unrelated seeds: `1` when a merge event recorded; mean `0.4` = P6 | Artifact `VERIFIED`; predicate code `UNRECOVERABLE` |

Closest committed **peer** (not wired to balanced runner):  
`61a8203:src/experiments/model_diversity_final_tuned/false_campaign_metrics.py` uses the same *count-based* campaign TP/FP/FN pattern and vehicle membership contamination.  
**Grade for “this is extract_run_metrics”:** `INFERRED` only — **not** `VERIFIED_FROM_SOURCE` for P7/P8.

---

## 4. What was searched and not found

- All local + `origin/*` branches, commit `61a82038`, recover-publication branch
- `git rev-list --objects --all`, path history for `*refinement*`, `*final_shared*`, `*extract_run*`
- GitHub code search (rate-limited; prior recovery already reported never-committed)
- Notebooks, shell scripts, manifests, frozen run dumps under `results/scenario_evaluation/runs/` (absent)

**No blob defining `extract_run_metrics` or `run_refinement_fcgnn` was found.**

---

## 5. Conclusion for metric generation

**STOP for executable recovery.**  
We can state **artifact-level equations** that the saved P7/P8 numbers obey. We **cannot** point to the executed function that produced those columns for the balanced publication run.
