# OCSLab Candidate Selection Audit

**Date:** 2026-06-20  
**Branch inspected:** `origin/cursor/campaign-clustering`  
**Inspection root:** `tmp_ocslab_candidate_inspection/` (read-only restore; not committed)  
**Expected missing root:** `new_experiments/final_end_to_end_publication_run/`

No experiments were rerun. No result roots were modified. No symlinks were created.

---

## Executive summary

| Question | Answer |
|----------|--------|
| Does any listed candidate match **paper headline vehicle + descriptor numbers**? | **No** |
| Does any listed candidate match **paper scenario edge range (370–1311 on 200-node graphs)**? | **No** (see ancillary `final_publication_scenarios/`) |
| Is `final_end_to_end_publication_run/` present in git? | **No** (never committed) |
| Recommended official OCSLab root from candidates 1–3 | **D — sync original missing folder** |
| Safe to symlink a candidate to `final_end_to_end_publication_run/`? | **No** (would mislabel a different run and change paper numbers) |

Paper headline numbers (H1/H2) live in **`paper/results/`** on `origin/cursor/campaign-clustering` — a curated IEEE export bundle, **not** one of the three candidate experiment roots.

---

## Reference: known paper / project numbers

| Dimension | Paper reference |
|-----------|-----------------|
| ROC-AUC | ~0.7855 |
| PR-AUC | ~0.9273 |
| Precision | ~97.28% |
| Recall | ~45.96% |
| F1 | ~62.43% |
| FPR | ~3.95% |
| Raw window | ~2076 B |
| Descriptor | ~165 B (~49 B gzip) |
| Compression | ~12.6× |
| Bandwidth reduction | ~92% |
| Full graph (notes) | ~86,947 nodes; ~2,217,345 edges; cos θ≈0.85; k≈50 |
| Scenario package | benign false campaign = 0; isolated ≠ fleet campaign; edge sweep **370–1311** on fixed **200-node** graphs |

**Authoritative paper CSV sources (git):**

- `paper/results/table_01_vehicle_level_ids.csv`
- `paper/results/table_02_descriptor_compactness_security.csv`
- `results/descriptor_compactness_security_summary.md`
- `new_experiments/final_publication_scenarios/FINAL_SCENARIO_EXPERIMENT_SUMMARY.md` (scenario / edge package)

---

## Candidate 1 — `final_end_to_end_publication_run_balanced/`

| Property | Value |
|----------|-------|
| Path (git) | `new_experiments/final_end_to_end_publication_run_balanced/` |
| Files | 98 |
| Size | ~37 MB |
| Role | **Separate balanced-split E2E rerun** (Chevrolet validation fix); explicitly documented vs original split |

### Inventory

| Category | Present |
|----------|---------|
| Summary MD | `BALANCED_PUBLICATION_SUMMARY.md`, `audit/original_vs_balanced_split.md` |
| Tables | `table_P1`–`table_P12` (CSV/MD/TeX) |
| Figures | `figure_P1`–`figure_P10` (PNG/PDF) |
| Validation | `validation/balanced_publication_validation.md`, split validation, artifact completeness |
| Vehicle-level | `results/vehicle_level/`, `table_P4_vehicle_level_results.csv` |
| Descriptor | `results/descriptor_analysis/`, `table_P5_descriptor_compactness_and_privacy.csv` |
| Graph | `results/scenario_evaluation/graph_statistics.csv`, `table_P9`, `table_P10` |
| Scenarios | `table_P6`–`table_P8`, `results/scenario_evaluation/*` |
| Campaign-size | `results/campaign_size/`, `table_P9` |
| Edge sensitivity | `results/edge_sensitivity/`, `table_P10`–`P11` |

### Key numbers extracted

**Vehicle-level (pooled, `table_P4`):**

| Metric | Value | Paper | Match? |
|--------|------:|------:|:------:|
| ROC-AUC | 0.884 | 0.786 | ✗ |
| PR-AUC | 0.969 | 0.927 | ✗ |
| Precision | 91.1% | 97.3% | ✗ |
| Recall | 86.2% | 46.0% | ✗ |
| F1 | 88.6% | 62.4% | ✗ |
| FPR | 31.0% | 3.95% | ✗ |

**Descriptor (`table_P5`):**

| Metric | Value | Paper | Match? |
|--------|------:|------:|:------:|
| Raw window | 1600 B | 2076 B | ✗ |
| Descriptor | 741 B | 165 B | ✗ |
| Compression | 2.16× | 12.6× | ✗ |
| Bandwidth reduction | 53.7% | 92% | ✗ |

**Scenarios (`table_P6`):**

| Scenario | false_campaign | incorrect_merge |
|----------|---------------:|------------------:|
| Benign | 0.0 | 0.0 |
| Isolated | 0.0 | 0.0 |
| Unrelated | 0.0 | **0.4** |

**Strong / weak campaign F1 (`table_P7`/`P8`, size 10):** strong **1.0**; weak **0.717** (paper scenario package uses different evaluation grid).

**Campaign-size strong F1:** 0.533 (n=2) → 0.733 (n=5) → **1.0** (n=10) per `BALANCED_PUBLICATION_SUMMARY.md`.

**Edge sensitivity (`table_P10`, 200-node graphs):** unique edges **292–1886**; best campaign F1 up to **1.0** in dense bins — **not** the paper’s 370–1311 sweep.

### Paper match verdict

**Does not match** headline paper numbers. Documented as a **later alternative run** with different split, descriptor byte budget, and fleet configuration (`similarity_threshold` 0.95 vs paper graph notes).

---

## Candidate 2 — `publication_ready/`

| Property | Value |
|----------|-------|
| Path (git) | `new_experiments/publication_ready/` |
| Files | 47 |
| Size | ~1.3 MB |
| Role | **Curated scenario-method comparison tables** from validated Phase-2/3 runs (S0–S4, M1–M4) |

### Inventory

| Category | Present |
|----------|---------|
| Summary MD | *(none at root)* |
| Tables | `table_01`–`table_08` (experimental design, scenarios, safety, strong/weak, ablation, stats, edge) |
| Figures | `figure_01`–`figure_05` (+ edge subfigures) |
| Validation | `validation/publication_output_audit.md`, `publication_validation_report.md` |
| Data | `data/validated_metrics.csv` (~200 KB row-level) |
| Vehicle-level | **Absent** |
| Descriptor compactness | **Absent** |
| Full-scale graph (86k nodes) | **Absent** |

### Key numbers extracted (M4 Proposed FCGNN)

**Safety (`table_03`):**

| Scenario | false campaign alert | incorrect merge |
|----------|---------------------:|----------------:|
| S0 benign | 0.05 ± 0.16 | 0.0 |
| S1 isolated | 0.05 ± 0.16 | 0.0 |
| S2 unrelated (n=5) | 0.753 ± 0.06 | 0.0 |

**Strong campaign F1 (`table_04`, M4, size 10):** **0.417 ± 0.44**  
**Weak campaign F1 (`table_05`, M4, size 10):** **0.433 ± 0.50**

**Edge sensitivity (`table_08`, M4):** unique undirected edges **2422–7005** (200-node scenario graphs) — **outside** paper note **370–1311**.

### Paper match verdict

Contains **scenario framing** aligned with the paper’s S0–S4 narrative but **not** the headline H1/H2 metrics. Edge grid and aggregated campaign F1 **do not match** the final scenario package documented in `final_publication_scenarios/`. **Incomplete** as a full OCSLab primary root.

---

## Candidate 3 — `results/` + `tables/` (top-level)

| Property | Value |
|----------|-------|
| Path (git) | `new_experiments/results/`, `new_experiments/tables/` |
| Files | 25 |
| Size | ~672 KB |
| Role | **Early scenario experiment aggregates** (S0–S4 per-scenario means) |

### Inventory

| Category | Present |
|----------|---------|
| Summary MD | Per-scenario table MD only |
| Tables | `table_S0`–`table_S4` |
| Results | `summary_mean_std.csv`, `run_level_metrics.csv`, `statistical_tests.csv` per scenario |
| Figures / validation / descriptor / full graph | **Absent** |

### Key numbers (fcgnn, S3 strong, coordination 1.0, size 10)

| Metric | Approx. value |
|--------|-------------|
| Precision | 0.930 |
| ROC-AUC | 0.938 |
| Campaign F1 (mean) | low / variable by method row |
| S0 fcgnn false campaign alert | 0.95 ± 0.22 |

### Paper match verdict

**Raw intermediate aggregates** — no vehicle-level IF paper table, no descriptor security table, no end-to-end publication layout. **Not** the paper primary run.

---

## Ancillary finding (not in user list): `paper/results/`

Restored read-only from the same branch during audit.

| Property | Value |
|----------|-------|
| Files | ~15 CSV/MD under `paper/results/` |
| Role | **IEEE paper export bundle** — matches headline H1/H2/H4 tables |

### Key numbers — **exact paper match**

**`table_01_vehicle_level_ids.csv` (Isolation Forest):**

| ROC-AUC | PR-AUC | Precision | Recall | F1 | FPR |
|--------:|-------:|----------:|-------:|---:|----:|
| **0.7855** | **0.9273** | **97.28%** | **45.96%** | **62.43%** | **3.95%** |

**`table_02_descriptor_compactness_security.csv`:**

| Raw (B) | Descriptor (B) | Compression | BW reduction |
|--------:|---------------:|------------:|-------------:|
| **2075.75** | **165.08** | **12.57×** | **92.05%** |

**Fleet graph (`final_gnn_graph_statistics.csv`):** 77,233 nodes; 819,914 edges (same order of magnitude as paper notes; not identical to 86,947 / 2.2M).

**Limitation:** This is a **curated export directory**, not the full experiment root expected by `build_cross_dataset_comparison.py` (no `pooled/tables` CTT-like layout, no single scenario CSV set).

---

## Ancillary finding: `final_publication_scenarios/`

| Property | Value |
|----------|-------|
| Role | Final scenario / edge-sensitivity package |
| Edge range (fcgnn sweep) | **370 – 1311** unique edges ✓ matches paper note |
| S0 false campaign (C3) | **0.0** mean |
| S1 isolated | local recall 1.0; no fleet campaign |

**Does not contain** paper vehicle-level or descriptor headline tables.

---

## Cross-candidate comparison matrix

| Metric | Paper | Cand. 1 balanced | Cand. 2 pub_ready | Cand. 3 results | paper/results |
|--------|------:|-----------------:|------------------:|----------------:|--------------:|
| ROC-AUC | 0.786 | 0.884 | — | ~0.94 (S3 local) | **0.786** ✓ |
| F1 | 62.4% | 88.6% | — | — | **62.4%** ✓ |
| Descriptor B | 165 | 741 | — | — | **165** ✓ |
| BW reduction | 92% | 54% | — | — | **92%** ✓ |
| Edge range (200-node) | 370–1311 | 292–1886 | 2422–7005 | — | — |
| Full E2E tables P1–P12 | — | ✓ | partial | ✗ | partial |
| Scenario S0 false=0 | ✓ | ✓ | ≈ (0.05 mean) | ✗ | — |

---

## Recommendation

### **D — Do not use any of candidates 1–3 as the official OCSLab primary result**

**Reasoning:**

1. **`final_end_to_end_publication_run/` was never committed** to this repository. None of the three folders uses that name or reproduces it as a unified root.

2. **None of candidates 1–3 matches the paper headline numbers** used in Section VII (vehicle IDS + descriptor compactness). Candidate 1 (balanced) would **materially change** every headline: higher recall/FPR, lower compression, different unrelated-merge rate (0.4 vs paper scenario package claims).

3. **`publication_ready/`** is a **scenario table subset** only — useful supplementary material, wrong edge grid vs final scenario package, no local/descriptor artifacts.

4. **`results/` + `tables/`** are **early scenario summaries** — incomplete and numerically inconsistent with the paper.

5. The **actual paper numbers** are preserved in **`paper/results/`** plus scenario evidence in **`final_publication_scenarios/`**, but these are **derived/export layers**, not substitutes for the missing end-to-end run folder that the comparison builder expects.

### If you must pick among listed candidates only (not recommended)

| Goal | Least-bad choice | Caveat |
|------|------------------|--------|
| Most complete E2E artifact tree | **A — balanced** | Changes all headline numbers; different split |
| Scenario tables for methods paper | **B — publication_ready** | Missing H1/H2; wrong edge sweep |
| Nothing | **D — sync original** | Correct path |

### Using balanced run — impact on paper

Symlinking **balanced → `final_end_to_end_publication_run/`** would force the comparison builder to report:

- Pooled F1 **~88.6%** instead of **62.4%**
- Descriptor **741 B / 54%** reduction instead of **165 B / 92%**
- Unrelated incorrect merge **0.4** instead of values from the original scenario package
- Strong/weak curves from **P7/P8** not matching `paper/results/` narrative

**Conclusion:** Using balanced as official primary **would change the paper numbers** and should not be done without rewriting the manuscript.

---

## Symlink safety assessment

| Action | Safe? |
|--------|:-----:|
| Symlink candidate 1 → `final_end_to_end_publication_run/` | **No** — mislabels a different experiment; breaks paper fidelity |
| Symlink candidate 2 → expected root | **No** — incomplete; wrong layout for comparison script |
| Symlink candidate 3 → expected root | **No** — incomplete |
| Sync original folder from OneDrive `CodeRepo/new_experiments/final_end_to_end_publication_run/` | **Yes** — intended solution |
| Assemble symlink from `paper/results/` + `final_publication_scenarios/` | **No** — fragmented exports; not a single run root |

---

## Next step (user action)

From local OneDrive machine:

```text
CodeRepo/new_experiments/final_end_to_end_publication_run/
```

Copy or sync into the workspace at the **same relative path**, then rerun **read-only**:

```bash
PYTHONPATH=. python3 scripts/build_cross_dataset_comparison.py
```

Until then, cross-dataset comparison OCSLab columns remain `SOURCE_NOT_IN_WORKSPACE`.

---

## Inspection provenance

| Candidate | Restore command |
|-----------|-----------------|
| 1 | `git archive origin/cursor/campaign-clustering new_experiments/final_end_to_end_publication_run_balanced` |
| 2 | `git archive … new_experiments/publication_ready` |
| 3 | `git archive … new_experiments/results new_experiments/tables` |

Temporary files under `tmp_ocslab_candidate_inspection/` may be deleted; they are not part of the repository.
