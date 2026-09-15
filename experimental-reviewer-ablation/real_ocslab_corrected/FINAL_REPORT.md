# Final report — corrected reviewer ablation (STOPPED)

## Status

**Experiment not executed.** Manuscript-aligned Sonata/Soul/Spark car_track traces are absent from the workspace. PR #22 results remain diagnostic-only and must not appear in the paper.

---

### A. Dataset provenance

Target: `Dataset/ocslab_pipeline/{Hyundai,Kia,Chevrolet}/`.

- Hyundai: **empty**
- Kia: **empty**
- Chevrolet: 5 classic Car-Hacking files (byte-identical to `normal_run_data` / DoS / Fuzzy / gear / RPM) — **not** Chevrolet Spark car_track
- Publication manifests require 21 `*_HY_Sonata_*` / `*_KIA_Soul_*` / `*_CHEVROLET_Spark_*` basenames — **0 present locally**
- Exact Sonata/Soul/Spark identity: **cannot be established from local pipeline files**; confirmed only via recovered OneDrive manifest paths

### B. Train/validation/test split

**Not produced.** Planned: source-trace contiguous 70/15/15 with no overlapping-window leakage; `split_manifest.csv` deferred.

### C. Exact campaign semantics recovered

**YES (audited).** Score-band + shared attack family + `compute_campaign_prototype` + `apply_coordination_strength(strength=1.0)`. See `CAMPAIGN_SEMANTICS_AUDIT.md`. PR #22 score-only grouping rejected.

### D. 24-D / 9-D feature path (planned)

`extract_window_features` → 24-D `BEHAVIOURAL_FEATURE_COLUMNS` → IF `anomaly_score` → derive 9-D `g_i` (`FEATURE_NAMES`). Not executed.

### E. Fleet scaler before cosine

Scaler verified: `fleet_benign_scaler.json` feature names/order **exact match** to 9-D `g_i`. Config requires apply-before-cosine; raw unscaled cosine forbidden. Application deferred pending TEST pool.

### F. Seed-11 cosine validation

**BLOCKED** — see `COSINE_VALIDATION.md`.

### G. M1–M4 definitions (frozen; not run)

| Method | Definition |
|--------|------------|
| M1 | Local IF only; campaign metrics N/A |
| M2 | Same nodes; descriptor → StandardScaler → PCA(8) → DBSCAN → gate |
| M3 | Publication-scaled similarity graph → 2-layer GCN (9→64→32) → StandardScaler → PCA(8) → DBSCAN → gate |
| M4 | Same graph → GraphSAGE 9→64→32, mean agg, ReLU, 30 ep, Adam 0.01, wd 5e-4, λ=0.25 → same clustering/gate |

Graph: τ=0.95, same-k=2, cross-k=5. Gate: β=0.5, fragment merge=0.85.

### H. Ten-seed results

**N/A — not run.**

### I. Independent-incident merge

**N/A — not run.**

### J. Metric definitions (unchanged from shared harness)

Matching: greedy one-to-one Jaccard ≥ 0.5.  
Campaign Precision = matched / #pred; Recall = matched / #GT; F1 = harmonic mean.  
Membership F1 = micro F1 over matched pair membership (unmatched GT→FN, unmatched pred→FP).  
Fragmentation = mean over GT of (#pred clusters with overlap).  
Incorrect merge rate / false campaign rate as in `experimental-reviewer-ablation/metrics.py`.

### K. Shared scenarios across variants

**N/A — not run.** Assertions planned: identical descriptor IDs, GT, and M3/M4 edges/features.

### L. Remaining differences vs historical publication

Even after data arrives: absolute Section VII numeric identity is not claimed (missing frozen descriptors/models; regenerated IF/windows). Methodology corrections target recoverable semantics only.

### M. Scientifically suitable for paper inclusion as reviewer ablation?

**NO — not yet.** Campaign semantics and scaler path are ready, but without Sonata/Soul/Spark inputs the corrected experiment cannot run. **PR #22 Campaign F1=0 must not be used in the paper.**

---

Required unblock: provide car_track CSVs, then re-run `check_dataset_gate.py` → pool → seed-11 cosine gate → full ten-seed M1–M4.
