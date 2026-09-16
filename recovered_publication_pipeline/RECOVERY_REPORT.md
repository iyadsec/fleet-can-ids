# Recovery report — authoritative balanced publication pipeline

## Classification: **C — NOT REPRODUCIBLE**

Exact historical **results and configs** for
`experimental-2026-06-23/01_primary_ocslab_balanced/`
(source: `new_experiments/final_end_to_end_publication_run_balanced/` on
`origin/cursor/campaign-clustering` @ `61a82038ffcac2066d57d524f9305be88dabe212`) are recoverable.

Exact historical **source** for the packages that *executed* that pipeline is **not**
recoverable from git, remotes, tags, reflog, stashes, or the workspace.

No single-seed or ten-seed reproduction was attempted: dependency closure is incomplete,
and inventing the missing modules would violate the recovery rules.

---

## 1. Historical source recovery table

| File / module | Branch | Commit | Status |
|---------------|--------|--------|--------|
| `src/experiments/final_end_to_end_publication_run_balanced/*` | `origin/cursor/campaign-clustering` | `61a82038ffcac2066d57d524f9305be88dabe212` | **RECOVERED EXACTLY** |
| `scripts/{run,build,validate}_balanced_*.py` | same | same | **RECOVERED EXACTLY** |
| Frozen configs, manifests, scaler JSON, validation_scenarios CSVs | same | same | **RECOVERED EXACTLY** |
| Frozen aggregate results (campaign_metrics, P6–P8, strong/weak summaries) | same | same | **RECOVERED EXACTLY** (reference only) |
| Transitive present peers (fleet_scaler_loader, local_descriptor_normalisation, graph/GNN/eval helpers, …) | same | same | **RECOVERED EXACTLY** (see `PROVENANCE.md`) |
| `src/experiments/final_end_to_end_publication_run/*` | — | — | **MISSING** (never committed) |
| `src/experiments/final_shared_configuration/*` | — | — | **MISSING** (never committed) |
| `src/experiments/coordinated_campaign_refinement/*` | — | — | **MISSING** (never committed) |
| `src/experiments/strong_campaign_extended/*` | — | — | **MISSING** (never committed) |
| Balanced descriptors / IF models / `processed/window_features.csv` | — | — | **MISSING** (never committed) |
| Non-balanced `new_experiments/final_end_to_end_publication_run/` tree | — | — | **MISSING** (never committed) |

See also: `MISSING_MODULES.md`, `PROVENANCE.md`.

---

## 2. Dependency tree

See `DEPENDENCY_TREE.md`.

Summary: balanced `runner.py` recovers cleanly, but immediately imports fifteen modules
that do not exist in any git object. Closest committed relatives
(`model_diversity_final_tuned`, `method_fcgnn`, generic `scenario_registry`) are
**not** semantically equivalent and were not substituted.

---

## 3. Configuration comparison

See `CONFIG_COMPARISON.md`.

Frozen YAML agrees with the audit checklist for fleet size, windows, seeds, campaign
sizes, graph caps, DBSCAN 0.5/2, cohesion 0.5, fragment centroid 0.85, GraphSAGE hidden
64 / lr 0.01 / epochs 30. λ, weight decay, and PCA wiring are not fully spelled in the
master YAML and depend on missing packages. Eval-code DBSCAN **defaults** remain 1.2/10
and must have been overridden by the missing `SharedFleetConfiguration` consumer.

---

## 4. Single-seed reproduction comparison

**Not run.** Reason: `ImportError` on missing packages; also missing
descriptors/models/window features and raw OCSLab path.

---

## 5. Ten-seed aggregate comparison

**Not run.** Frozen reference aggregates were copied for documentation only under
`frozen_results_reference/` and match the published summary numbers; that is **not** a
reproduction.

---

## 6. Final classification

### **C. NOT REPRODUCIBLE**

Required executable code and intermediate data remain missing. Frozen outputs alone do
not constitute a runnable pipeline.

---

## 7. Recommendation on controlled M1/M2/M3/M4 reviewer ablation

**Do not run the reviewer ablation against the publication pipeline yet.**

A controlled ablation requires the same executable path that produced
`01_primary_ocslab_balanced`. That path is still incomplete.

Safe next options (require your explicit choice):

1. **Recover from outside git** — author machine / OneDrive / Cursor agent transcript
   archives that still contain the four missing packages + balanced descriptors/models.
2. **Option B** — approve a reduced protocol on one *currently runnable* committed stack,
   with provenance stating results are **not** bit-comparable to Section VII.
3. **Stop** — keep frozen Section VII tables; do not claim a new component ablation.

No ablation code was implemented in this recovery pass. Current `src/` production files
were not overwritten.

Working branch: `cursor/recover-publication-pipeline-8f28`
