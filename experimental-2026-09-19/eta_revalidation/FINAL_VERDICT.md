# FINAL VERDICT

# **STOP_NO_VALIDATION_SPLIT**

η was **not** selected. The primary OCSLab final evaluation was **not** run. CTT was **not** run. Historical P7/P8 artifacts were **not** modified.

---

## Primary blocker

The repository does **not** contain a **runnable** primary OCSLab **fleet-level validation split** suitable for selecting η under the prescribed rule.

### What exists

| Artifact | Sufficiency |
|----------|-------------|
| `…/validation_scenarios/validation_manifest.csv` | Metadata for historical V0–V4 validation runs (seeds, segment IDs, hashes). Mostly **empty** `event_ids`. **Not** executable scenario input. |
| `parameter_search.csv` / `selection_report.md` | Diagnostics from the historical joint search that froze graph/DBSCAN/cohesion — **not** a replayable validation dataset for η. |
| `model_diversity_final_tuned/validation_scenarios.py` | Peer builder requiring `descriptors[split==validation]` — descriptors **absent**. |
| `build_mixed_validation_suite` | **Never committed** (`MISSING_MODULES.md`). |

### What is missing

1. Balanced publication descriptors / `window_features.csv` / IF models for validation segments.  
2. Per-scenario validation descriptor tables referenced by the manifest.  
3. OCSLab raw traces in this environment (`Dataset/` has no usable traces).  
4. The missing shared-configuration validation suite consumer.

### Forbidden substitutions (not performed)

- Final held-out OCSLab **test** partition as a stand-in for validation.  
- CTT for η selection.  
- Optimising η to reproduce historical Strong/Weak F1 `0.533/0.733/1.000` and `0.067/0.500/0.717`.

---

## Concurrent independent blocker

### **STOP_METRIC_IMPLEMENTATION_UNRECOVERED**

Even with validation data restored, a defensible Campaign F1 / Membership F1 / fragmentation recomputation for this methodology still requires the unrecovered publication matcher/emitter (`extract_run_metrics` / refinement pipeline).  

Ablation greedy Jaccard≥0.5 is **not** proven identical to P7/P8 and **must not** be silently used to complete this experiment. No replacement metric was invented.

---

## What was completed despite STOP

1. New experiment directory `experimental-2026-09-19/eta_revalidation/` (historical June tree untouched).  
2. Independent `|C_k| >= η` gate retained/confirmed in `publication_fleet_core` (`min_campaign_cluster_size`; not coupled to DBSCAN `min_samples=2`).  
3. Unit tests for η↔DBSCAN independence, post-clustering application, `|C_k|` vs `r_k`, and no GT leakage.  
4. Predeclared candidate set `{2,3,5,10}` **before** evaluation (`eta_candidate_set_predeclared.json`).  
5. Recovered non-η fleet freeze recorded (`frozen_fleet_configuration.json`) with η unset.  
6. Selection explicitly **not** performed (`eta_selection.json`).

---

## What would flip this to `GO_ETA_FROZEN`

1. Restore runnable primary OCSLab fleet **validation** scenarios (no test leakage).  
2. Recover or separately ratify an explicit evaluation protocol (matcher/membership/fragmentation) without pretending it is the historical emitter.  
3. Run validation-only η selection on `{2,3,5,10}` with the predeclared rule.  
4. Write `eta_selection.json` with `selected_eta`, then freeze.  
5. Only then run final OCSLab under `final/` and compute η decision impact.  
6. Keep CTT unseen until the full freeze exists.

---

## Verdict code

```text
STOP_NO_VALIDATION_SPLIT
```

(Concurrent: `STOP_METRIC_IMPLEMENTATION_UNRECOVERED`)
