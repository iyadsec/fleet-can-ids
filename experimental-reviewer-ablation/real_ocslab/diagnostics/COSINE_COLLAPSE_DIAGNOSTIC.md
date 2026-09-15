# Cosine Collapse Diagnostic (PR #22 real_ocslab)

**Scope:** Diagnose why campaign–benign cosine ≈ 1.0.  
**Constraints honored:** no experiment edits, no τ retuning, no feature changes shipped into the ablation, no M1–M4 re-run.

**Artifacts:** `experimental-reviewer-ablation/real_ocslab/diagnostics/`
- `cosine_collapse_audit.json`
- `cosine_distributions_raw.csv`
- `cosine_distributions_transforms.csv`
- `feature_magnitudes_9d.csv` / `feature_magnitudes_24d.csv`
- `graph_density_seed11.csv`

---

## Executive answer

Cosine collapses because **PR #22 builds the similarity graph on raw unscaled 9-D GNN inputs `g_i`**, where two **constant** coordinates (`frame_count = message_rate = 100`) dominate ≈ **99.85%** of \(\|g_i\|_2^2\). Cosine’s per-vector L2 normalization does **not** fix per-dimension imbalance, so nearly all pairs (including campaign vs benign) have cosine ≥ 0.95.

The recoverable publication path applies a **benign-train fleet z-score scaler before cosine** (`behavior_only_vehicle_normalized`). That scaler zeros the constant dimensions (`std` floored to 1.0 with mean 100 → scaled value 0) and restores discrimination: campaign–benign pairs ≥ 0.95 drop from **100% → ~0.5%**.

Separately, PR #22 also uses **different vehicle data** than the manuscript’s Sonata / Soul / Spark platforms.

---

## 1. Similarity collapse quantification

Frozen scenarios over 10 seeds. Unrelated malicious = attack nodes from the same-seed `unrelated_incidents` scenario (cross-set pairs).

### 1a. Raw 9-D GNN input `g_i` (what PR #22 actually uses)

| Scenario | Pair set | mean | std | min | p25 | median | p75 | max | % ≥ 0.95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strong | A camp–camp | 0.999988 | 2.3e-5 | 0.999841 | 0.999988 | 0.999997 | 0.999999 | 1.000 | **100** |
| strong | B camp–benign | 0.999971 | 2.7e-5 | 0.999762 | 0.999961 | 0.999979 | 0.999989 | 1.000 | **100** |
| strong | C camp–unrelated | 0.999949 | — | 0.999654 | — | 0.999958 | — | — | **100** |
| strong | D benign–benign | 0.999988 | — | 0.999912 | — | 0.999990 | — | — | **100** |
| strong | E unrelated–unrelated | 0.999946 | — | 0.999676 | — | 0.999956 | — | — | **100** |
| weak | A–E (all buckets) | ≈1.0 | — | ≥0.9997 | — | ≈1.0 | — | — | **100** |

**Verdict:** on raw `g_i`, cosine carries **no class contrast**.

### 1b. Raw 24-D descriptor `d_i` (joined from pool; not the PR #22 graph input)

| Scenario | Pair set | mean | min | median | % ≥ 0.95 |
|---|---|---:|---:|---:|---:|
| strong | A camp–camp | 0.9909 | 0.9212 | 0.9937 | 99.32 |
| strong | B camp–benign | 0.9720 | 0.8779 | 0.9762 | **87.25** |
| strong | C camp–unrelated | 0.9516 | 0.8295 | 0.9503 | 50.14 |
| weak | B camp–benign | 0.9822 | 0.9425 | 0.9828 | **99.90** |

Raw 24-D is less totally collapsed than raw 9-D, but **still mostly above τ=0.95** for campaign–benign (especially weak).

---

## 2. Feature magnitudes (why cosine fails)

### 9-D `g_i` (all frozen strong/weak nodes)

| feature | min | max | mean | std | mean share of \(\|g\|^2\) |
|---|---:|---:|---:|---:|---:|
| **frame_count** | 100 | 100 | 100 | **0** | **0.4993** |
| **message_rate** | 100 | 100 | 100 | **0** | **0.4993** |
| can_id_entropy | 2.44 | 5.84 | 4.78 | 0.44 | 0.00115 |
| payload_entropy | — | — | — | — | ≪0.001 |
| anomaly_score | — | — | — | — | ≪0.001 |
| burstiness / timing / ratios | — | — | — | — | ≪0.001 |

**`frame_count` + `message_rate` ≈ 99.85% of squared norm.** They are constant 100 under window size 100. After cosine L2-normalization, every vector is nearly the same direction.

### 24-D `d_i`

Largest contributors are `frame_count` (~11% of \(\|d\|^2\)) and several `byte_std_*` (~7–9% each). No single pair reaches the 9-D domination level, which is why raw 24-D cosine is high but not identically 1.0.

---

## 3. Where cosine normalization occurs (PR #22 path)

Exact call chain used by the ablation:

```
scenario_builder_real.pack_frozen
  → X = df[FEATURE_NAMES]          # raw 9-D floats
run_real_ablation.build_shared_graph(frozen["X"], ...)
  → methods.build_shared_graph
    → build_scenario_graph_from_features(X, ...)
      → build_cross_vehicle_constrained_knn_edges(
            X, metric="cosine", similarity_threshold=0.95,
            top_k_same=2, top_k_cross=5)
```

Evidence: `experimental-reviewer-ablation/methods.py`, `src/graph/scenario_graph.py`.

### Explicit answer

Cosine is calculated on:

**A. raw unscaled feature values**

then sklearn/numpy cosine applies **per-vector L2 normalization only**.

Not B (standardized), not a fleet z-score, not a separate pre-L2 feature transform.

| Step | Present in PR #22? |
|---|---|
| Descriptor / 9-D features | Yes (raw) |
| Fleet / StandardScaler before similarity | **No** |
| Per-vector L2 inside cosine | Yes (implicit) |
| Threshold τ=0.95 | Yes |
| Constrained same/cross kNN | Yes |

---

## 4. Publication / recoverable graph feature path

Recoverable evidence (not assumption):

1. `configs/fleet_ids.yaml` sets  
   `similarity_feature_view: behavior_only_vehicle_normalized` (campaign graph block).
2. `src/graph/fleet_similarity_features.py` / recovered twin: for that view,  
   `fleet_scaler_provenance` is **required**; features pass through `apply_fleet_scaler` / local benign normalization **before** the cosine matrix is formed.
3. Default in `parse_fleet_graph_similarity_settings` is `behavior_only_vehicle_normalized`.
4. Comment in fleet graph builder: Euclidean path fits a StandardScaler; **cosine path does not auto-scale** — scaling is expected via the similarity **view**, not via the metric helper.

**Publication representation for cosine graph:** scaled behaviour features via benign fleet scaler (9-D behaviour view), **not** raw 24-D and **not** raw 9-D.

**PR #22 does not match that path.**

---

## 5. Recovered scaler artifact

Path:  
`recovered_publication_pipeline/new_experiments/final_end_to_end_publication_run_balanced/scalers/fleet_benign_scaler.json`

| Field | Value |
|---|---|
| Type | Benign-train **z-score** (means/stds; near-zero std floored to 1.0) |
| `#features` | **9** |
| Order | `anomaly_score, frame_count, message_rate, burstiness, mean_inter_arrival_time, std_inter_arrival_time, can_id_entropy, most_common_can_id_ratio, payload_entropy` |
| Fit rows | 11423 train benign (`attack_labels_used: false`) |
| Matches PR #22 `FEATURE_NAMES` | **Yes (exact)** |

Critical fitted stats:

- `frame_count`: mean=100, **std=1.0** (floored) → scaled value always 0  
- `message_rate`: mean=100, **std=1.0** → scaled value always 0  

So the historical scaler **removes the dominating constant dimensions** from cosine.

Loader/application: `recovered_publication_pipeline/src/experiments/local_descriptor_normalisation.py::apply_fleet_scaler` and `fleet_scaler_loader.py`.

**Yes — this scaler was intended to be applied before cosine graph construction in the publication similarity view.** Its absence in PR #22 is a primary mechanistic explanation of the collapse.

---

## 6. Controlled diagnostic transforms (no ablation re-run)

Representations on frozen test scenarios only:

| ID | Definition |
|---|---|
| **R0** | PR #22 raw 9-D |
| **R1** | Standardize with means/stds fit on **train benign only** from `all_scored_windows` (n=7215), applied to frozen scenarios |
| **R2** | Historical `fleet_benign_scaler.json` (exact feature match) |

### Strong campaigns (10 seeds) — 9-D

| Repr | within-campaign mean | % ≥ 0.95 | camp vs benign mean | % ≥ 0.95 | camp vs unrelated % ≥ 0.95 |
|---|---:|---:|---:|---:|---:|
| R0 | ≈1.000 | **100** | ≈1.000 | **100** | **100** |
| R1 | 0.739 | 39.5 | 0.120 | **0.20** | 14.9 |
| R2 | 0.788 | 47.2 | 0.145 | **0.48** | 18.4 |

### Weak campaigns — camp vs benign % ≥ 0.95

| R0 | R2 |
|---:|---:|
| 100 | **1.06** |

### 24-D diagnostic (R2 N/A — scaler is 9-D)

| Repr | strong camp–benign % ≥ 0.95 |
|---|---:|
| R0 raw 24-D | 87.25 |
| R1 train-benign scaler 24-D | **0.001** |

**Interpretation (diagnostic only):** missing standardization / fleet scaler is sufficient to explain the observed cosine collapse. This is **not** a claim that R1/R2 maximize Campaign F1.

---

## 7. Graph density consequence (seed 11; no GraphSAGE)

Same k-caps: same-vehicle ≤2, cross-vehicle ≤5, τ=0.95.

### Strong campaign

| | R0 raw | R2 historical scaler |
|---|---:|---:|
| All undirected pairs ≥ 0.95 | **100%** | **3.71%** |
| Retained edges after k-caps | 899 | 472 |
| Same-vehicle / cross-vehicle edges | 256 / 643 | 68 / 404 |
| Campaign–campaign edges | 170 | 191 |
| **Campaign–benign edges** | **122** | **38** |
| Campaign connected components | **1** | **5** |
| Largest campaign component | **50 / 50** | **39 / 50** |

### Weak campaign

| | R0 | R2 |
|---|---:|---:|
| Retained edges | 920 | 445 |
| Campaign–benign edges | 128 | 59 |
| Campaign components | 1 (size 50) | 5 (largest 31) |

**Conclusion:** under R0 the graph is saturated with near-universal τ edges, so campaign nodes wire heavily into benign background. Under the historically justified scaler, τ becomes selective; campaign–benign wiring drops sharply (and campaign connectivity fragments — a separate issue for later, not solved here).

---

## 8. Dataset mismatch (finalized)

### Manuscript platforms
Hyundai **Sonata**, Kia **Soul**, Chevrolet **Spark**.

Recoverable publication manifests reference files such as:
`Attack_free_HY_Sonata_train.csv`, `..._KIA_Soul_...`, `..._CHEVROLET_Spark_...`
(OneDrive challenge “car_track” paths in recovered manifests).

### PR #22 loader tags (actual)

| Tag | What it is |
|---|---|
| **OCSLab_Car** | HCRL **classic Car-Hacking** single-vehicle traces (`normal_run_data`, DoS, Fuzzy, gear, RPM). OEM **not** asserted. |
| **Challenge_D** | Car Hacking Challenge **Preliminary Training track D** (`Pre_train_D_*`). |
| **Challenge_S** | Challenge Preliminary Training **track S** (`Pre_train_S_*`). |

### Prominence

**PR #22 is not using the manuscript’s three OEM vehicle datasets.**

Workspace also contains `Dataset/ocslab_pipeline/{Hyundai,Kia,Chevrolet}/` with attack_free / flooding / fuzzy / malfunction files, but **filenames do not contain Sonata/Soul/Spark**, and **`build_scored_pool.py` does not use those directories**.

---

## 9. Root-cause ranking

### HIGH confidence
1. **Missing fleet scaler / standardization before cosine** — publication requires it; PR #22 omits it; R2 restores τ discrimination quantitatively.  
2. **Feature-scale domination** — constant `frame_count`/`message_rate` own ~99.85% of \(\|g\|^2\).  
3. **Wrong descriptor representation for cosine graph** — raw `g_i` instead of scaler-conditioned behaviour view.  
4. **Graph over-connectivity** — mechanical consequence of (1)–(3): 100% of pairs ≥ τ; dense campaign–benign edges.  
5. **Wrong dataset/platform inputs vs manuscript** — OCSLab_Car/Challenge_D/S ≠ Sonata/Soul/Spark.

### MEDIUM confidence
6. **Campaign behavioural incoherence in GT construction** — score-band + attack_type only; no prototype blend; multi-platform mix (prior diagnostic). Still relevant for F1 even after scaling (R2 leaves campaign fragmented into 5 components on seed 11).  
7. **DBSCAN over-merge** — expected once graph/features don’t separate campaign from benign (prior F1=0 analysis).  
8. **Campaign–GT Jaccard matching failure** — terminal symptom of over-merged predictions.

### LOW confidence / not primary
9. **τ=0.95 “incompatibility”** — τ is fine under the publication representation; it is non-discriminative only on raw `g_i`. Do **not** retune τ to “fix” raw features.  
10. **Campaign gate rejection** — gate accepts polluted high-cohesion clusters; failure is not rejection.  
11. **GNN architecture** — M2 already F1=0 on the same raw features; GNN is not required to explain collapse.

---

## 10. Decision

### A. Is PR #22 scientifically comparable to the FLEET-GUARD methodology in the paper?
**NO.**

Reasons: (i) similarity graph skips the recoverable benign fleet scaler / normalized behaviour view; (ii) vehicle data are classic OCSLab + Challenge D/S, not Sonata/Soul/Spark; (iii) campaign GT omits publication-style coordination prototype semantics.

### B. Is Campaign F1 = 0 evidence against FLEET-GUARD itself?
**No — it is evidence that PR #22 instantiated a different feature/data pipeline.**

The cosine collapse is explained by raw constant-dominated `g_i` without the publication scaler. Under the historical scaler (diagnostic only), campaign–benign pairs at τ=0.95 fall from 100% to ~0.5%.

### C. Minimum correction to make the reviewer ablation use the actual methodology
**Do not implement yet.** Minimum scientifically required corrections:

1. **Apply the publication benign fleet z-score (or equivalent train-benign standardization with the same feature order) before cosine / constrained kNN**, matching `behavior_only_vehicle_normalized`.  
2. **Align datasets with the manuscript claim** (Sonata/Soul/Spark challenge tracks), *or* explicitly reframe the ablation as classic+Challenge D/S and stop claiming OEM equivalence.  
3. (Likely also required for a fair behavioural-correlation test, but secondary to (1)–(2):) restore publication campaign coordination semantics rather than score-band+type-only GT.

No τ change is justified by these results as a first fix.

---

*End of diagnostic. No ablation code, thresholds, or generators were modified.*
