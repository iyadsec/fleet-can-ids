# Diagnostic Report: Why Campaign F1 = 0 (real OCSLab ablation)

**Scope:** Diagnose only. No campaign-generator changes, no τ/β retuning, no M1–M4 re-run for metric improvement, no manuscript edits.

**Artifacts:** `experimental-reviewer-ablation/real_ocslab/diagnostics/`

---

## Executive verdict

Campaign F1 = 0 for M2/M3/M4 on strong and weak is **not** primarily because τ = 0.95 isolates campaign nodes.

It is because:

1. **Raw 9-D descriptors make nearly every pair look identical under cosine** (campaign–campaign, campaign–benign, and campaign–unrelated-malicious all ≥ 0.95 at ~100%), so the graph is over-connected and DBSCAN/gate cannot separate the planted campaign from background.
2. **Ground-truth campaigns are score-band + `attack_type` assemblies**, not publication-style behaviourally coordinated campaigns (no shared prototype blend; fleet vehicles mix physical platforms/traces).
3. Methods therefore emit **one large over-merged “campaign”** that fails Jaccard ≥ 0.5 matching.

This experiment is **not yet a clean test of the FLEET-GUARD behavioural-correlation hypothesis** on real OCSLab windows.

---

## 1. Dataset identity (loader-traced)

| Tag | Original dataset | Physical platform (as coded) | Trace files | Attack types | Labels | Paper Sonata/Soul/Spark? |
|---|---|---|---|---|---|---|
| **OCSLab_Car** | HCRL classic Car-Hacking | Loader tag only; OEM **not** asserted | `normal_run_data.txt`, `DoS_dataset.csv`, `Fuzzy_dataset.csv`, `gear_dataset.csv`, `RPM_dataset.csv` under classic root | attack_free, flooding, fuzzy, malfunction_gear, malfunction_rpm | CSV `R`/`T`; normal_txt all benign | **No** |
| **Challenge_D** | Car Hacking Challenge Preliminary Training (track D) | Loader tag only | `Pre_train_D_{0,1,2}.csv` | SubClass → flooding / fuzzing / replay / spoofing | Class Normal vs Attack* | **No** |
| **Challenge_S** | Challenge Preliminary Training (track S) | Loader tag only | `Pre_train_S_{0,1,2}.csv` | Same subclass taxonomy | Same | **No** |

Evidence: `build_scored_pool.py` assigns `physical_vehicle` / `vehicle_model` explicitly; `artifacts/pool_summary.json` lists the same three tags.

Paper OEM platforms (Hyundai Sonata / Kia Soul / Chevrolet Spark) appear in **recovered publication manifests** (`Attack_free_HY_Sonata_*`, `Attack_free_KIA_Soul_*`, `Attack_free_CHEVROLET_Spark_*`), **not** in this real_ocslab loader.

---

## 2. Campaign construction

**Rule (code):** `scenario_builder_real.py` / pool builder:

- Pick one `attack_type` with enough windows in the score band (strong ≥ 0.80 or weak ∈ [0.55, 0.80)).
- For each of 5 fleet vehicles, sample 10 windows from that type × band pool.
- Background: suspicious benign windows on the other 15 vehicles.

**Required:** score band + attack-type label.  
**Not required:** behavioural cosine similarity, same physical platform, same source trace, shared prototype / coordination blend.

### Seed 11 (strong)

- 50 campaign members, all **spoofing**
- Platforms: Challenge_S 26 / Challenge_D 24
- Traces: four Challenge files
- **Each of veh_00…veh_04 mixes Challenge_D and Challenge_S** inside the same simulated vehicle

Weak seed 11 is the same pattern (spoofing; two platforms; four traces; per-vehicle platform mix).

Members are selected because they share an **attack-type label and anomaly-score range**, not because they are verified behaviourally related observations beyond that label.

Full member tables: `diagnostics/campaign_members_seed11.csv`, `campaign_members_all_seeds.csv`.

---

## 3. Attack-type consistency (10 seeds × strong/weak = 20 GT campaigns)

| Metric | Value |
|---|---|
| Distinct attack types per GT campaign | **exactly 1 in 100%** (0% mix flooding+fuzzing+replay+…) |
| Multiple source platforms | **85%** |
| Multiple source traces | **85%** |
| Mean platforms / traces | 1.95 / 3.15 |

So campaigns do **not** currently mix unrelated attack families by label. They **do** routinely mix physical platforms and traces. Behavioural relatedness is **not** enforced beyond the shared type string.

---

## 4. Pre-graph behavioural coherence (raw 9-D cosine)

Across all seeds (strong / weak means):

| Pair set | Mean cosine | % pairs ≥ 0.95 |
|---|---|---|
| Within campaign | ~0.99999 | **100%** |
| Campaign vs benign | ~0.99997 | **100%** |
| Campaign vs unrelated malicious (sibling unrelated scenario) | ~0.99995 | **100%** |

**Critical:** under the features actually used for the graph, the planted campaign is **not separable** from background by cosine.

### Feature-scale note (diagnostic only; no retune)

On seed-11 strong scenario:

- `frame_count` and `message_rate` are **constant 100.0** (zero variance) → dominate L2 direction.
- After **within-scenario z-scoring**, within-campaign mean cosine ≈ **0.55** (only ~8% ≥ 0.95) while campaign–benign mean ≈ **−0.21** (~0.2% ≥ 0.95).

So a behavioural signal exists after normalization, but the frozen ablation path uses **raw** 9-D features for τ-graph / clustering / gate cohesion.

---

## 5. Graph connectivity (τ=0.95, same-k=2, cross-k=5)

| Scenario (mean over 10 seeds) | Campaign nodes | Edges among campaign | Cross-vehicle campaign edges | Components | Largest component | % isolated |
|---|---|---|---|---|---|---|
| strong | 50 | ~199 | ~134 | **1.0** | **50** | **0%** |
| weak | 50 | ~179 | ~110 | **1.0** | **50** | **0%** |

The GT campaign **is** a single connected component before GraphSAGE. Fragmentation of the **graph** is **not** the F1=0 cause. Connectivity is excessive relative to background (because almost all pairs clear τ).

---

## 6. Cluster / gate failure (M2, M3, M4)

For every strong/weak seed and all three methods:

- **n_pred = 1**, **n_matched = 0**, Campaign F1 = 0
- Failure modes present: **C** (DBSCAN over-merging), **E** (Jaccard < 0.5), matching failure; often also **B** (fragmentation before merge)
- Mode **D** (gate rejection) and **F** (no predicted cluster) are **absent**

### M2 seed-11 strong (pre-gate)

- DBSCAN: many small cross-vehicle clusters (27 non-noise), each with cohesion ≈ 1.0 → **gate passes**
- Fragment merge (threshold 0.85) collapses accepted clusters → **1 gated campaign**
- Final pred example: size **71**, only **15/50** GT members, **56** benign, 20 vehicles → Jaccard ≪ 0.5

M3/M4 show the same terminal pattern (one over-merged gated campaign, Jaccard fail). Because M2 already fails the same way on raw descriptors, this is **not** primarily a GraphSAGE-specific representation collapse.

---

## 7. τ diagnostic (no retuning)

Fraction of pairs with cosine ≥ 0.95 on **raw** 9-D features (means over seeds):

| Pair class | Strong | Weak |
|---|---|---|
| Campaign pairs | 100% | 100% |
| Benign pairs | 100% | 100% |
| Campaign vs benign | 100% | 100% |
| Campaign vs unrelated malicious | 100% | 100% |

τ = 0.95 is **not** “too strict for real data connectivity.” On these features it is **non-discriminative**.

---

## 8. Comparison to original controlled-scenario logic

Recoverable publication evidence (`recovered_publication_pipeline/...`):

| Aspect | Publication controlled campaign | real_ocslab generator |
|---|---|---|
| Attack relationship | Same attack type (defaults **malfunction** strong/weak) | Same attack type (sampled from available pool; seed 11 → spoofing) |
| Vehicles | Distinct OEM **catalog instances** (Hyundai / Kia / Chevrolet) | Virtual fleet IDs sampling ≤3 loader tags |
| Score band | Strong / weak thresholds | Same thresholds |
| Behavioural coherence | **`compute_campaign_prototype` + `apply_coordination_strength`** blends malicious features toward a shared family prototype (validation uses strength=1.0) | **No** prototype blend |
| Composition label | `controlled_same_attack` | Score + type only |
| Platforms | Sonata / Soul / Spark manifests | OCSLab_Car / Challenge_D / Challenge_S |

The new generator copied the **scenario shape** (20×10, campaign size 5) and **score/type banding**, but omitted the recoverable mechanism that **enforced** behavioural coordination.

---

## 9. Final diagnosis classification

| # | Cause | Applies? | Evidence |
|---|---|---|---|
| **1** | GT campaign construction does not enforce behavioural coherence | **Primary** | Score+type only; no prototype; 85% multi-platform/trace; per-vehicle platform mix |
| **2** | τ=0.95 produces insufficient real-data connectivity | **No (opposite)** | Campaign fully connected; 0% isolated; 100% of pairs (incl. vs benign) ≥ 0.95 |
| **3** | GNN representation failure | Secondary at most | M2 (no GNN) already F1=0 with same over-merge/Jaccard pattern |
| **4** | DBSCAN failure | **Yes (over-merge; some pre-merge fragmentation)** | Mode C in 10/10; B in many seeds; M2: many tiny clusters then merge to one |
| **5** | Campaign-gate failure | **Not rejection** | Gate **accepts** polluted high-cohesion clusters (mode D never fires) |
| **6** | Campaign–GT matching failure | **Yes (terminal)** | Jaccard always < 0.5; n_matched=0 |
| **7** | Feature / preprocessing mismatch | **Primary co-cause** | Constant `frame_count`/`message_rate`; raw cosine collapse; z-score restores separation diagnostically |
| **8** | Other | Dataset identity ≠ paper OEM fleet | Three loader tags ≠ Sonata/Soul/Spark |

### Causal chain (quantitative)

1. Construction plants a “campaign” as same-type score-band windows without publication-style prototype coordination (**§2, §8**).
2. Raw 9-D cosine cannot distinguish that set from benign / other malicious (**§4, §7** → 100% ≥ τ).
3. Graph is fully connected among campaign nodes **and** densely linked to background (**§5**).
4. DBSCAN + cohesion gate + fragment merge yield one large mixed cluster (**§6**).
5. Jaccard matching fails → Campaign F1 = 0 (**§6**).

---

## 10. What this means for the reviewer ablation

Before redesigning the ablation, decide whether the scientific target is:

- **A.** Publication-faithful *controlled behavioural coordination* on real windows (needs recoverable coordination semantics and/or feature view matching the historical similarity path), or  
- **B.** A *score-band / attack-type* multi-vehicle association task on raw 9-D features (current setup) — which does **not** currently measure the FLEET-GUARD behavioural-correlation claim.

This report does **not** change generators, thresholds, or methods. It only diagnoses.
