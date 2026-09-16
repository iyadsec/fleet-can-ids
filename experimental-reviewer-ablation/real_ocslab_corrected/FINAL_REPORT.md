# Final report — corrected reviewer ablation

## Status

**Executed** on manuscript car_track Sonata / Soul / Spark traces (downloaded from HCRL Challenge 2019 Dropbox).  
PR #22 `real_ocslab/` remains diagnostic provenance and **must not** be used in the paper.

Weak campaigns: **UNSUPPORTED** on held-out TEST under regenerated IF (only 2 weak-band malicious windows; need ≥25). No τ/DBSCAN/gate retuning performed.

---

### A. Dataset provenance

| Platform | Local path | Identity |
|----------|------------|----------|
| Hyundai Sonata | `Dataset/ocslab_pipeline/_source_car_track/car_track_*/*Sonata*` | **Confirmed** by filenames |
| Kia Soul | `.../*KIA_Soul*` | **Confirmed** |
| Chevrolet Spark | `.../*CHEVROLET_Spark*` | **Confirmed** |

Source: HCRL In-Vehicle Network Intrusion Detection Challenge (`car_track_{preliminary,final_1st,final_2nd}_train`).  
21/21 publication `balanced_split_manifest.csv` basenames present. Classic Car-Hacking placeholders removed from OEM dirs.

### B. Split / leakage

Publication `balanced_split_manifest.csv` contiguous segments / complete-trace assignments.  
`split_manifest.csv` written; zero multi-split segment leakage. Overlapping windows never cross partitions.

### C. Campaign semantics

Prototype blend `coordination_strength=1.0` on coordinated rows. Composition strong `{H:2,K:2,C:1}`.  
Primary family preferred `malfunction`; **fell back to `fuzzy`** (no malfunction in strong TEST band under this split).

### D–E. Feature / scaler path

24-D → IF (benign TRAIN only) → 9-D `g_i` → **`fleet_benign_scaler.json` (exact feature match) → cosine → constrained kNN**.

### F. Seed-11 cosine (scaled)

| Bucket | mean | median | % ≥ 0.95 |
|--------|-----:|-------:|---------:|
| camp–camp | 0.711 | 0.779 | 28.8 |
| camp–benign | −0.064 | −0.090 | **0.00** |

**PASS** — not PR #22 global collapse.

### G. M1–M4 definitions

As frozen in `config.yaml` (τ=0.95, k=2/5, β=0.5, fragment=0.85, GraphSAGE 9→64→32, 30 ep).  
M2 clusters **scaled 9-D `g_i`** (same representation as graph node features).

### H. Ten-seed results (mean ± std)

| Method | Strong Campaign F1 | Weak Campaign F1 | Strong Membership F1 | Weak Membership F1 | Independent Merge Rate |
|--------|-------------------:|-----------------:|---------------------:|-------------------:|-----------------------:|
| M1 local IF | N/A | N/A | N/A | N/A | N/A |
| M2 descriptor clustering | **0.113 ± 0.064** | N/A | **0.267 ± 0.134** | N/A | **0.191 ± 0.051** |
| M3 GCN | 0.000 ± 0.000 | N/A | 0.000 ± 0.000 | N/A | 0.750 ± 0.250 |
| M4 GraphSAGE | 0.000 ± 0.000 | N/A | 0.000 ± 0.000 | N/A | 0.717 ± 0.236 |

### I. Unrelated-incident merging

Graph methods show high incorrect-merge rates (~0.72–0.75). M2 lower (~0.19).

### J. Metrics

Shared harness: Jaccard ≥ 0.5 greedy matching; Campaign P/R/F1; Membership F1; Fragmentation; merge/false-campaign rates (`metrics.py`).

### K. Shared scenarios

Verified per seed: M3 edge signature == M4; shared scaled features; synthetic=0; identical frozen scenario CSVs.

### L. Remaining vs historical publication

- Regenerated IF scores (not frozen publication descriptors) → TEST weak-band nearly empty; strong TEST dominated by fuzzy.
- Absolute Section VII numeric identity not claimed.
- Weak campaigns not evaluated.

### M. Paper suitability?

**Conditionally YES for a corrected strong/unrelated component ablation**, with explicit caveats:
1. Weak arm unsupported under regenerated IF on this TEST split (do not invent weak windows / retune thresholds).
2. Under frozen methodology, **M2 > M3/M4** on strong Campaign F1; GraphSAGE does **not** outperform GCN or descriptor clustering here.
3. PR #22 zero-F1 results remain invalid for the paper.

---

## Scientific interpretation (no retuning)

1. **Local IF:** provides anomaly scores / local alerts only; no campaign reconstruction (N/A).
2. **Descriptor clustering (M2):** only method with non-zero strong Campaign F1 (~0.11); fragments into many clusters (often matches partially).
3. **Graph representation learning (M3/M4):** after publication scaler, graphs are selective (~300 edges) but GNN+DBSCAN yields typically **one** predicted campaign that fails Jaccard ≥ 0.5 → Campaign F1 = 0.
4. **GraphSAGE vs GCN:** no benefit — both 0.000 strong Campaign F1; similar high unrelated merge rates.
5. **Strong vs weak:** weak not measurable on TEST.
6. **Unrelated merging:** graph methods merge independent incidents more than M2.

**Honest conclusion:** correcting the scaler/data/campaign-semantics pipeline removes the PR #22 cosine-collapse artifact, but under frozen τ/DBSCAN/gate settings this controlled ablation does **not** show GraphSAGE superiority over descriptor clustering.
