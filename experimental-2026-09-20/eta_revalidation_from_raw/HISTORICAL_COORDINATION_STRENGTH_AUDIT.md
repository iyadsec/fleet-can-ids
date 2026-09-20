# HISTORICAL_COORDINATION_STRENGTH_AUDIT.md

## Question

Do saved historical **P7/P8** (and related balanced-publication) artifacts
distinguish `coordination_strength = 1.0` from other allowed values (esp. **0.75**)?

**Answer: No.**

---

## s=1 signature test (saved artifacts only)

Recovered code: at \(s=1\), \(\varepsilon=0\) and \(\mathbf{d}'_i[C]=\mathrm{clip}(\mathbf{p})\), so
transformed coordinates collapse across coordinated targets.

| Required measurement | Status |
|----------------------|--------|
| Identify coordinates \(C\) in a **saved post-coordination** table | **Impossible** — no such table in archive |
| Within-campaign feature variance on \(C\) | **Not computable** |
| Pairwise equality / near-equality | **Not computable** |
| Pairwise cosine on \(C\) | **Not computable** |
| Number of unique transformed vectors | **Not computable** |
| Numerical tolerance | **n/a** |

**Result:** neither the \(s=1\) nor the \(s<1\) signature can be confirmed or rejected from P7/P8 artifacts.

High F1, high cross-vehicle edge counts, or small `predicted_campaign_size` are **not** accepted here as proof of \(s=1\).

---

## Indirect graph / cluster signatures

| Metric (Strong / Weak means) | Observation | Constraint on \(s\)? |
|------------------------------|-------------|----------------------|
| `cross_vehicle_edges` | ~165 / ~99 | Compatible with several \(s\) that raise similarity enough to pass τ=0.95 often; **non-unique** |
| `largest_cluster_size` | ~33.6 / ~11.1 | DBSCAN geometry aggregate; **does not** encode descriptor collapse |
| `malicious_cross_vehicle_edge_purity` | **all NaN** | No information |
| Cohesion / cosine columns | **absent** | No information |

These remain **INDIRECT** and **insufficient** to separate 1.0 from 0.75.

---

## Publication cluster artifact revisit

- Strong mean `largest_cluster_size` ≈ **33.63** from `campaign_metrics.csv` / P7 (30 runs, 200 nodes).
- Weak mean ≈ **11.1** (P8 family).
- No accompanying descriptor/embedding/cluster-assignment dump.
- **Cannot** infer coordination strength from these aggregates without forbidden performance fitting.

---

## Classifications (artifact-centric)

| Scenario | Classification | Strongest evidence |
|----------|----------------|--------------------|
| **S3** | **INSUFFICIENT_EVIDENCE** | Archive proves coordination was **enabled** (`behavioural_coordination_only: true`) but records **no numeric \(s\)** and **no** post-blend feature table for an s=1 collapse test. Peer code suggesting 1.0 is **not** a P7 artifact. |
| **S4** | **INSUFFICIENT_EVIDENCE** | Same gap for P8; peer V4 hardcode 1.0 is PEER_IMPLEMENTATION only; registry allows [0.75, 1.0]. |

Not `EXACT_VALUE_RECOVERED` (no direct publication config value).  
Not `STRONGLY_SUPPORTED_1_0` (no unique artifact signature).  
`SUPPORTED_RANGE_ONLY` would require treating generic registry ranges as publication config; they are **not** proven to be the missing `PUBLICATION_SCENARIOS` values, so the stricter label is **INSUFFICIENT_EVIDENCE** for the numeric strength used in the historical P7/P8 run.
