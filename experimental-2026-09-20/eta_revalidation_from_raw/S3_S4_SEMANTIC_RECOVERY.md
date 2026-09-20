# S3_S4_SEMANTIC_RECOVERY.md

## Pipeline placement (where construction enters)

```text
real CAN frames
  → windows (size 100 / stride 50)          [real]
  → extract BEHAVIOURAL_FEATURE_COLUMNS     [real measurements]
  → IsolationForest anomaly_score           [real score on real windows]
  → scenario sampling (which windows/vehicles)  [controlled composition]
  → apply_coordination_strength on behavioural features of target rows
        [CONTROLLED DESCRIPTOR MODIFICATION — not new CAN]
  → derive GNN 9-D (incl. unchanged anomaly_score + derived message_rate/burstiness/payload_entropy)
  → graph / GraphSAGE / DBSCAN / campaign gate
```

### Classification of the publication experiment

**B. Selected real CAN windows but modified their behavioural descriptors through prototype blending.**

- Not A (unchanged descriptors) for coordinated malicious/coordinated-role targets when strength > 0.
- Not C (entirely synthetic descriptors).
- Windows/frames remain real OCSLab traces; only selected behavioural feature cells are rewritten.

### Wording checks

| Statement | Technically correct? |
|-----------|----------------------|
| “No synthetic CAN attack frames were generated” | **Yes** — blending does not synthesize CAN frames/traffic. |
| “Descriptors were synthetically modified” / “behavioural features were artificially coordinated” | **Yes** — must be disclosed for coordinated scenarios with strength > 0. |

Do not conflate the two.

Evidence: **VERIFIED_FROM_CODE** (`coordination_strength.py`; master YAML `behavioural_coordination_only: true`).

---

## S3 vs S4 — recovered difference

| Property | S3 (strong) | S4 (weak) | Evidence label |
|----------|-------------|-----------|----------------|
| Anomaly-score eligibility | Prefer / require **strong band** `score ≥ 0.80` for malicious descriptors | Prefer / require **weak band** `0.55 ≤ score < 0.80` | **VERIFIED_FROM_CODE** (`_strong_band_mask` / `_weak_band_mask`; thresholds FREEZE) |
| Descriptor sampling | `_build_attacked_vehicle_chunk(..., attack_strength="strong")` + benign-on-attacked | same with `"weak"` | **VERIFIED_FROM_CODE** (peer) |
| Attack-family default for prototype | `STRONG_ATTACK_DEFAULT = "malfunction"` | `WEAK_ATTACK_DEFAULT = "malfunction"` (**same string**) | **VERIFIED_FROM_CODE** |
| Coordination strength (peer validation) | **1.0** hardcoded | **1.0** hardcoded | **VERIFIED_FROM_CODE** (peer only) |
| Coordination strength (baseline reconstructor) | **1.0** | **1.0** | **VERIFIED_FROM_CODE** (`ocslab_publication_baseline.py`) |
| Coordination strength (missing suite / PUBLICATION_SCENARIOS) | **UNRECOVERABLE** exact | **UNRECOVERABLE** exact; registry allows **[0.75, 1.0]** | **UNRECOVERABLE** / range **VERIFIED_FROM_CODE** |
| Prototype construction | mean behavioural vector over `attack_type` | same | **VERIFIED_FROM_CODE** |
| Target mask (peer) | `scenario_role == "coordinated"` (includes benign-on-attacked rows) | same | **VERIFIED_FROM_CODE** |
| Descriptors / vehicle | 10 (5 mal + 5 benign-on-attacked) | 10 | **VERIFIED_FROM_CODE** / ARTIFACT |
| Campaign size (val base) | 5 | 5 | **VERIFIED_FROM_ARTIFACT** |
| Campaign size (TEST) | 2, 5, 10 | 2, 5, 10 | **VERIFIED_FROM_ARTIFACT** |
| Fleet size | 20 | 20 | **VERIFIED_FROM_ARTIFACT** |
| Benign background | fleet_size − campaign_size benign vehicles × 10 benign desc | same | **VERIFIED_FROM_CODE** |

### Concise rule (peer-supported)

\[
\begin{aligned}
\text{S3} &= \text{strong anomaly band} + \text{shared campaign} + \text{prototype blend} \\
\text{S4} &= \text{weak anomaly band} + \text{shared campaign} + \text{same blend machinery}
\end{aligned}
\]

**Primary verified difference:** anomaly-score band used when selecting malicious descriptors — **not** a different blend equation. Peer code does **not** use a lower strength for S4.

Stand-in S4 strength **0.35** is **not** supported by the recovered peer validation builder or the publication baseline reconstructor.

---

## Expected graph effect (synthetic / analytic; no validation run)

Blending increases cross-vehicle behavioural similarity because targets are moved toward a **shared** \(\mathbf{p}\).

- At \(s=1\): behavioural vectors on \(C\) become identical → cosine on \(C\) → **1.0** (synthetic demo).
- On **9-D GNN** inputs, cosine need not reach 1.0 because **`anomaly_score` is not blended** and may differ across windows/vehicles; derived `message_rate`/`burstiness`/`payload_entropy` follow the blended behavioural fields.
- At \(s=0.35\): substantial residual of originals remains (+ noise) → cosine can stay far below τ=0.95 (synthetic demo ≈0.38 even in 2-D toy case).

τ=0.95 unchanged; this section is explanatory only.
