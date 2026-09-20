# PROPOSED_REVISED_METRIC_PROTOCOL_V3.md

**Status:** **APPROVED/FROZEN** for the prospective η-revalidation experiment. Do not modify.  
**Protocol id:** `metric_protocol=revised_jaccard_0.5_v3`  
**Supersedes:** `PROPOSED_REVISED_METRIC_PROTOCOL_V2.md` (v2) for prospective η-revalidation.

**Purpose:** Define one explicit, reproducible evaluation protocol for the prospective η-revalidation experiment because the historical P7/P8 matcher/emitter (`extract_run_metrics`) is **UNRECOVERABLE**.

This protocol is **not** claimed to be the historical publication matcher. Results under this protocol must be labeled **revised-protocol** (`revised_jaccard_0.5_v3`), never mixed with historical P7/P8 numbers without that distinction.

---

## Changes from V2 (membership aggregation only)

| Topic | V2 | V3 |
|-------|----|----|
| Primary micro MemP | Matched-pair FP only; unmatched predicted vehicles **excluded** from primary MemP | Unmatched predicted campaigns contribute \(\lvert V(P)\rvert\) to **primary** \(\mathrm{FP}_{\mathrm{vehicle}}\) |
| Optional `membership_precision_including_unmatched_pred` | Documented as audit-only | **Removed** (behavior is now primary) |
| Campaign FP vs membership FP | Avoided “double-count” by excluding unmatched \(P\) from MemP | **Not double-counting**: campaign FP = erroneous campaign declaration; membership FP = erroneous vehicle–campaign membership assignment |

All other V2 definitions are preserved unchanged: \(\tau_J=0.5\), maximum-weight Jaccard matching, campaign TP/FP/FN, unmatched-GT treatment, incorrect merging, fragmentation, false-campaign, DBSCAN noise, campaign gate, η definition, seed aggregation, non-claims.

---

## Scope

Applies to controlled fleet scenarios with known ground-truth campaign / incident vehicle sets (validation now; final OCSLab later — only after η freeze).

Does **not** apply to CTT in this phase.

---

## Definitions

### Units

| Symbol | Meaning |
|--------|---------|
| Node / descriptor | One behavioural window promoted into the fleet graph (candidate node) |
| Vehicle | Distinct `vehicle_token` / scenario vehicle id |
| Predicted campaign cluster | A DBSCAN cluster (label ≥ 0) that **passes the campaign gate** |
| GT campaign | A controlled ground-truth attacked-vehicle set for the scenario |
| GT incident | For independent-multi scenarios: one independently constructed attacked-vehicle group; `incident(v)` maps attacked vehicle \(v\) to that group |

### Campaign gate (prediction side — independent of metrics)

A non-noise DBSCAN cluster \(C_k\) is a **predicted campaign** iff:

\[
r_k \ge \gamma,\quad |C_k| \ge \eta,\quad c_k \ge \beta
\]

with \(r_k\) = distinct vehicles in \(C_k\), \(|C_k|\) = node count, \(c_k\) = centroid cohesion (paper formula).  
DBSCAN noise (`-1`) is **never** a predicted campaign and never enters \(n_P\), fragmentation, or incorrect-merge checks.

### Predicted campaign vehicle set

For predicted campaign \(P\):

\[
V(P) = \{\text{vehicle ids of nodes in } P\}
\]

### GT campaign vehicle set

For GT campaign \(G\):

\[
V(G) = \{\text{controlled attacked vehicles in that campaign}\}
\]

Benign-only scenarios have \(n_{\mathrm{GT}}=0\).

---

## Campaign matching (unchanged from V2)

Keep vehicle-set Jaccard:

\[
J(P,G)=\frac{|V(P)\cap V(G)|}{|V(P)\cup V(G)|}
\]

**Frozen:** \(\tau_J = 0.5\).

This is a **prospective revised-protocol** choice. **Do not** claim \(\tau_J=0.5\) was used by historical P7/P8.

### Maximum-weight one-to-one assignment

1. Build the predicted-campaign × GT-campaign Jaccard matrix \(J(P_i,G_j)\) for all pairs.
2. Pairs with \(J(P_i,G_j) < 0.5\) are **ineligible**.
3. Among eligible pairs, compute a **maximum-weight one-to-one assignment** maximizing total Jaccard similarity (e.g. Hungarian / linear-sum assignment on eligible weights).
4. Constraints: each predicted campaign matches at most one GT campaign; each GT campaign matches at most one predicted campaign.

Then:

| Quantity | Definition |
|----------|------------|
| \(n_P\) | # predicted campaigns (gate-passing, non-noise) |
| \(n_G\) | # GT campaigns |
| \(n_M\) / `TP_campaign` | # matched pairs |
| `FP_campaign` | \(n_P - \mathrm{TP}_{\mathrm{campaign}}\) |
| `FN_campaign` | \(n_G - \mathrm{TP}_{\mathrm{campaign}}\) |
| Campaign Precision | \(\mathrm{TP}_{\mathrm{campaign}}/n_P\) if \(n_P>0\) else 0 |
| Campaign Recall | \(\mathrm{TP}_{\mathrm{campaign}}/n_G\) if \(n_G>0\) else 0 |
| Campaign F1 | harmonic mean of campaign precision & recall (0 if both 0) |

### Special case \(n_G = 0\)

Any predicted campaign is a **false campaign**.  
`TP_campaign = 0`, `FP_campaign = n_P`, `FN_campaign = 0`.  
False-campaign indicator per run: \(1\{n_P > 0\}\).

Unmatched predicted campaigns are recorded as unmatched predictions and counted in `FP_campaign`. They must not silently disappear.

---

## Membership metrics (V3)

Membership is evaluated as **vehicle–campaign membership assignments**.

**Removed (from V1/V2):** “best matched pair only”; largest-cluster substitute for unmatched GT; V2 exclusion of unmatched predicted campaigns from primary MemP; optional `membership_precision_including_unmatched_pred`.

### Accumulation (primary micro)

Initialize \(\mathrm{TP}_{\mathrm{vehicle}}=\mathrm{FP}_{\mathrm{vehicle}}=\mathrm{FN}_{\mathrm{vehicle}}=0\).

For **every matched pair** \((P,G)\):

\[
\begin{aligned}
\mathrm{TP}_{\mathrm{vehicle}} &\mathrel{+}= |V(P)\cap V(G)| \\
\mathrm{FP}_{\mathrm{vehicle}} &\mathrel{+}= |V(P)\setminus V(G)| \\
\mathrm{FN}_{\mathrm{vehicle}} &\mathrel{+}= |V(G)\setminus V(P)|
\end{aligned}
\]

For **every unmatched predicted campaign** \(P\):

\[
\mathrm{FP}_{\mathrm{vehicle}} \mathrel{+}= |V(P)|
\]

For **every unmatched GT campaign** \(G\):

\[
\mathrm{FN}_{\mathrm{vehicle}} \mathrel{+}= |V(G)|
\]

Then **primary** micro membership:

\[
\begin{aligned}
\mathrm{MemP} &= \mathrm{TP}_{\mathrm{vehicle}}/(\mathrm{TP}_{\mathrm{vehicle}}+\mathrm{FP}_{\mathrm{vehicle}}) \\
\mathrm{MemR} &= \mathrm{TP}_{\mathrm{vehicle}}/(\mathrm{TP}_{\mathrm{vehicle}}+\mathrm{FN}_{\mathrm{vehicle}}) \\
\mathrm{MemF1} &= 2\,\mathrm{MemP}\,\mathrm{MemR}/(\mathrm{MemP}+\mathrm{MemR})
\end{aligned}
\]

with the usual zero-denominator handling (0 when the relevant denominator is 0).

Retain **per-pair** membership rows for matched pairs (auditability). For an unmatched GT campaign \(G\): per-campaign membership recall = 0 and F1 = 0; do **not** select the largest predicted cluster as a substitute.

### Unmatched predicted campaigns

Unmatched predicted campaigns remain **campaign-level FPs** (`FP_campaign`) **and** contribute \(|V(P)|\) to **membership-level** \(\mathrm{FP}_{\mathrm{vehicle}}\) in the primary micro metric.

Record each unmatched \(P\) (with \(V(P)\)) in an `unmatched_predictions` table for audit.

**If the same vehicle belongs to multiple unmatched predicted campaigns, count each occurrence as a separate erroneous vehicle–campaign membership assignment** (add \(|V(P)|\) per unmatched \(P\), without deduplicating vehicles across those campaigns).

### Campaign FP vs membership FP (not double-counting)

These measure **different** errors:

| Level | Error meaning |
|-------|----------------|
| Campaign FP | Erroneous **campaign declaration** |
| Membership FP | Erroneous **vehicle–campaign membership assignment** |

Counting an unmatched predicted campaign at both levels is intentional and is **not** treated as double-counting.

Benign vehicles inside a matched \(V(P)\) are membership **false positives** (`FP_vehicle`).

---

## Incorrect merging (unchanged from V2)

**Do not** define incorrect merging using the one-to-one matcher (a one-to-one matcher cannot assign one \(P\) to multiple GT campaigns).

For controlled **independent-incident** scenarios, let `incident(v)` be the controlled GT incident membership of attacked vehicle \(v\).

A predicted campaign \(P\) is an incorrect merge iff:

\[
\mathrm{IncorrectMerge}(P)=1
\iff
\bigl|\{ \mathrm{incident}(v): v\in V(P),\ v\ \text{belongs to a GT incident}\}\bigr| \ge 2
\]

Run-level incorrect-merging indicator:

\[
1\{\exists\,P:\ \mathrm{IncorrectMerge}(P)=1\}
\]

Benign vehicles do **not** create an incorrect merge; they are membership false positives only.

Keep this metric **separate** from Jaccard campaign matching.

---

## Fragmentation (unchanged from V2)

For GT campaign \(G\), fragments are predicted campaign clusters that contain **at least one** vehicle belonging to \(G\):

\[
\mathrm{fragments}(G)=\bigl|\{P:\ |V(P)\cap V(G)| > 0\}\bigr|
\]

\[
\mathrm{Fragmentation}(G)=1\{\mathrm{fragments}(G) > 1\}
\]

| Case | `fragments(G)` | `Fragmentation(G)` |
|------|----------------|--------------------|
| Complete miss | 0 | 0 |
| Single covering predicted campaign | 1 | 0 |
| GT split across ≥2 predicted campaigns | ≥2 | 1 |

A completely unrelated false predicted campaign (no vehicles from \(G\)) is a **campaign FP**, **not** a fragment of \(G\).

For multiple GT campaigns: compute fragmentation **per GT campaign**, then aggregate explicitly (e.g. mean of `Fragmentation(G)` over \(G\), and/or sum of `fragments(G)` — state which when reporting).

Noise is excluded (not a predicted campaign).

---

## False campaign rate

For benign / \(n_G=0\) runs: false-campaign rate = \(1\{n_P>0\}\) per run.  
Aggregate mean ± std over seeds.

---

## Aggregation over seeds

- Seeds: publication list for final tables; validation seeds from the reconstructed suite.
- Report mean ± std over seeds; campaign-size strata 2/5/10 when present.
- Label every table: `metric_protocol=revised_jaccard_0.5_v3`.

---

## Explicit non-claims

- Not the historical `extract_run_metrics` / P7/P8 matcher.
- \(\tau_J=0.5\) is **not** asserted as the historical publication threshold.
- Not authorised to “reproduce” Strong/Weak historical F1 figures as a success criterion for this protocol.
- Ablation Jaccard≥0.5 inheritance is only via this **approved** revised protocol, not silent default.

---

## Implementation check (pre-η)

Unit tests on **synthetic sets of vehicle ids only** (no synthetic CAN) must cover:

1. Perfect one-campaign match  
2. Partial match with Jaccard ≥ 0.5  
3. Partial match with Jaccard < 0.5  
4. Complete campaign miss  
5. False predicted campaign  
6. One GT campaign split across two predicted campaigns  
7. Two independent GT incidents merged into one prediction  
8. Two GT campaigns and two correctly separated predictions  
9. Benign vehicles added to an otherwise correct predicted campaign  
10. DBSCAN noise excluded from predicted campaigns  

These tests must **not** be used to tune \(\tau_J\).

---

## Approval gate

**Do not run η candidates until this V3 protocol (or a further amended written protocol) is approved.**

η remains completely unevaluated under this checkpoint.

---

## FREEZE STATUS

**APPROVED/FROZEN** for the prospective η-revalidation experiment (`revised_jaccard_0.5_v3`). Do not modify metric definitions, `tau_J=0.5`, or this document during/after η evaluation.

