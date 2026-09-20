# PROPOSED_REVISED_METRIC_PROTOCOL_V2.md

**Status:** PROPOSAL ONLY — **not approved**, **not used** for any η evaluation.  
**Protocol id:** `metric_protocol=revised_jaccard_0.5_v2`  
**Supersedes:** `PROPOSED_REVISED_METRIC_PROTOCOL.md` (v1) for prospective η-revalidation once approved.

**Purpose:** Define one explicit, reproducible evaluation protocol for the prospective η-revalidation experiment because the historical P7/P8 matcher/emitter (`extract_run_metrics`) is **UNRECOVERABLE**.

This protocol is **not** claimed to be the historical publication matcher. Results under this protocol must be labeled **revised-protocol** (`revised_jaccard_0.5_v2`), never mixed with historical P7/P8 numbers without that distinction.

---

## Changes from V1 (methodological only)

| Topic | V1 | V2 |
|-------|----|----|
| Campaign matching | Greedy one-to-one on Jaccard | **Maximum-weight** one-to-one on Jaccard |
| \(\tau_J\) | Proposed 0.5 | **Frozen** \(\tau_J = 0.5\) (prospective revised choice; **not** claimed historical) |
| Membership | Best matched pair only | **All matched pairs**; micro-aggregate \(\sum\) TP/FP/FN vehicles; retain per-pair |
| Unmatched GT | Optional largest-\(P\) substitute | **Forbidden** — membership recall/F1 = 0 |
| Incorrect merging | Via one-to-one matcher overlap | **Independent** of matcher: multi-incident vehicles inside one predicted campaign |
| Fragmentation | \(n_P\) / \(1\{n_P>1\}\) for \(n_G=1\) | Only predicted campaigns with \(\lvert V(P)\cap V(G)\rvert>0\) count as fragments of \(G\) |

Preserved unchanged from V1: predicted-campaign / campaign-gate definition, DBSCAN noise handling, false-campaign definition, seed aggregation, revised-protocol labeling, non-claim vs historical P7/P8.

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

## Campaign matching (V2)

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

## Membership metrics (V2)

**Removed:** V1 “best matched pair only” rule and any substitute of the largest unmatched predicted cluster.

For **every matched pair** \((P,G)\):

\[
\begin{aligned}
\mathrm{TP}_{\mathrm{vehicle}}(P,G) &= |V(P)\cap V(G)| \\
\mathrm{FP}_{\mathrm{vehicle}}(P,G) &= |V(P)\setminus V(G)| \\
\mathrm{FN}_{\mathrm{vehicle}}(P,G) &= |V(G)\setminus V(P)|
\end{aligned}
\]

Per-pair membership:

\[
\begin{aligned}
P_{\mathrm{mem}}(P,G) &= \mathrm{TP}_{\mathrm{vehicle}}/(\mathrm{TP}_{\mathrm{vehicle}}+\mathrm{FP}_{\mathrm{vehicle}}) \\
R_{\mathrm{mem}}(P,G) &= \mathrm{TP}_{\mathrm{vehicle}}/(\mathrm{TP}_{\mathrm{vehicle}}+\mathrm{FN}_{\mathrm{vehicle}}) \\
F1_{\mathrm{mem}}(P,G) &= 2\,P_{\mathrm{mem}}\,R_{\mathrm{mem}}/(P_{\mathrm{mem}}+R_{\mathrm{mem}})
\end{aligned}
\]

(0 when the relevant denominator is 0.)

### Run-level aggregation (preferred: micro)

\[
\begin{aligned}
\mathrm{TP}^{\Sigma}_{\mathrm{vehicle}} &= \sum_{(P,G)\in\mathrm{Matches}} \mathrm{TP}_{\mathrm{vehicle}}(P,G) \\
\mathrm{FP}^{\Sigma}_{\mathrm{vehicle}} &= \sum_{(P,G)\in\mathrm{Matches}} \mathrm{FP}_{\mathrm{vehicle}}(P,G) \\
\mathrm{FN}^{\Sigma}_{\mathrm{vehicle}} &= \sum_{(P,G)\in\mathrm{Matches}} \mathrm{FN}_{\mathrm{vehicle}}(P,G)
\end{aligned}
\]

Then micro membership precision / recall / F1 from the summed counts.

Retain **per-pair** membership rows for auditability.

### Unmatched GT campaign \(G\)

- Membership recall = 0  
- Membership F1 = 0  
- Do **not** select the largest predicted cluster as a substitute.

For micro-aggregation, an unmatched GT \(G\) contributes:

- \(\mathrm{TP}_{\mathrm{vehicle}}=0\), \(\mathrm{FP}_{\mathrm{vehicle}}=0\), \(\mathrm{FN}_{\mathrm{vehicle}}=|V(G)|\)

### Unmatched predicted campaigns and membership precision

Unmatched predicted campaigns are **campaign-level FPs** (`FP_campaign`).

They are **not** included in the matched-pair micro membership sums above (those sums are only over matched pairs + unmatched-GT FN contributions).

To prevent unmatched predicted vehicles from disappearing from audit:

1. Record each unmatched \(P\) with \(V(P)\) in an `unmatched_predictions` table.
2. Report `unmatched_predicted_vehicle_count = sum_{unmatched P} |V(P)|` (vehicle instances; if a vehicle appears in multiple unmatched \(P\), count per campaign membership).
3. **Aggregate membership precision (micro, matched-focused)** does **not** add those vehicles as membership FPs (avoids double-counting with campaign FP).  
4. Optional audit statistic (not the primary MemP):  
   `membership_precision_including_unmatched_pred = TP^Σ_vehicle / (TP^Σ_vehicle + FP^Σ_vehicle + sum_{unmatched P} |V(P)|)`.

Primary reported MemP/MemR/MemF1 use the micro rule in (preferred micro) plus unmatched-GT FN contributions; unmatched predictions affect **campaign FP**, with the optional audit MemP documented when cited.

Benign vehicles inside a matched \(V(P)\) are membership **false positives** (`FP_vehicle`).

---

## Incorrect merging (V2)

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

## Fragmentation (V2)

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
- Label every table: `metric_protocol=revised_jaccard_0.5_v2`.

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

**Do not run η candidates until this V2 protocol (or a further amended written protocol) is approved.**

η remains completely unevaluated under this checkpoint.
