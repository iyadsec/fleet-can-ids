# PROPOSED_REVISED_METRIC_PROTOCOL.md

**Status:** PROPOSAL ONLY — **not approved**, **not used** for any η evaluation.  
**Purpose:** Define one explicit, reproducible evaluation protocol for the prospective η-revalidation experiment because the historical P7/P8 matcher/emitter (`extract_run_metrics`) is **UNRECOVERABLE**.

This protocol is **not** claimed to be the historical publication matcher. Results under this protocol must be labeled **revised-protocol**, never mixed with historical P7/P8 numbers without that distinction.

---

## Scope

Applies to controlled fleet scenarios with known ground-truth campaign vehicle sets (validation now; final OCSLab later — only after η freeze).

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

### Campaign gate (prediction side — independent of metrics)

A non-noise DBSCAN cluster \(C_k\) is a **predicted campaign** iff:

\[
r_k \ge \gamma,\quad |C_k| \ge \eta,\quad c_k \ge \beta
\]

with \(r_k\) = distinct vehicles in \(C_k\), \(|C_k|\) = node count, \(c_k\) = centroid cohesion (paper formula).  
DBSCAN noise (`-1`) is **never** a predicted campaign.

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

## Proposed matching rule (explicit — requires approval)

**Greedy one-to-one matching on vehicle-set Jaccard:**

1. Compute Jaccard \(J(P,G)=|V(P)\cap V(G)|/|V(P)\cup V(G)|\) for every predicted campaign \(P\) and GT campaign \(G\).
2. Discard pairs with \(J < \tau_J\).
3. Greedily accept pairs in decreasing \(J\), each \(P\) and each \(G\) used at most once.

**Proposed default:** \(\tau_J = 0.5\).

**Rationale for proposing (not asserting historical identity):**

- Provides a fully specified, reproducible matcher.
- Matches a common ablation peer, **but** prior audits forbid claiming it is the historical P7/P8 emitter.
- Artifact identities on saved P7/P8 rows are also consistent with **count-based** scoring when \(n_{\mathrm{GT}}=1\); Jaccard is stricter and explicit about overlap.

**Alternative (if reviewers reject Jaccard):** count-based detection for \(n_{\mathrm{GT}}=1\): a run is a campaign TP iff ≥1 predicted campaign exists; extra predicted campaigns are FPs. This must be chosen **before** η sweeps if preferred — do not switch after seeing scores.

---

## Campaign-level scores

Let \(n_P\) = # predicted campaigns, \(n_G\) = # GT campaigns, \(n_M\) = # matched pairs.

| Quantity | Definition |
|----------|------------|
| TP | \(n_M\) |
| FP | \(n_P - n_M\) |
| FN | \(n_G - n_M\) |
| Precision | \(\mathrm{TP}/n_P\) if \(n_P>0\) else 0 |
| Recall | \(\mathrm{TP}/n_G\) if \(n_G>0\) else 0 (or 0 if \(n_G=0\) and evaluating campaign detection on attack scenarios) |
| F1 | harmonic mean of precision & recall |

### Special scenarios

| Scenario | Rule |
|----------|------|
| Benign / no GT campaign (\(n_G=0\)) | Any \(n_P>0\) ⇒ false campaign; rate = 1{ \(n_P>0\) } per run |
| Isolated attack | Prefer \(n_P=0\); multi-vehicle predicted campaigns count as false escalations per scenario policy |
| Unrelated multi-incident (\(n_G>1\)) | **Incorrect merging** = 1 iff a single predicted campaign matches (or overlaps) ≥2 GT campaigns under the approved matcher; else 0 |

---

## Membership scores (vehicle level)

Using the **best matched** pair \((P^\*,G^\*)\) when matches exist; if \(n_G=1\) and \(n_M=0\) but \(n_P\ge1\), optionally evaluate largest \(P\) vs \(G\) **only if approved** (default proposal: **require a match**, else membership = 0).

\[
\begin{aligned}
\mathrm{MemP} &= |V(P^\*)\cap V(G^\*)| / |V(P^\*)| \\
\mathrm{MemR} &= |V(P^\*)\cap V(G^\*)| / |V(G^\*)| \\
\mathrm{MemF1} &= 2\,\mathrm{MemP}\,\mathrm{MemR}/(\mathrm{MemP}+\mathrm{MemR})
\end{aligned}
\]

Benign vehicles inside \(V(P^\*)\) are membership **false positives**.

---

## Fragmentation

For scenarios with \(n_G=1\):

| Quantity | Definition |
|----------|------------|
| `fragments_per_true_campaign` | \(n_P\) |
| `fragmentation_rate` | \(1\{n_P > 1\}\) |

Noise is excluded (not a predicted campaign). Misses (\(n_P=0\)) ⇒ fragmentation_rate = 0, fragments = 0.

---

## Aggregation

- Seeds: publication list when evaluating final tables; validation seeds from the reconstructed suite.
- Report mean ± std over seeds; campaign-size strata 2/5/10 when present.
- Label every table: `metric_protocol=revised_jaccard_0.5_v1` (or the approved id).

---

## Explicit non-claims

- Not the historical `extract_run_metrics` implementation.
- Not authorised to “reproduce” Strong/Weak F1 0.533/0.733/1.000 and 0.067/0.500/0.717 as a success criterion.
- Ablation Jaccard≥0.5 is used **only if this proposal is approved**, not by default silent inheritance.

---

## Approval gate

**Do not run η candidates until this protocol (or an amended written protocol) is approved.**

Return checkpoint: awaiting review → see `CHECKPOINT_VERDICT.md`.
