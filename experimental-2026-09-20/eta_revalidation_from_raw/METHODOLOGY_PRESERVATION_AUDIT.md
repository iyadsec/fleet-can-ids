# METHODOLOGY_PRESERVATION_AUDIT.md

**Status:** AUDIT ONLY — no manuscript edit, no experiment execution  
**Scope correction:** Preserve FLEET-GUARD architecture, methodology, Algorithms 1–2, equations, and existing symbol definitions.  
**Problem class:** Implementation / reproducibility recovery of controlled campaign construction — **not** methodological redesign.

**Documents reviewed:**

- `PROSPECTIVE_CAMPAIGN_PROTOCOL_V1.md`
- `PROSPECTIVE_CAMPAIGN_MANIFEST_SCHEMA.md`
- `COORDINATION_STRENGTH_SELECTION_PLAN.md`
- `PROSPECTIVE_VS_HISTORICAL_BOUNDARY.md`

**Classification vocabulary:**

| Class | Meaning | Adoption |
|-------|---------|----------|
| `SAFE_IMPLEMENTATION_DETAIL` | Code / experiment-setup recovery consistent with the current paper | May adopt in implementation |
| `EXPERIMENTAL_REPORTING_ONLY` | Clarification for Experimental Setup / Controlled Campaign Generation text only | May clarify later in setup text; not a detector change |
| `WOULD_CHANGE_METHODOLOGY` | Would alter paper-level method / scenario / pipeline meaning | **Must NOT adopt** |
| `WOULD_CHANGE_ALGORITHM` | Would alter Algorithm 1 or 2 | **Must NOT adopt** |
| `WOULD_CHANGE_SYMBOLS` | Would add/redefine paper symbols or notation-table entries | **Must NOT adopt** |

---

## 0. Non-negotiable preservation confirmations

| Asset | Status under this audit |
|-------|-------------------------|
| FLEET-GUARD architecture | **Unchanged** — no component may be inserted into the deployed/detection pipeline |
| Algorithm 1 | **Unchanged** |
| Algorithm 2 | **Unchanged** |
| Mathematical equations (detector / gate / graph) | **Unchanged** |
| Notation table / existing symbol meanings | **Unchanged** |
| S0–S4 conceptual definitions | **Unchanged** |
| Strong/weak anomaly thresholds | **Unchanged** (`0.55` / `0.80`) |
| Graph threshold τ semantics | **Unchanged** |
| γ, η, β semantics | **Unchanged**; η remains \(\lvert C_k\rvert \ge \eta\), not DBSCAN `min_samples` |
| Coordination strength \(s\) as paper/model symbol | **Forbidden** — must **not** become a paper symbol |

**Executable detection pipeline (must remain):**

```text
real CAN windows
  → existing 24-D behavioural feature extraction
  → existing Isolation Forest anomaly scoring
  → existing descriptor generation
  → existing graph construction
  → existing GraphSAGE
  → existing DBSCAN
  → existing campaign gate
```

Controlled campaign construction occurs **only** in Experimental Evaluation setup **before** fleet-level evaluation. It is **not** inference-time FLEET-GUARD.

---

## 1. Item-by-item audit

### A. Framing / naming

| # | Proposed item (from prospective docs) | Classification | Decision |
|---|----------------------------------------|----------------|----------|
| A1 | Label work as a **“NEW / PROSPECTIVE protocol”** that redesigns evaluation methodology | `WOULD_CHANGE_METHODOLOGY` if adopted as paper method | **REJECT as methodology.** Reframe as **implementation recovery** of missing `build_mixed_validation_suite` / experimental setup reproducibility. |
| A2 | Treat missing historical builder as justification to invent a new experimental methodology | `WOULD_CHANGE_METHODOLOGY` | **REJECT.** Missing builder ⇒ recover implementation consistent with paper S0–S4; do not redesign. |
| A3 | Research-question framing as two orthogonal paper axes including “coordination strength \(s\)” | `WOULD_CHANGE_METHODOLOGY` / `WOULD_CHANGE_SYMBOLS` if \(s\) enters paper method | **REJECT for paper.** Paper already distinguishes S3/S4 by **local evidence** under coordinated campaigns. \(s\) may exist only as an **internal construction dial** in code, not as a paper axis/symbol. |
| A4 | Wording: “controlled coordinated campaign construction” / “controlled descriptor-level campaign construction” | `EXPERIMENTAL_REPORTING_ONLY` | **OK** for Experimental Setup disclosure. |
| A5 | Disclose: real CAN windows; no synthetic CAN frames; descriptor-level controlled construction where used | `EXPERIMENTAL_REPORTING_ONLY` | **OK** (future setup clarification only; not now). |
| A6 | Claim natural coordinated attacks / causal common attacker | `WOULD_CHANGE_METHODOLOGY` (misrepresentation) | **REJECT** (already forbidden in prospective docs — retain rejection). |
| A7 | Claim historical P7/P8 exact reproduction when \(s\) unrecovered | Misreporting | **REJECT** as historical claim; does not authorize redesign. |

### B. S0–S4 scenario definitions

| # | Proposed item | Classification | Decision |
|---|---------------|----------------|----------|
| B1 | Preserve S0 = benign / no attack | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt** (already paper definition). |
| B2 | Preserve S1 = isolated single-vehicle attack | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt**. |
| B3 | Preserve S2 = independent multi-vehicle attacks; no shared campaign prototype across incidents | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt**. |
| B4 | Preserve S3 = strong local evidence + coordinated multi-vehicle campaign | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt**. |
| B5 | Preserve S4 = weak local evidence + coordinated multi-vehicle campaign | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt**. |
| B6 | Strong/weak = frozen anomaly bands only (`≥0.80` vs `[0.55,0.80)`) | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt**; do not redefine. |
| B7 | Redefine S3/S4 by different coordination strengths | `WOULD_CHANGE_METHODOLOGY` | **REJECT** (prospective docs already forbid weakening S4 via \(s\); reinforce). |
| B8 | Introduce new paper-level variables to distinguish S3 vs S4 coordination | `WOULD_CHANGE_SYMBOLS` / `WOULD_CHANGE_METHODOLOGY` | **REJECT**. |

### C. Coordination blend / \(s\)

| # | Proposed item | Classification | Decision |
|---|---------------|----------------|----------|
| C1 | Use recovered `apply_coordination_strength` / prototype blend **only** inside experimental campaign construction | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt** as experiment-setup code, **outside** Algorithms 1–2 and detection pipeline. |
| C2 | Transform only `BEHAVIOURAL_FEATURE_COLUMNS` (24 named features); leave `anomaly_score`, IDs, GT untouched | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt**. |
| C3 | Document recovered blend formula and \(\sigma=0.02(1-s)\) in **internal** recovery notes / code comments | `SAFE_IMPLEMENTATION_DETAIL` | **OK** internally. |
| C4 | Add blend equation / \(s\) / \(\tilde{p}\) / \(\varepsilon\) to paper equations or notation table | `WOULD_CHANGE_SYMBOLS` / `WOULD_CHANGE_METHODOLOGY` | **REJECT.** |
| C5 | Modify Algorithm 1 or 2 to include coordination strength | `WOULD_CHANGE_ALGORITHM` | **REJECT.** |
| C6 | Treat \(s\) as FLEET-GUARD architecture / graph / GraphSAGE / campaign-gate parameter | `WOULD_CHANGE_METHODOLOGY` / `WOULD_CHANGE_SYMBOLS` | **REJECT.** |
| C7 | Candidate construction values \(s\in\{0.75,1.0\}\) as **code configuration** to recover consistent S3/S4 instantiation | `SAFE_IMPLEMENTATION_DETAIL` (config only) | **OK** as unrecovered builder config recovery — **not** a paper symbol. |
| C8 | Formal paper-level “coordination strength selection” as part of FLEET-GUARD methodology | `WOULD_CHANGE_METHODOLOGY` | **REJECT.** |
| C9 | Non-F1 construction sanity checks (cosine, variance, identical-vector fraction) when choosing builder config | `SAFE_IMPLEMENTATION_DETAIL` | **OK** as engineering checks for non-degenerate **evaluation setup**, not as detector method. |
| C10 | Introduce \(\delta_{\mathrm{coord}}\) (or similar) into paper notation | `WOULD_CHANGE_SYMBOLS` | **REJECT** for paper. May exist only as internal engineering threshold if needed. |
| C11 | Stand-in S4 strength `0.35` as paper or default method | `WOULD_CHANGE_METHODOLOGY` (inconsistent with recovered peers / paper S3–S4 meaning) | **REJECT** (already rejected in prospective docs). |
| C12 | Insert prototype blending into inference-time / deployed pipeline | `WOULD_CHANGE_METHODOLOGY` / `WOULD_CHANGE_ALGORITHM` | **REJECT.** |

### D. Fleet budget / sizes / thresholds

| # | Proposed item | Classification | Decision |
|---|---------------|----------------|----------|
| D1 | Fleet 20 × 10 descriptors = 200 nodes | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt** (existing publication config). |
| D2 | Campaign sizes `{2,5,10}` | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt**. |
| D3 | Keep strong/weak thresholds `0.80` / `0.55` | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt**; do not change. |
| D4 | Change thresholds, τ, γ, η, β meanings to fix validation issues | `WOULD_CHANGE_METHODOLOGY` / `WOULD_CHANGE_SYMBOLS` | **REJECT.** |

### E. η

| # | Proposed item | Classification | Decision |
|---|---------------|----------------|----------|
| E1 | Preserve η as \(\lvert C_k\rvert \ge \eta\) | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt** — definition unchanged. |
| E2 | Equate η with DBSCAN `min_samples` | `WOULD_CHANGE_SYMBOLS` / `WOULD_CHANGE_METHODOLOGY` | **REJECT.** |
| E3 | Remove η or reinterpret η | `WOULD_CHANGE_METHODOLOGY` / `WOULD_CHANGE_ALGORITHM` | **REJECT.** |
| E4 | Re-validate η under consistent recovered experimental setup (validation only; existing candidate set) | `SAFE_IMPLEMENTATION_DETAIL` | **OK** as implementation consistency / revalidation — **not** a definition change. |
| E5 | Treat prior reconstructed η=2 as redesign of η semantics | `WOULD_CHANGE_METHODOLOGY` | **REJECT.** Prior result is provisional under non-equivalent builder; solve via consistent implementation, not redefinition. |

### F. Manifest / audit artifacts

| # | Proposed item | Classification | Decision |
|---|---------------|----------------|----------|
| F1 | Per-descriptor construction manifest (source trace, window, scores, hashes, GT ids) | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt** for reproducibility of experimental setup. |
| F2 | Record construction config field `coordination_strength` in **code manifests** | `SAFE_IMPLEMENTATION_DETAIL` | **OK** as implementation provenance — **not** paper notation. |
| F3 | Publish \(s\) as a FLEET-GUARD model/paper symbol because it appears in manifests | `WOULD_CHANGE_SYMBOLS` | **REJECT.** Manifest ≠ notation table. |
| F4 | Hashing / original-descriptor store | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt**. |

### G. Prototype construction choices

| # | Proposed item | Classification | Decision |
|---|---------------|----------------|----------|
| G1 | Prototype = mean behavioural vector for an `attack_type` (recovered code) | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt** in builder only. |
| G2 | Freeze exact prototype pool boundary / platform attack-family overrides in code for reproducibility | `SAFE_IMPLEMENTATION_DETAIL` | **Adopt** when recovering the suite — document in setup/code, not as new detector math. |
| G3 | Add prototype construction as a new paper algorithm step | `WOULD_CHANGE_ALGORITHM` | **REJECT.** |

### H. Detection stack

| # | Proposed item | Classification | Decision |
|---|---------------|----------------|----------|
| H1 | Keep graph construction, GraphSAGE, DBSCAN, campaign gate unchanged | `SAFE_IMPLEMENTATION_DETAIL` | **Required preserve.** |
| H2 | Alter graph/GNN/DBSCAN/gate to compensate for missing builder | `WOULD_CHANGE_METHODOLOGY` / `WOULD_CHANGE_ALGORITHM` | **REJECT.** |
| H3 | Use `revised_jaccard_0.5_v3` as evaluation metric protocol already frozen for revalidation | `EXPERIMENTAL_REPORTING_ONLY` / evaluation protocol (already in flight) | Does **not** alter Algorithms 1–2 or detector equations; out of scope to redesign further here. |

### I. Paper impact (future, not now)

| # | Proposed item | Classification | Decision |
|---|---------------|----------------|----------|
| I1 | Modify manuscript architecture / equations / Algorithms 1–2 / notation for \(s\) | `WOULD_CHANGE_*` | **REJECT.** Manuscript not modified in this stage. |
| I2 | Later Experimental Setup clarification only (construction disclosure; real data; no synthetic CAN; descriptor-level control if used) | `EXPERIMENTAL_REPORTING_ONLY` | **Allowed later** if reviewers require; must not redefine detector. |

---

## 2. Summary of rejected (must not adopt)

Anything classified `WOULD_CHANGE_METHODOLOGY`, `WOULD_CHANGE_ALGORITHM`, or `WOULD_CHANGE_SYMBOLS`:

1. Treating the missing builder as license to redesign FLEET-GUARD or invent a new paper methodology.  
2. Elevating coordination strength \(s\) (or \(\tilde{p}\), \(\varepsilon\), \(\delta_{\mathrm{coord}}\)) into paper symbols, equations, or Algorithms 1–2.  
3. Making \(s\) a graph / GraphSAGE / campaign-decision / architecture parameter.  
4. Redefining S3/S4 by coordination strength instead of local anomaly evidence.  
5. Inserting prototype blending into the inference-time detection pipeline.  
6. Changing η’s mathematical definition or equating η with DBSCAN `min_samples`.  
7. Changing τ, γ, β, score thresholds, or S0–S4 conceptual meanings to “fix” reproducibility.  
8. Framing “NEW PROSPECTIVE PROTOCOL” as a replacement methodology rather than **implementation recovery** of experimental campaign construction.

---

## 3. What remains the real fix

**Exact implementation / reproducibility issue still needing fixing:**

> Recover and implement a `build_mixed_validation_suite` (or equivalent) that instantiates the **already paper-defined** S0–S4 controlled evaluation conditions using real OCSLab windows and the recovered experimental construction helpers (descriptor sampling by existing strong/weak bands; shared campaign membership for S3/S4; prototype-blend construction dial **only in experimental setup**), with full audit manifests — **without** altering FLEET-GUARD architecture, Algorithms 1–2, equations, or symbol definitions.

This is an **implementation-recovery** problem. It is **not** a methodology redesign problem.

η issues are to be addressed later by **revalidation under a consistent recovered experimental setup**, preserving \(\lvert C_k\rvert \ge \eta\).

---

## 4. Required reframing of prior prospective docs (guidance only; docs not rewritten as methodology)

| Prior wording risk | Required interpretation going forward |
|--------------------|----------------------------------------|
| “NEW PROSPECTIVE PROTOCOL” as method | **Implementation recovery plan** for experimental campaign construction |
| \(s\) as research axis / paper dial | **Internal builder configuration** only; never paper/model symbol |
| Blend equation in protocol §3 | **Code-level construction detail**; not a FLEET-GUARD equation |
| “Freeze \(s\) then select η” as new method | Order of **engineering recovery / revalidation**; η definition unchanged |
| Manifest `coordination_strength` column | Setup provenance field; not notation-table entry |

No manuscript changes in this audit. No experiments executed.

---

## 5. Audit verdict

```text
METHODOLOGY_PRESERVATION_AUDIT_COMPLETE
```

FLEET-GUARD architecture, Algorithms 1 and 2, equations, and existing symbol definitions remain **unchanged**.  
\(s\) will **not** become a paper/model symbol.  
Adopt only `SAFE_IMPLEMENTATION_DETAIL` and (later, if needed) `EXPERIMENTAL_REPORTING_ONLY` items.  
Reject all `WOULD_CHANGE_*` items.
