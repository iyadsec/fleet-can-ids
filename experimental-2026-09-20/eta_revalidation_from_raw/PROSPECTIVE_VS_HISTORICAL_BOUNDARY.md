# PROSPECTIVE_VS_HISTORICAL_BOUNDARY.md

**Status:** Boundary declaration for NEW / PROSPECTIVE protocol V1  
**Forensic verdict cited:** `COORDINATION_REBUILD_VERDICT.md`

| Item | Verdict |
|------|---------|
| S3 historical coordination strength | `INSUFFICIENT_EVIDENCE` |
| S4 historical coordination strength | `INSUFFICIENT_EVIDENCE` |
| Historical builder | Unrecoverable (`build_mixed_validation_suite` missing) |
| Decision | `REBUILD_ONLY_AS_NEW_PROSPECTIVE_PROTOCOL` |

---

## 1. What this prospective protocol is

A **transparent, new** controlled descriptor-level campaign construction protocol for the revised FLEET-GUARD evaluation that:

- preserves the research question (local evidence strength ⊥ cross-vehicle coordination strength);  
- reuses **recovered** mathematical and implementation pieces where verified;  
- explicitly marks **prospective choices** where history is silent;  
- emits an auditable manifest for every constructed node.

---

## 2. What this prospective protocol is not

| Non-claim | Statement |
|-----------|-----------|
| Not historical P7/P8 | Does **not** reconstruct or reproduce publication P7/P8 runs |
| Not recovered \(s\) | \(s\in\{0.75,1.0\}\) are **candidates**, not historical facts |
| Not natural coordination | Not “naturally occurring coordinated attacks” |
| Not causal attacker | No claim of common attacker provenance in the real world |
| Not synthetic CAN | Does not generate synthetic CAN frames; it may **rewrite behavioural descriptor cells** under a controlled transform |
| Not η freeze | Does not select or freeze η; prior η=2 is **PROVISIONAL ONLY** |
| Not stand-in 0.35 | Rejects reconstructed stand-in weak strength `0.35` as historical or prospective S4 coordination |

---

## 3. Inherited recovered pieces vs prospective pieces

### RECOVERED (may be cited as methodology provenance)

- Blend equation \(d'=(1-s)d+s\tilde{p}+\varepsilon\)  
- Noise \(\sigma=0.02(1-s)\) and target-subset clipping  
- Transformed coordinate set = `BEHAVIOURAL_FEATURE_COLUMNS` (24 named features)  
- `anomaly_score` not blended  
- Strong/weak score bands `≥0.80` vs `[0.55, 0.80)`  
- Fleet budget 20×10=200; campaign sizes `{2,5,10}`  
- Prototype = mean behavioural vector for an `attack_type`  
- S3 vs S4 primary difference = score band, not a different blend equation  
- Master YAML `behavioural_coordination_only: true`

### PROSPECTIVE_CHOICE_REQUIRED (must not be silently invented)

- Single frozen \(s\) among `{0.75, 1.0}`  
- Exact prototype pool boundary (which descriptor table rows enter the mean)  
- Chevrolet / platform-specific attack-family overrides  
- Numeric degeneracy margins in the \(s\) selection plan (\(\delta_{\mathrm{coord}}\), etc.)  
- Exact missing-suite sampling quirks of `build_mixed_validation_suite`  
- η among `{2,3,5,10}`

### NOT_APPLICABLE

- Claiming a unique historical P7/P8 `coordination_strength` artifact value  
- Claiming equivalence between prospective validation seeds and publication TEST seeds without disclosure  
- Using detection F1 to define coordination strength  

---

## 4. Allowed wording vs forbidden wording

**Allowed:**

- “controlled coordinated campaign construction”  
- “controlled descriptor-level campaign construction”  
- “real CAN observations / real extracted descriptors / controlled descriptor transformation / constructed fleet campaign membership”  
- “prospective coordination strength candidate \(s=…\)”  
- “recovered blend implementation; unrecoverable historical strength”

**Forbidden:**

- “reproduced P7/P8 coordination”  
- “historical \(s\) was 1.0” (or 0.75) stated as fact  
- “naturally occurring multi-vehicle coordinated attack dataset”  
- “common attacker confirmed by construction”

---

## 5. Relationship to prior reconstructed validation

The earlier reconstructed validation path (stand-in builder; weak blend 0.35; partial seed mismatch) is **not equivalent** to this prospective protocol and **not equivalent** to historical publication construction.

Results from that reconstruction (including provisional η=2) **do not** transfer as publication-validated parameters.

---

## 6. Stop line

Design documents in this set stop at protocol specification.  
No campaign generation, validation, TEST, CTT, η evaluation, or manuscript edit is authorized by this boundary file.
