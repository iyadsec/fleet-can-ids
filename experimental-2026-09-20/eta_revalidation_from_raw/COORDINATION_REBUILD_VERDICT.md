# COORDINATION_REBUILD_VERDICT.md

## Classifications

| | |
|--|--|
| **S3 coordination strength** | `INSUFFICIENT_EVIDENCE` |
| **S4 coordination strength** | `INSUFFICIENT_EVIDENCE` |
| **Reconstruction consequence** | `REBUILD_ONLY_AS_NEW_PROSPECTIVE_PROTOCOL` |

## Strongest evidence (each)

**S3 — strongest historical fact:**  
`behavioural_coordination_only: true` in the authoritative master YAML, plus P7 aggregate metrics **without** any saved `coordination_strength` or post-blend descriptors. Peer hardcode `1.0` exists but is **not** P7 evidence.

**S4 — strongest historical fact:**  
Same master-YAML flag and P8 aggregates; same absence of numeric strength / feature dumps. Peer V4 `1.0` is **not** P8 evidence. Stand-in `0.35` is **not** publication evidence.

## Consequence

```text
REBUILD_ONLY_AS_NEW_PROSPECTIVE_PROTOCOL
```

Meaning:

- Enough recovered **methodology** (blend equation, score bands, budgets, peer builders) exists to define a **transparent new** validation-suite protocol.
- **Not** enough P7/P8 artifact evidence to claim the historical publication used exactly \(s=1.0\) versus another value in an allowed range (e.g. 0.75).
- Therefore any rebuild must **disclose** the chosen strength as a **prospective** setting, not as recovered historical truth.
- Do **not** claim exact reproduction of `build_mixed_validation_suite`.
- Do **not** treat provisional η=2 as publication-validated until such a prospective suite is agreed and re-run under the frozen V3 metric protocol.

No experiments were run in this audit. No code or parameters changed.
