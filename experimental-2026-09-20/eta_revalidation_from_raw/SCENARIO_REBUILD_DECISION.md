# SCENARIO_REBUILD_DECISION.md

```text
SCENARIO_REBUILD_PARTIALLY_JUSTIFIED
```

## Decision

A **full claim** that we can reconstruct the missing `build_mixed_validation_suite`
**without inventing methodology** is **not** justified.

A **partial** rebuild that **composes independently recovered peers**
(`validation_scenarios._coordinated_campaign_scenario`,
`apply_coordination_strength`, `DescriptorBudget`, score-band chunking)
**is** justified **only if** explicitly labeled as a **peer-composed prospective
validation suite**, not as recovery of the missing module.

### Why not fully justified

| Blocker | Status |
|---------|--------|
| `build_mixed_validation_suite` source | UNRECOVERABLE |
| Exact strength schedule inside that missing module | UNRECOVERABLE (peer/baseline suggest 1.0 for V3/V4 but do not prove the missing file) |
| Validation campaign-size strata generator (cs 2/5/10) | UNRECOVERABLE (stub artifact rows only) |
| `PUBLICATION_SCENARIOS` / TEST orchestration | UNRECOVERABLE (separate from validation suite, but shows same recovery gap) |

### Why partially justified

| Component | Status |
|-----------|--------|
| Blend equation + prototype mean | RECOVERED |
| S3/S4 score-band difference | RECOVERED |
| Node budget / fleet geometry | RECOVERED |
| Peer V0–V4 construction path | RECOVERED / RECOVERABLE_BY_COMPOSITION |
| Strength=1.0 for peer validation V3 **and** V4 | RECOVERABLE_BY_COMPOSITION (two independent code paths; still not the missing module) |

### Explicit non-claims

- Do **not** treat stand-in S4 strength **0.35** as publication evidence.
- Do **not** claim bit-identical recreation of `build_mixed_validation_suite`.
- Do **not** re-run η / validation / TEST from this audit alone.

### If a rebuild is later authorized

Required disclosure: coordinated scenarios **modify behavioural descriptor
features** via prototype blending while retaining **real CAN windows** and
**unmodified anomaly_score**.
