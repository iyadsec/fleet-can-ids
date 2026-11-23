# S3/S4 descriptor-construction sensitivity analysis (2026-11-21)

A separate analysis of the June 2026 strong (S3) and weak (S4) campaign scenarios. It does **not** replace
the June results, which remain the primary controlled baseline
(`experimental-2026-06-23/01_primary_ocslab_balanced/`).

## Question

In June, S3/S4 blend a shared malfunction-derived prototype (strength 1.0, computed over all splits) into
all 10 descriptors of every attacked participant, benign windows included, before graph construction.
This analysis asks how much of the June detection depends on that controlled descriptor similarity.

## Conditions

All conditions reuse the June artefacts read-only: descriptors, Isolation Forest scores, local thresholds
(0.45 / 0.70), global fleet scaler, shared fleet configuration (τ 0.95, k_same 2, k_cross 5, DBSCAN
eps 0.5 / min_samples 2, consolidation 0.85, decision gate), and campaign sizes 2/5/10 × 10 seeds. Only
descriptor construction changes. The conditions were declared before any result was inspected; nothing was
tuned.

| Condition | Construction |
|-----------|--------------|
| A_reproduction | June code path, coordination strength 1.0 (reproduction check) |
| B_natural | Coordination strength 0.0: unmodified OCSLab descriptors, benign windows untouched, no prototype |
| C_partial_train | Strength 0.5 on malicious windows only; prototype from the **train** split only |

B uses real attack windows assembled into a multi-vehicle scenario. These are unmodified OCSLab
observations, **not** naturally recorded coordinated attacks. OCSLab contains no fleet-wide campaigns.

## Results

Condition A reproduced all 60 June S3/S4 runs exactly (campaign F1 and membership F1, per run).

*Strict detection* = fraction of runs with at least one accepted campaign cluster containing coordinated
malicious windows from ≥ 2 attacked vehicles (`strict_detection.py`).

| Scenario | Size | Strict detection June / B / C | Membership F1 June / B / C | Campaign F1 June / B / C |
|----------|------|-------------------------------|----------------------------|--------------------------|
| Strong | 2 | 0.5 / 0.0 / 0.0 | 0.490 / 0.040 / 0.040 | 0.533 / 0.367 / 0.267 |
| Strong | 5 | 0.8 / 0.0 / 0.0 | 0.725 / 0.000 / 0.000 | 0.733 / 0.100 / 0.200 |
| Strong | 10 | 1.0 / 0.5 / 0.4 | 0.928 / 0.317 / 0.464 | 1.000 / 0.550 / 0.633 |
| Weak | 2 | 0.1 / 0.0 / 0.0 | 0.057 / 0.000 / 0.000 | 0.067 / 0.100 / 0.100 |
| Weak | 5 | 0.4 / 0.0 / 0.1 | 0.564 / 0.000 / 0.060 | 0.500 / 0.000 / 0.100 |
| Weak | 10 | 0.8 / 0.0 / 0.1 | 0.825 / 0.000 / 0.075 | 0.717 / 0.000 / 0.100 |

Removing the controlled descriptor similarity significantly reduced detection. Paired Wilcoxon tests
against June (30 pairs per scenario type) give p ≤ 0.002 for campaign F1 and membership F1 in both B and C.
Campaign F1 under B/C is inflated by clusters containing only benign windows; strict detection and
membership F1 are the informative measures. In B, same-participant malicious descriptors have mean 9-D
cosine 0.73–0.87, while only 0–1.4% of cross-platform malicious pairs reach τ = 0.95
(`natural_pair_similarity.csv`). Natural cross-vehicle similarity therefore rarely meets the graph's edge
threshold.

Descriptor serialization (322 test windows, 8 recordings, `descriptor_serialization.py`): packed binary
44 B vs 2,035 B raw (97.84% smaller); CSV text 79 B vs 4,679 B (98.32%). These are per-window payload sizes,
not fleet bandwidth.

## Files

| File | Content |
|------|---------|
| `run_campaign_validation.py` | Runs conditions A/B/C (`--conditions`, `--seeds`) |
| `summarise.py` | Reproduction check, per-condition summary, paired tests |
| `strict_detection.py` | Strict detection for June and validation runs |
| `pair_similarity.py` | 9-D cosine by pair type in condition B |
| `descriptor_serialization.py` | Binary/CSV serialization sizes |
| `outputs/*.csv` | Summary and run-level results |

The first condition C attempt failed (`KeyError: 'split'`). Its 60 FAIL rows, with local tracebacks, are
kept in `run_level_metrics_A_reproduction_B_natural_C_partial_train.csv`. In the `trace` column the local
directory prefix of the author's machine is replaced by `<CODE_REPO>/`; no other cell differs from the
file as generated (SHA-256 of the generated file
`bad35d1f1fd4bc14e8d481c23bcf156347ef62277d77de2b885adecb63b28a90`). The corrected C rerun is
`run_level_metrics_C_partial_train.csv`. Per-run CSVs (`outputs/runs/`) and logs are not committed.

## Reproduction

Requires the June code (branch `cursor/campaign-clustering`, commit `61a8203`), the June run directory and
the uncommitted June modules published in
[`reproducibility/june_2026/`](../reproducibility/june_2026/), plus local
OCSLab data. Python 3.13, torch 2.12, torch-geometric 2.7, scikit-learn, pandas, scipy.

```bash
export FLEET_GUARD_CODE_ROOT=/path/to/code   # June project root
cd "$FLEET_GUARD_CODE_ROOT"                   # resolve_project_root uses the working directory
python /path/to/validation_2026-11-21_natural_campaigns/run_campaign_validation.py
python /path/to/validation_2026-11-21_natural_campaigns/summarise.py
python /path/to/validation_2026-11-21_natural_campaigns/strict_detection.py
python /path/to/validation_2026-11-21_natural_campaigns/pair_similarity.py
python /path/to/validation_2026-11-21_natural_campaigns/descriptor_serialization.py
```

`run_campaign_validation.py` serves the committed source for project files that a sync client truncated to
0 bytes. It substitutes a file only when git has a non-empty version.
