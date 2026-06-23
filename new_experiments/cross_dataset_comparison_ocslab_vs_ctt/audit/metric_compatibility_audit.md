# Metric Compatibility Audit

**Comparison type:** descriptive cross-dataset comparison

OCSLab / DataChallenge 2019 is the **primary controlled fleet-campaign evaluation**.
can-train-and-test provides **independent external validation** across additional vehicles,
manufacturers, and attack types. The datasets are **not identical or directly interchangeable**.

## Classification key

| Code | Meaning |
|------|---------|
| A | Directly comparable |
| B | Comparable with caveat |
| C | Not directly comparable |

## Metric-by-metric audit

- **number of vehicles** — Directly comparable (A): Count of distinct vehicles; definition aligned.
- **attack families** — Directly comparable (A): Count/list of attack types; naming differs by dataset.
- **number of windows** — Comparable with caveat (B): Windowing parameters may differ; compare descriptively.
- **local precision** — Comparable with caveat (B): Different vehicles, attacks, labels, thresholds.
- **local recall** — Comparable with caveat (B): Different vehicles, attacks, labels, thresholds.
- **local F1** — Comparable with caveat (B): Different vehicles, attacks, labels, thresholds.
- **ROC-AUC** — Comparable with caveat (B): Score ranking comparable in spirit; distributions differ.
- **PR-AUC** — Comparable with caveat (B): Class imbalance and label sets differ.
- **descriptor size** — Directly comparable (A): Same descriptor schema; byte counts directly readable.
- **bandwidth reduction** — Directly comparable (A): Same ratio definition when raw-window proxy matches.
- **candidate transmission rate** — Directly comparable (A): Same definition: weak candidates / windows.
- **graph nodes** — Comparable with caveat (B): Node caps and sampling may differ between runs.
- **graph edges** — Comparable with caveat (B): Threshold grids differ; compare trends not absolute counts.
- **cross-vehicle edge percentage** — Directly comparable (A): Same edge-typing definition in framework.
- **benign fleet false campaign rate** — Comparable with caveat (B): Same scenario intent; simulation details differ.
- **isolated attack false campaign rate** — Comparable with caveat (B): Same scenario intent; simulation details differ.
- **unrelated incident incorrect merge rate** — Comparable with caveat (B): Same metric definition (v3); attack mix differs.
- **strong campaign F1** — Comparable with caveat (B): Controlled simulations; not real synchronized campaigns.
- **weak campaign F1** — Comparable with caveat (B): Controlled simulations; not real synchronized campaigns.
- **campaign-size trend** — Comparable with caveat (B): Supported sizes differ; trend comparison only.
- **edge-sensitivity trend** — Comparable with caveat (B): Grid coverage may differ; trend comparison only.
- **runtime** — Comparable with caveat (B): Hardware, caps, and dataset size differ.
- **memory** — Comparable with caveat (B): Hardware, caps, and dataset size differ.

## Global caveats

1. Local detection metrics are **descriptive** because datasets differ in vehicle population, attack design, and train/test construction.
2. Fleet scenario metrics use **controlled simulations** on both datasets; can-train-and-test does **not** contain real synchronized fleet campaigns.
3. Unrelated-incident **incorrect_merge_rate** must be reported explicitly when elevated (CTT pooled value = 1.0).
4. OCSLab numeric cells marked `SOURCE_NOT_IN_WORKSPACE` require syncing `new_experiments/final_end_to_end_publication_run/` and re-running this script.
