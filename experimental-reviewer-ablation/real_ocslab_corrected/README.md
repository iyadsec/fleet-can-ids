# real_ocslab_corrected — Reviewer ablation (methodology correction)

Answers Reviewer Sec 4.5.2 using recoverable FLEET-GUARD methodology on **Sonata / Soul / Spark** car_track traces.

**PR #22 (`real_ocslab/`) preserved** — do not use its Campaign F1=0 in the paper.

## Results (10 seeds)

| Method | Strong Campaign F1 | Weak | Strong Membership F1 | Independent Merge Rate |
|--------|-------------------:|-----:|---------------------:|-----------------------:|
| M1 local IF | N/A | N/A | N/A | N/A |
| M2 descriptor clustering | **0.113 ± 0.064** | N/A | 0.267 ± 0.134 | 0.191 ± 0.051 |
| M3 GCN | 0.000 ± 0.000 | N/A | 0.000 | 0.750 ± 0.250 |
| M4 GraphSAGE | 0.000 ± 0.000 | N/A | 0.000 | 0.717 ± 0.236 |

Weak arm unsupported on TEST under regenerated IF (2 weak-band windows). No parameter retuning.

See `FINAL_REPORT.md` for full A–M report and interpretation.

## Reproduce

```bash
python experimental-reviewer-ablation/real_ocslab_corrected/check_dataset_gate.py
python experimental-reviewer-ablation/real_ocslab_corrected/build_scored_pool_corrected.py  # if pool missing
python experimental-reviewer-ablation/real_ocslab_corrected/run_corrected_ablation.py
```
