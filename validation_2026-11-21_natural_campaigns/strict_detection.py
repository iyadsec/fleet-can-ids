"""Strict per-run detection check applied identically to June baseline and validation runs (read-only)."""
import glob
import os
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent / "outputs"
JUNE_RUNS = Path(os.environ["FLEET_GUARD_CODE_ROOT"]) / ("new_experiments/final_end_to_end_publication_run_balanced/results/scenario_evaluation/runs")


def strict(run_dir):
    e = pd.read_csv(run_dir / "event_predictions.csv")
    s = pd.read_csv(run_dir / "source_windows.csv").set_index("event_id")
    acc = e[e.final_decision == "coordinated_attack"]
    valid = benign_only = 0
    clusters = acc.groupby("cluster_id")
    for _, g in clusters:
        if g.empty:
            continue
        m = s.loc[g.event_id]
        mal = m[(m.ground_truth_malicious == 1) & (m.scenario_role == "coordinated")]
        if mal.vehicle_token.nunique() >= 2:
            valid += 1
        elif m.ground_truth_malicious.sum() == 0:
            benign_only += 1
    return valid, benign_only


rows = []
for d in sorted(JUNE_RUNS.iterdir()):
    mt = re.match(r"(strong_campaign|weak_campaign)_cs(\d+)_seed(\d+)_20260614T20(?!13)", d.name)
    if mt:
        v, b = strict(d)
        rows.append(dict(condition="June_baseline", scenario_key=mt[1], campaign_size=int(mt[2]), seed=int(mt[3]), valid=v, benign_only=b))
for d in sorted((HERE / "runs").iterdir()):
    mt = re.match(r"(\w+?)__(strong_campaign|weak_campaign)_cs(\d+)_seed(\d+)$", d.name)
    v, b = strict(d)
    rows.append(dict(condition=mt[1], scenario_key=mt[2], campaign_size=int(mt[3]), seed=int(mt[4]), valid=v, benign_only=b))
df = pd.DataFrame(rows)
df["valid_detection"] = (df.valid > 0).astype(float)
df["benign_only_alert"] = (df.benign_only > 0).astype(float)
print(df.groupby("condition").size())
out = df.groupby(["scenario_key", "campaign_size", "condition"])[["valid_detection", "benign_only_alert"]].mean().round(2).unstack("condition")
print(out.to_string())
df.to_csv(HERE / "strict_detection_per_run.csv", index=False)
out.to_csv(HERE / "strict_detection_summary.csv")
