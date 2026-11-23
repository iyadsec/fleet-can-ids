"""Summarise the 21 Nov 2026 S3/S4 validation against the June 2026 balanced baseline (read-only)."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent / "outputs"
JUNE = Path(os.environ["FLEET_GUARD_CODE_ROOT"]) / ("new_experiments/final_end_to_end_publication_run_balanced/results/scenario_evaluation/run_level_metrics.csv")

KEY = ["scenario_key", "campaign_size", "seed"]
CMP = ["campaign_f1", "membership_f1", "n_predicted_campaign_clusters", "benign_vehicles_included",
       "attacked_vehicles_correctly_included", "unique_edges", "cross_vehicle_edges", "candidate_nodes"]

june = pd.read_csv(JUNE)
june = june[june.scenario_key.isin(["strong_campaign", "weak_campaign"])].copy()
june["condition"] = "June_baseline"

ab = pd.read_csv(HERE / "run_level_metrics_A_reproduction_B_natural_C_partial_train.csv")
ab = ab[ab.condition.isin(["A_reproduction", "B_natural"])]
c = pd.read_csv(HERE / "run_level_metrics_C_partial_train.csv")
new = pd.concat([ab, c], ignore_index=True)
assert (new.status == "ok").all(), new[new.status != "ok"]

rep = new[new.condition == "A_reproduction"].set_index(KEY)[CMP].sort_index()
jb = june.set_index(KEY)[CMP].sort_index()
diff = (rep - jb.loc[rep.index]).abs()
print("Reproduction: runs", len(rep), "| max abs diff per metric:")
print(diff.max().to_string())
print("runs with any difference:", int((diff.sum(axis=1) > 1e-9).sum()))

allr = pd.concat([june, new[new.condition != "A_reproduction"]], ignore_index=True)
allr["detected"] = (allr["n_predicted_campaign_clusters"] >= 1).astype(float)
agg = (allr.groupby(["scenario_key", "condition", "campaign_size"])
       .agg(campaign_f1=("campaign_f1", "mean"), campaign_f1_sd=("campaign_f1", "std"),
            membership_f1=("membership_f1", "mean"), detection_rate=("detected", "mean"),
            mean_pred_clusters=("n_predicted_campaign_clusters", "mean"),
            benign_veh_included=("benign_vehicles_included", "mean"),
            attacked_included=("attacked_vehicles_correctly_included", "mean"))
       .round(3))
order = {"June_baseline": 0, "B_natural": 1, "C_partial_train": 2}
agg = agg.reset_index().sort_values(["scenario_key", "campaign_size", "condition"], key=lambda s: s.map(order) if s.name == "condition" else s)
print(agg.to_string(index=False))
agg.to_csv(HERE / "summary_by_condition.csv", index=False)

diag_cols = ["n_unique_malicious_24d_vectors", "mal_mean_pairwise_cosine_9d", "mal_cross_vehicle_pairs_cos_ge_0.95",
             "campaign_source_traces"]
d = new.groupby(["condition", "scenario_key", "campaign_size"])[diag_cols].mean().round(3)
print(d.to_string())
d.to_csv(HERE / "campaign_variation_diagnostics.csv")
print(new[new.condition == "B_natural"].groupby("scenario_key")["campaign_attack_types"].value_counts().to_string())

rows = []
for sk in ("strong_campaign", "weak_campaign"):
    for cond in ("B_natural", "C_partial_train"):
        for metric in ("campaign_f1", "membership_f1"):
            a = june[june.scenario_key == sk].set_index(["campaign_size", "seed"])[metric].sort_index()
            b = new[(new.condition == cond) & (new.scenario_key == sk)].set_index(["campaign_size", "seed"])[metric].sort_index()
            dd = (b - a.loc[b.index]).to_numpy()
            p = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
            rows.append({"scenario": sk, "condition": cond, "metric": metric, "june_mean": a.mean(),
                         "new_mean": b.mean(), "mean_delta": dd.mean(), "wilcoxon_p": p, "n_pairs": len(dd)})
t = pd.DataFrame(rows).round(4)
print(t.to_string(index=False))
t.to_csv(HERE / "paired_tests_vs_june.csv", index=False)
