"""Statistical tests for design comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.shared_statistics import holm_adjust, paired_test


def run_design_comparisons(all_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = [
        ("A_D0_vs_D1", "D0", "D1", ["campaign_precision", "campaign_f1", "membership_precision", "benign_vehicles_included", "fragmentation_rate"]),
        ("B_D1_vs_D2", "D1", "D2", ["campaign_f1", "membership_f1", "fragmentation_rate", "incorrect_merging_rate", "end_to_end_latency"]),
        ("C_D2_size", "D2", "D2", []),
    ]
    for family, da, db, metrics in families:
        if family == "C_D2_size":
            sub = all_df[all_df["design"] == "D2"]
            pairs = [(2, 5), (5, 10), (2, 10)]
            for s_a, s_b in pairs:
                for metric in ["campaign_f1", "membership_precision", "fragmentation_rate"]:
                    a = sub[sub["campaign_size"] == s_a].sort_values("seed")[metric].to_numpy()
                    b = sub[sub["campaign_size"] == s_b].sort_values("seed")[metric].to_numpy()
                    n = min(len(a), len(b))
                    if n < 3:
                        continue
                    p, test, eff, sig = paired_test(a[:n], b[:n])
                    rows.append({"family": family, "comparison": f"{s_a}_vs_{s_b}", "metric": metric,
                                 "paired_seeds": n, "test": test, "mean_difference": float(np.mean(a[:n]-b[:n])),
                                 "p_value_raw": p, "effect_size": eff, "significant": sig})
            continue
        for cs in [2, 5, 10]:
            for metric in metrics:
                if metric not in all_df.columns:
                    continue
                a = all_df[(all_df["design"] == da) & (all_df["campaign_size"] == cs)].sort_values("seed")[metric].to_numpy()
                b = all_df[(all_df["design"] == db) & (all_df["campaign_size"] == cs)].sort_values("seed")[metric].to_numpy()
                n = min(len(a), len(b))
                if n < 3:
                    continue
                p, test, eff, sig = paired_test(a[:n], b[:n])
                rows.append({"family": family, "comparison": f"cs{cs}_{da}_vs_{db}", "metric": metric,
                             "paired_seeds": n, "test": test, "mean_difference": float(np.mean(a[:n]-b[:n])),
                             "p_value_raw": p, "effect_size": eff, "significant": sig})
    if rows:
        adj = holm_adjust([r["p_value_raw"] for r in rows])
        for i, r in enumerate(rows):
            r["p_value_holm"] = adj[i]
            r["significant"] = bool(r["significant"] and adj[i] < 0.05)
    return pd.DataFrame(rows)
