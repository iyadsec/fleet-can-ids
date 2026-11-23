"""Statistical analysis for final shared configuration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.shared_statistics import holm_adjust, paired_test, wilcoxon_paired_test


def run_campaign_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = [
        ("strong_size", "S3", [(2, 5), (5, 10), (2, 10)]),
        ("weak_size", "S4", [(2, 5), (5, 10), (2, 10)]),
        ("strong_vs_weak", "S3", []),
    ]
    metrics = ["campaign_f1", "membership_precision", "membership_recall", "fragmentation_rate",
               "benign_vehicles_included", "candidate_transmission_rate", "end_to_end_latency"]

    for family, sid, pairs in families:
        if family == "strong_vs_weak":
            for cs in [2, 5, 10]:
                for metric in metrics:
                    if metric not in df.columns:
                        continue
                    a = df[(df["scenario_id"] == "S3") & (df["campaign_size"] == cs)].sort_values("seed")[metric].to_numpy()
                    b = df[(df["scenario_id"] == "S4") & (df["campaign_size"] == cs)].sort_values("seed")[metric].to_numpy()
                    n = min(len(a), len(b))
                    if n < 3:
                        continue
                    p, test, eff, sig = paired_test(a[:n], b[:n])
                    if not np.allclose(a[:n], b[:n]) and np.std(a[:n] - b[:n], ddof=1) == 0:
                        p, test, eff, sig = wilcoxon_paired_test(a[:n], b[:n])
                    rows.append({
                        "family": family, "comparison": f"cs{cs}_S3_vs_S4", "metric": metric,
                        "paired_seeds": n, "test": test, "mean_difference": float(np.mean(a[:n] - b[:n])),
                        "p_value_raw": p, "effect_size": eff, "significant": sig,
                    })
            continue
        sub = df[df["scenario_id"] == sid]
        for s_a, s_b in pairs:
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                a = sub[sub["campaign_size"] == s_a].sort_values("seed")[metric].to_numpy()
                b = sub[sub["campaign_size"] == s_b].sort_values("seed")[metric].to_numpy()
                n = min(len(a), len(b))
                if n < 3:
                    continue
                p, test, eff, sig = paired_test(a[:n], b[:n])
                rows.append({
                    "family": family, "comparison": f"{s_a}_vs_{s_b}", "metric": metric,
                    "paired_seeds": n, "test": test, "mean_difference": float(np.mean(a[:n] - b[:n])),
                    "p_value_raw": p, "effect_size": eff, "significant": sig,
                })

    if rows:
        adj = holm_adjust([r["p_value_raw"] for r in rows])
        for i, r in enumerate(rows):
            r["p_value_holm"] = adj[i]
            r["significant"] = bool(r["significant"] and adj[i] < 0.05)
    return pd.DataFrame(rows)
