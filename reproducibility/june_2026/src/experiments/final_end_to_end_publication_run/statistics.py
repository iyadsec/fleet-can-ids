"""Statistical analysis for E2E publication run."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.experiments.shared_statistics import holm_adjust, paired_test


def run_primary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if df.empty or "test_condition" not in df.columns:
        return pd.DataFrame()

    families = [
        ("strong_campaign_size", "Strong Coordinated Campaign", [(2, 5), (5, 10), (2, 10)]),
        ("weak_campaign_size", "Weak Coordinated Campaign", [(2, 5), (5, 10), (2, 10)]),
    ]
    metrics = ["campaign_f1", "membership_precision", "membership_recall", "fragmentation", "end_to_end_latency"]

    for family, condition, pairs in families:
        sub = df[df["test_condition"] == condition]
        for s_a, s_b in pairs:
            for metric in metrics:
                if metric not in sub.columns:
                    continue
                a = sub[sub["campaign_size"] == s_a].sort_values("seed")[metric].to_numpy()
                b = sub[sub["campaign_size"] == s_b].sort_values("seed")[metric].to_numpy()
                n = min(len(a), len(b))
                if n < 3:
                    continue
                if np.allclose(a[:n], b[:n]):
                    rows.append(
                        {
                            "family": family,
                            "comparison": f"{s_a}_vs_{s_b}",
                            "metric": metric,
                            "paired_seeds": n,
                            "test": "identical",
                            "mean_difference": 0.0,
                            "p_value_raw": 1.0,
                            "effect_size": 0.0,
                            "significant": False,
                        }
                    )
                    continue
                p, test, eff, sig = paired_test(a[:n], b[:n])
                rows.append(
                    {
                        "family": family,
                        "comparison": f"{s_a}_vs_{s_b}",
                        "metric": metric,
                        "paired_seeds": n,
                        "test": test,
                        "mean_difference": float(np.mean(a[:n] - b[:n])),
                        "p_value_raw": p,
                        "effect_size": eff,
                        "significant": sig,
                    }
                )

    if rows:
        adj = holm_adjust([r["p_value_raw"] for r in rows])
        for i, r in enumerate(rows):
            r["p_value_holm"] = adj[i]
            r["significant"] = bool(r.get("significant", False) and adj[i] < 0.05)
    out = pd.DataFrame(rows)
    return out
