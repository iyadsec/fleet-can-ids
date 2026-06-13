"""Statistical tests for Phase 4 model diversity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.experiments.evaluation_correction.statistics import format_p_value
from src.experiments.statistical_testing import _paired_test, holm_correction


def run_model_diversity_statistics(fleet_df: pd.DataFrame, *, alpha: float = 0.05) -> pd.DataFrame:
    sub = fleet_df[fleet_df["framework_config"].isin(["C2", "C3"])].copy()
    if sub.empty:
        return pd.DataFrame()

    families = {
        "A_strong": (sub[sub["attack_strength"] == "strong"], ["campaign_f1", "campaign_detection_rate", "membership_purity", "false_campaign_alert_rate"]),
        "B_weak": (sub[sub["attack_strength"] == "weak"], ["campaign_f1", "campaign_detection_rate", "false_campaign_alert_rate"]),
    }
    rows: list[dict] = []
    for family_name, (block, metrics) in families.items():
        if block.empty:
            continue
        for strength in sorted(block["attack_strength"].dropna().unique()):
            for dl in sorted(block["diversity_level"].dropna().unique()):
                grp = block[(block["attack_strength"] == strength) & (block["diversity_level"] == dl)]
                for metric in metrics:
                    if metric not in grp.columns:
                        continue
                    pivot = grp.pivot_table(index="seed", columns="framework_config", values=metric, aggfunc="first")
                    if "C3" not in pivot.columns or "C2" not in pivot.columns:
                        continue
                    diff = (pivot["C3"] - pivot["C2"]).dropna().to_numpy()
                    test, p, effect, etype = _paired_test(diff)
                    ci_low, ci_high = np.nan, np.nan
                    if len(diff) >= 2:
                        ci_low, ci_high = stats.t.interval(0.95, len(diff) - 1, loc=np.mean(diff), scale=stats.sem(diff))
                    rows.append(
                        {
                            "family": family_name,
                            "attack_strength": strength,
                            "diversity_level": int(dl),
                            "metric": metric,
                            "comparison": "C3 vs C2",
                            "paired_seeds": int(len(diff)),
                            "test": test,
                            "mean_difference": float(np.mean(diff)) if len(diff) else np.nan,
                            "ci95_low": float(ci_low),
                            "ci95_high": float(ci_high),
                            "raw_p_value": p,
                            "effect_size": effect,
                            "effect_size_type": etype,
                        }
                    )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["adjusted_p_value"] = holm_correction(out["raw_p_value"].fillna(1.0).tolist())
    out["significant"] = out["adjusted_p_value"] < alpha
    out["adjusted_p_value_formatted"] = out["adjusted_p_value"].map(format_p_value)
    out["ci95"] = out.apply(
        lambda r: f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
        if pd.notna(r["ci95_low"]) and pd.notna(r["ci95_high"])
        else "N/A",
        axis=1,
    )
    return out
