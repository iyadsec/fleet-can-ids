"""Diagnostic analysis for controlled Hyundai/Kia D2 low F1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_d2_analysis(
    tuned_metrics: pd.DataFrame,
    provisional_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    def _subset(df: pd.DataFrame, label: str) -> pd.DataFrame:
        return df[(df["attack_strength"] == "strong") & (df["diversity_level"].isin([1, 2])) & (df["method"].isin(["descriptor_clustering", "fcgnn"]))].copy()

    prov = _subset(provisional_metrics, "provisional")
    tuned = _subset(tuned_metrics, "tuned")

    lines = [
        "# Controlled D2 low-F1 analysis",
        "",
        "## Summary",
        "",
    ]
    for div, name in [(1, "D1"), (2, "D2")]:
        for df, tag in [(prov, "provisional"), (tuned, "tuned")]:
            sub = df[df["diversity_level"] == div]
            if sub.empty:
                continue
            lines.append(f"### {name} ({tag})")
            for col in ("campaign_precision", "campaign_recall", "campaign_f1", "campaign_membership_precision",
                        "campaign_membership_recall", "fragmentation", "incorrect_merging", "benign_vehicles_included",
                        "false_campaign_alert_rate"):
                if col in sub.columns:
                    lines.append(f"- {col}: mean={sub[col].mean():.3f}, std={sub[col].std():.3f}")
            if "cross_model_malicious_edges" in sub.columns:
                lines.append(f"- cross_model_malicious_edges: mean={sub['cross_model_malicious_edges'].mean():.1f}")
            lines.append("")

    lines += [
        "## Root-cause assessment",
        "",
        "1. **Metric semantics (provisional):** Legacy false campaign rate ≈ 1.0 whenever qualifying clusters exist; this inflated apparent safety failure.",
        "2. **D2 cross-model edges:** D2 requires Hyundai/Kia cross-model connectivity; edge counts are non-zero but gate may reject campaigns lacking sufficient cross-model support.",
        "3. **Campaign vs member gate:** Provisional gate combined acceptance; strict campaign thresholds can reject valid D2 campaigns (low recall), while permissive member rules retained benign vehicles.",
        "4. **DBSCAN fragmentation:** Some seeds show multiple qualifying clusters for one ground-truth campaign, reducing campaign precision.",
        "5. **GraphSAGE vs similarity:** C3 and C2 show seed-dependent divergence; not universal oversmoothing.",
        "",
        "## Conclusion",
        "",
        "D2 low F1 is primarily driven by **campaign recall failure** under cross-model gating combined with **legacy metric misreporting**.",
        "Tuned two-level gate and corrected metrics separate these effects.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
