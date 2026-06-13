"""Compare original and corrected Phase 4 results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_comparison_report(
    orig_root: Path,
    corr_root: Path,
    *,
    audit_summary: dict,
    fleet_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> None:
    orig_fleet = pd.read_csv(orig_root / "results/fleet_campaign_metrics.csv") if (orig_root / "results/fleet_campaign_metrics.csv").exists() else pd.DataFrame()
    orig_sim = pd.read_csv(orig_root / "results/descriptor_similarity.csv") if (orig_root / "results/descriptor_similarity.csv").exists() else pd.DataFrame()

    lines = [
        "# Original vs corrected Phase 4",
        "",
        "## 1. Benign source counts",
        "",
        "| Model | Original test benign (audit) | Corrected test benign |",
        "|-------|------------------------------|----------------------|",
        f"| Hyundai | 4,934 | {audit_summary.get('corrected_test_benign', {}).get('Hyundai', {}).get('test_benign_descriptors', 'N/A')} |",
        "| Kia | 0 | "
        f"{audit_summary.get('corrected_test_benign', {}).get('Kia', {}).get('test_benign_descriptors', 'N/A')} |",
        "| Chevrolet | 0 | "
        f"{audit_summary.get('corrected_test_benign', {}).get('Chevrolet', {}).get('test_benign_descriptors', 'N/A')} |",
        "",
        "## 2. Benign platform composition",
        "",
        "- Original: homogeneous Hyundai-only benign background (15 Hyundai benign instances).",
        "- Corrected: fixed 5 Hyundai + 5 Kia + 5 Chevrolet benign instances.",
        "",
        "## 3. Campaign metrics (C3, strong)",
        "",
    ]

    if not orig_fleet.empty and not fleet_df.empty:
        for dl in sorted(fleet_df["diversity_level"].dropna().unique()):
            om = orig_fleet[(orig_fleet.framework_config == "C3") & (orig_fleet.attack_strength == "strong") & (orig_fleet.diversity_level == dl)]
            cm = fleet_df[(fleet_df.framework_config == "C3") & (fleet_df.attack_strength == "strong") & (fleet_df.diversity_level == dl)]
            lines.append(
                f"- D{int(dl)}: campaign F1 original={om['campaign_f1'].mean():.3f}, corrected={cm['campaign_f1'].mean():.3f}; "
                f"false campaign rate original={om['false_campaign_alert_rate'].mean():.3f}, corrected={cm['false_campaign_alert_rate'].mean():.3f}"
            )

    lines += ["", "## 4. Descriptor similarity", ""]
    if not orig_sim.empty and not sim_df.empty:
        ob = orig_sim["benign_cross_model_similarity"].mean()
        cb = sim_df["benign_cross_model_similarity"].mean()
        lines.append(f"- Mean cross-model benign similarity: original={ob:.3f}, corrected={cb:.3f}")

    lines += [
        "",
        "## 5. Statistical conclusions",
        "",
    ]
    if not stats_df.empty:
        sig = stats_df[stats_df.get("significant", False) == True]
        lines.append(f"- Corrected paired tests with Holm correction: {len(sig)} significant comparisons in families A–C.")
    else:
        lines.append("- See `results/statistical_tests.csv`.")

    lines += [
        "",
        "## 6. Superseded conclusions",
        "",
        "The following original Phase 4 conclusions are **superseded**:",
        "",
        "1. Kia and Chevrolet had no usable benign descriptors — they existed but were excluded from test by global benign split.",
        "2. Homogeneous Hyundai-only benign fleet background — corrected run uses heterogeneous 5/5/5 composition.",
        "3. False-campaign rates measured without cross-platform benign diversity — corrected rates reflect heterogeneous benign fleet.",
        "",
        "## 7. Conclusions that remain valid",
        "",
        "- Weak D3 unsupported (no weak Chevrolet malicious descriptors).",
        "- Local IDS metrics unchanged (same descriptors, same IF model).",
        "- C1 campaign metrics remain N/A.",
        "- Hierarchical separation between local detection and fleet correlation preserved.",
    ]

    (corr_root / "comparison/original_vs_corrected_phase4.md").write_text("\n".join(lines), encoding="utf-8")
