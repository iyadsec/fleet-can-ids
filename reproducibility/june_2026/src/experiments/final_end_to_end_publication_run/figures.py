"""Publication figures P1–P10."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_figures(out_dir: Path, *, predictions: pd.DataFrame, df: pd.DataFrame, descriptor_scale: pd.DataFrame | None = None, edge_df: pd.DataFrame | None = None) -> list[str]:
    out_dir = Path(out_dir)
    generated: list[str] = []
    test = predictions[predictions["split"] == "test"]

    if not test.empty and test["label"].nunique() > 1:
        y = test["label"].astype(int)
        scores = test["anomaly_score"].astype(float)
        prec, rec, _ = precision_recall_curve(y, scores)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rec, prec)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Vehicle-level precision-recall")
        _save(fig, out_dir / "figure_P1_vehicle_level_precision_recall")
        generated.append("P1")

        fig, ax = plt.subplots(figsize=(6, 4))
        for label, name in [(0, "benign"), (1, "malicious")]:
            sub = test[test["label"] == label]["anomaly_score"]
            ax.hist(sub, bins=30, alpha=0.6, label=name)
        ax.set_xlabel("Anomaly score")
        ax.set_ylabel("Count")
        ax.legend()
        _save(fig, out_dir / "figure_P2_anomaly_score_distribution")
        generated.append("P2")

    if descriptor_scale is not None and not descriptor_scale.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(descriptor_scale["fleet_size"], descriptor_scale["raw_can_mb_per_window_batch"], label="Raw CAN")
        ax.plot(descriptor_scale["fleet_size"], descriptor_scale["descriptor_mb_per_window_batch"], label="Descriptor")
        ax.set_xlabel("Fleet size")
        ax.set_ylabel("MB per batch")
        ax.legend()
        _save(fig, out_dir / "figure_P3_descriptor_bandwidth_scaling")
        generated.append("P3")

    if not df.empty and "test_condition" in df.columns:
        camp = df[df["test_condition"].str.contains("Campaign", na=False)]
        if not camp.empty:
            agg = camp.groupby(["test_condition", "campaign_size"])["campaign_f1"].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            for cond, sub in agg.groupby("test_condition"):
                ax.plot(sub["campaign_size"], sub["campaign_f1"], marker="o", label=cond.replace(" Coordinated Campaign", ""))
            ax.set_xlabel("Campaign size")
            ax.set_ylabel("Campaign F1")
            ax.legend()
            _save(fig, out_dir / "figure_P4_strong_vs_weak_campaign_F1")
            generated.append("P4")

            for metric, stem in [("campaign_precision", "P5"), ("campaign_recall", "P5"), ("membership_precision", "P6"), ("fragmentation", "P7")]:
                if metric not in camp.columns:
                    continue
            p5 = camp.groupby(["test_condition", "campaign_size"])[["campaign_precision", "campaign_recall"]].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            for cond, sub in p5.groupby("test_condition"):
                ax.plot(sub["campaign_size"], sub["campaign_precision"], marker="o", label=f"{cond} precision")
                ax.plot(sub["campaign_size"], sub["campaign_recall"], marker="x", linestyle="--", label=f"{cond} recall")
            ax.set_xlabel("Campaign size")
            ax.legend(fontsize=8)
            _save(fig, out_dir / "figure_P5_campaign_precision_recall_vs_size")
            generated.append("P5")

            p6 = camp.groupby(["test_condition", "campaign_size"])[["membership_precision", "membership_recall"]].mean().reset_index()
            fig, ax = plt.subplots(figsize=(6, 4))
            for cond, sub in p6.groupby("test_condition"):
                ax.plot(sub["campaign_size"], sub["membership_precision"], marker="o", label=cond)
            ax.set_xlabel("Campaign size")
            ax.set_ylabel("Membership precision")
            ax.legend(fontsize=8)
            _save(fig, out_dir / "figure_P6_membership_precision_recall_vs_size")
            generated.append("P6")

            frag_col = "fragmentation_rate" if "fragmentation_rate" in camp.columns else "fragmentation"
            if frag_col in camp.columns:
                p7 = camp.groupby(["test_condition", "campaign_size"])[frag_col].mean().reset_index()
                fig, ax = plt.subplots(figsize=(6, 4))
                for cond, sub in p7.groupby("test_condition"):
                    ax.plot(sub["campaign_size"], sub[frag_col], marker="o", label=cond)
                ax.set_xlabel("Campaign size")
                ax.set_ylabel("Fragmentation")
                ax.legend(fontsize=8)
                _save(fig, out_dir / "figure_P7_fragmentation_vs_campaign_size")
                generated.append("P7")

    if edge_df is not None and not edge_df.empty and "unique_edges" in edge_df.columns:
        for col, stem in [
            ("campaign_f1", "figure_P8_campaign_F1_vs_unique_edges"),
            ("false_campaign_rate", "figure_P9_false_campaign_rate_vs_unique_edges"),
            ("end_to_end_latency", "figure_P10_runtime_vs_unique_edges"),
        ]:
            if col not in edge_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            for cond, sub in edge_df.groupby("test_condition"):
                g = sub.groupby("unique_edges")[col].mean()
                ax.plot(g.index, g.values, marker="o", label=cond)
            ax.set_xlabel("Unique undirected edges")
            ax.set_ylabel(col)
            ax.legend(fontsize=8)
            _save(fig, out_dir / stem)
            generated.append(stem.split("_")[1])

    return generated
