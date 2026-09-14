#!/usr/bin/env python3
"""Controlled Reviewer-A ablation runner (M1–M4) with shared frozen scenarios."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXP))

from metrics import campaign_metrics, local_detection_metrics  # noqa: E402
from methods import (  # noqa: E402
    build_shared_graph,
    run_m1_local,
    run_m2_descriptor,
    run_m3_gcn,
    run_m4_graphsage,
    set_all_seeds,
)
from scenario_builder import (  # noqa: E402
    build_scenario,
    freeze_scenario,
    load_frozen_scenario,
    pack_frozen,
)


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def env_fingerprint() -> dict[str, Any]:
    import sklearn
    import torch

    try:
        import torch_geometric as pyg

        pyg_v = pyg.__version__
    except Exception:
        pyg_v = "unknown"
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unknown"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_geometric": pyg_v,
        "git_commit": commit,
    }


def evaluate_method(
    *,
    method: str,
    scenario: str,
    frozen: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method, "scenario": scenario}
    row.update(local_detection_metrics(frozen["is_attack"], result["local_alert"]))
    if not result.get("produces_campaigns", True):
        row.update(
            {
                "campaign_precision": np.nan,
                "campaign_recall": np.nan,
                "campaign_f1": np.nan,
                "membership_f1": np.nan,
                "fragmentation": np.nan,
                "incorrect_merge_rate": np.nan,
                "false_campaign_rate": np.nan,
                "n_gt_campaigns": int(
                    len(set(frozen["gt_campaign_id"].tolist()) - {-1})
                ),
                "n_pred_campaigns": 0,
                "n_matched_campaigns": 0,
                "campaign_metrics_status": "N/A_no_campaign_reconstruction",
            }
        )
    else:
        cm = campaign_metrics(
            frozen["gt_campaign_id"],
            result["pred_campaign_id"],
            frozen["vehicles"],
            is_unrelated=(scenario == "unrelated_incidents"),
        )
        row.update(cm)
        row["campaign_metrics_status"] = "computed"
    return row


def run_one_seed(
    cfg: dict[str, Any],
    seed: int,
    cache_dir: Path,
    verify_lines: list[str],
) -> list[dict[str, Any]]:
    set_all_seeds(seed)
    fleet = cfg["fleet"]
    gcfg = cfg["graph"]
    rows: list[dict[str, Any]] = []

    for scenario in cfg["scenarios"]:
        df = build_scenario(
            scenario,
            seed,
            n_vehicles=int(fleet["n_vehicles"]),
            windows_per_vehicle=int(fleet["windows_per_vehicle"]),
            campaign_size=int(fleet["campaign_size"]),
        )
        freeze_scenario(df, cache_dir)
        df = load_frozen_scenario(cache_dir, scenario, seed)
        frozen = pack_frozen(df)

        shared_graph = build_shared_graph(
            frozen["X"],
            frozen["vehicles"],
            frozen["meta"],
            similarity_threshold=float(gcfg["similarity_threshold"]),
            max_same_vehicle_neighbors=int(gcfg["max_same_vehicle_neighbors"]),
            max_cross_vehicle_neighbors=int(gcfg["max_cross_vehicle_neighbors"]),
            seed=seed,
        )
        shared_graph_b = build_shared_graph(
            frozen["X"],
            frozen["vehicles"],
            frozen["meta"],
            similarity_threshold=float(gcfg["similarity_threshold"]),
            max_same_vehicle_neighbors=int(gcfg["max_same_vehicle_neighbors"]),
            max_cross_vehicle_neighbors=int(gcfg["max_cross_vehicle_neighbors"]),
            seed=seed,
        )
        assert shared_graph["edge_sig"] == shared_graph_b["edge_sig"], (
            "M3/M4 graph edge sets differ under identical inputs"
        )

        m1 = run_m1_local(frozen, cfg)
        m2 = run_m2_descriptor(frozen, cfg, seed)
        m3 = run_m3_gcn(frozen, cfg, seed, shared_graph)
        m4 = run_m4_graphsage(frozen, cfg, seed, shared_graph)

        assert frozen["descriptor_ids"] == df["event_id"].astype(str).tolist()
        assert len(m2["pred_campaign_id"]) == len(frozen["descriptor_ids"])
        assert len(m3["pred_campaign_id"]) == len(frozen["descriptor_ids"])
        assert len(m4["pred_campaign_id"]) == len(frozen["descriptor_ids"])
        assert m3["edge_sig"] == m4["edge_sig"] == shared_graph["edge_sig"]

        verify_lines.append(
            f"PASS seed={seed} scenario={scenario} nodes={len(df)} "
            f"edges={len(shared_graph['edge_sig'])} "
            f"shared_ids=OK graph_m3==graph_m4 "
            f"gt_campaigns={sorted(set(frozen['gt_campaign_id'].tolist()) - {-1})}"
        )

        method_results = [
            ("M1_local_IF", m1),
            ("M2_descriptor_clustering", m2),
            ("M3_GCN", m3),
            ("M4_GraphSAGE", m4),
        ]
        for method_name, result in method_results:
            metrics = evaluate_method(
                method=method_name,
                scenario=scenario,
                frozen=frozen,
                result=result,
            )
            metrics["seed"] = seed
            metrics["n_nodes"] = int(len(df))
            metrics["n_graph_edges"] = int(len(shared_graph["edge_sig"]))
            rows.append(metrics)

            pred_df = frozen["meta"][
                ["event_id", "vehicle_id", "is_attack", "gt_campaign_id"]
            ].copy()
            pred_df["pred_campaign_id"] = result["pred_campaign_id"]
            pred_df["local_alert"] = result["local_alert"].astype(int)
            pred_df["method"] = method_name
            pred_df.to_csv(
                cache_dir / f"pred_{method_name}_{scenario}_seed{seed}.csv",
                index=False,
            )

    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "campaign_f1",
        "membership_f1",
        "campaign_precision",
        "campaign_recall",
        "fragmentation",
        "incorrect_merge_rate",
        "false_campaign_rate",
        "local_f1",
        "local_precision",
        "local_recall",
        "n_matched_campaigns",
    ]
    rows = []
    for method in raw["method"].unique():
        for scenario in raw["scenario"].unique():
            sub = raw[(raw["method"] == method) & (raw["scenario"] == scenario)]
            rec: dict[str, Any] = {
                "method": method,
                "scenario": scenario,
                "n_seeds": len(sub),
            }
            for col in metric_cols:
                vals = pd.to_numeric(sub[col], errors="coerce")
                rec[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
                rec[f"{col}_std"] = (
                    float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
                )
            rows.append(rec)
    return pd.DataFrame(rows)


def ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    methods = [
        "M1_local_IF",
        "M2_descriptor_clustering",
        "M3_GCN",
        "M4_GraphSAGE",
    ]

    def cell(method: str, scenario: str, metric: str) -> str:
        if method == "M1_local_IF" and metric in {
            "campaign_f1",
            "membership_f1",
            "incorrect_merge_rate",
        }:
            return "N/A"
        sub = summary[
            (summary["method"] == method) & (summary["scenario"] == scenario)
        ]
        if sub.empty:
            return "N/A"
        mean = sub.iloc[0][f"{metric}_mean"]
        std = sub.iloc[0][f"{metric}_std"]
        if pd.isna(mean):
            return "N/A"
        if pd.isna(std):
            return f"{mean:.3f}"
        return f"{mean:.3f} ± {std:.3f}"

    rows = []
    for method in methods:
        rows.append(
            {
                "method": method,
                "strong_campaign_f1": cell(method, "strong_campaign", "campaign_f1"),
                "weak_campaign_f1": cell(method, "weak_campaign", "campaign_f1"),
                "strong_membership_f1": cell(
                    method, "strong_campaign", "membership_f1"
                ),
                "weak_membership_f1": cell(method, "weak_campaign", "membership_f1"),
                "unrelated_incorrect_merge_rate": cell(
                    method, "unrelated_incidents", "incorrect_merge_rate"
                ),
                "strong_local_f1": cell(method, "strong_campaign", "local_f1"),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=EXP / "config.yaml")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run seed 11 only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds or list(cfg["seeds"])
    if args.smoke:
        seeds = [11]

    cache_dir = EXP / "scenario_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = env_fingerprint()
    (EXP / "provenance.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )

    verify_lines = [
        f"# Verification report {fingerprint['timestamp_utc']}",
        f"git_commit={fingerprint['git_commit']}",
        f"python={fingerprint['python']} torch={fingerprint['torch']} "
        f"pyg={fingerprint['torch_geometric']} sklearn={fingerprint['sklearn']}",
    ]
    all_rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"=== seed {seed} ===", flush=True)
        all_rows.extend(run_one_seed(cfg, seed, cache_dir, verify_lines))

    raw = pd.DataFrame(all_rows)
    raw.to_csv(EXP / "raw_results.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(EXP / "summary_results.csv", index=False)
    table = ablation_table(summary)
    table.to_csv(EXP / "ablation_table.csv", index=False)
    (EXP / "verification_report.txt").write_text(
        "\n".join(verify_lines) + "\n", encoding="utf-8"
    )
    print(table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
