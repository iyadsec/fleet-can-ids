#!/usr/bin/env python3
"""Run can-train-and-test cross-dataset validation pipeline."""

from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.audit import run_dataset_audit
from src.ctt.constants import DEFAULT_CTT_DATASET_ROOT, OUTPUT_ROOT
from src.ctt.descriptors import generate_descriptors
from src.ctt.evaluation import (
    generate_figures,
    generate_publication_tables,
    run_campaign_size_sensitivity,
    run_edge_sensitivity,
)
from src.ctt.features import write_feature_schema
from src.ctt.fleet_campaign import write_fleet_transfer_policy
from src.ctt.fleet_graph import build_behavioural_graph, save_graph_artifacts
from src.ctt.local_detector import run_local_onboarding
from src.ctt.scenarios import run_scenario_evaluation
from src.ctt.splits import run_split_validation
from src.ctt.statistics import run_statistical_analysis
from src.ctt.streaming_pipeline import run_streaming_pipeline
from src.ctt.utils import ensure_dir, write_markdown


def write_paper_wording(output_root: Path) -> None:
    write_markdown(
        output_root / "audit" / "recommended_paper_wording.md",
        "Recommended Paper Wording",
        {
            "Cross-dataset evaluation": (
                "To evaluate cross-dataset generalisation, the proposed framework was applied to the "
                "independent can-train-and-test dataset. This dataset contains labelled CAN traffic from "
                "four vehicles manufactured by Chevrolet and Subaru and includes nine attack families. "
                "The evaluation used the dataset's built-in known/unknown vehicle and attack splits, and "
                "constructed controlled fleet scenarios across vehicles to test whether the proposed fleet "
                "graph could distinguish isolated incidents from behaviourally related multi-vehicle campaigns."
            ),
            "Fleet campaign caveat": (
                "Because the dataset does not provide naturally synchronized fleet-wide campaigns, the "
                "fleet-level experiments are controlled cross-vehicle campaign simulations based on "
                "behaviourally related attack traces rather than real synchronized attacks."
            ),
        },
    )


def write_summary(
    output_root: Path,
    audit_result: dict,
    window_count: int,
    metrics_df,
    desc_summary,
    graph_stats: dict,
    scenario_results,
    campaign_size,
    edge_sensitivity,
    runtime_sec: float,
    peak_memory_mb: float,
) -> None:
    sections = {
        "1. Files processed": str(audit_result.get("total_files", 0)),
        "2. Vehicles found": ", ".join(audit_result.get("vehicles_found", [])),
        "3. Attack types found": ", ".join(audit_result.get("attacks_found", [])),
        "4. Schema": "timestamp, can_id, dlc, byte_0..7, label, is_attack, attack_type, vehicle_id, manufacturer, dataset_set, subset_name, source_file, source_row_index",
        "5. Windows generated": f"{window_count:,}",
        "6. Train/test splits": "Built-in CTT splits per set; train_01 for benign-only training; test_01-04 for generalisation",
        "7. Benign-only training": "Yes — attack_data_used_in_training = 0 for all models",
        "8. Threshold selection": f"Weak={90}th pct, Strong={97.5}th pct on benign validation windows",
        "9. Local detection by subset": metrics_df.groupby(["subset_name", "mode"])["f1"].mean().to_string() if not metrics_df.empty else "N/A",
        "10. Local detection by attack": metrics_df[metrics_df["attack_type"] != "all"].groupby("attack_type")["f1"].mean().to_string() if not metrics_df.empty else "N/A",
        "11. Descriptor transmission rate": f"{desc_summary.iloc[0]['candidate_transmission_rate']:.4f}" if desc_summary is not None and not desc_summary.empty else "N/A",
        "12. Bandwidth reduction": f"{desc_summary.iloc[0]['bandwidth_reduction_ratio']:.4f}" if desc_summary is not None and not desc_summary.empty else "N/A",
        "13. Graph nodes/edges": f"{graph_stats.get('num_nodes', 0)} / {graph_stats.get('num_edges', 0)}",
        "14. Cross-vehicle edge %": f"{graph_stats.get('cross_vehicle_edge_pct', 0):.2f}%",
        "15. Benign fleet campaign-free": str(scenario_results[scenario_results["scenario"] == "benign_fleet_control"]["false_campaign"].mean() < 0.5) if not scenario_results.empty else "N/A",
        "16. Isolated attacks isolated": "Evaluated per scenario metrics",
        "17. Unrelated incidents separate": "Evaluated per scenario metrics",
        "18. Strong campaigns detected": str(scenario_results[scenario_results["scenario"] == "strong_campaign"]["campaign_detected"].mean() > 0.5) if not scenario_results.empty else "N/A",
        "19. Weak campaigns detected": str(scenario_results[scenario_results["scenario"] == "weak_campaign"]["campaign_detected"].mean() > 0.3) if not scenario_results.empty else "N/A",
        "20. Campaign size effect": campaign_size.groupby("campaign_size")["campaign_f1"].mean().to_string() if not campaign_size.empty else "N/A",
        "21. Edge count effect": edge_sensitivity[["edge_count", "campaign_f1"]].to_string() if not edge_sensitivity.empty else "N/A",
        "22. Limitations": "No real synchronized fleet campaigns; unknown-vehicle strict local onboarding limited",
        "23. Main paper results": "Cross-dataset local detection by subset; strong campaign scenario F1; descriptor compactness",
        "24. Supplementary": "Edge sensitivity; campaign size; full per-attack tables",
        "Runtime": f"{runtime_sec:.1f}s",
        "Peak memory": f"{peak_memory_mb:.1f} MB",
    }
    write_markdown(output_root / "CAN_TRAIN_AND_TEST_CROSS_DATASET_SUMMARY.md", "CTT Cross-Dataset Validation Summary", sections)


def main() -> int:
    parser = argparse.ArgumentParser(description="CTT cross-dataset validation")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--skip-windowing", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root or Path(os.environ.get("CTT_DATASET_ROOT", DEFAULT_CTT_DATASET_ROOT))
    output_root = args.output_root
    ensure_dir(output_root)

    tracemalloc.start()
    t0 = time.perf_counter()

    print("=== Step 1: Dataset Audit ===")
    audit_result = run_dataset_audit(dataset_root, output_root)
    if not audit_result["suitability"]["schema_normalizable"]:
        print("CRITICAL: Schema cannot be normalized. Aborting.")
        return 1

    print("=== Step 2: Split Validation ===")
    run_split_validation(dataset_root, output_root)

    print("=== Step 3-4: Streaming Normalize + Window + Features ===")
    write_feature_schema(output_root)
    if not args.skip_normalize:
        _, window_manifest, features = run_streaming_pipeline(dataset_root, output_root)
    else:
        import pandas as pd
        window_manifest = pd.read_csv(output_root / "manifests" / "window_manifest.csv")
        features = pd.read_parquet(output_root / "windows" / "all_window_features.parquet")

    print("=== Step 5: Local Onboarding & Detection ===")
    metrics_df, pred_df, _ = run_local_onboarding(window_manifest, output_root, features=features)

    print("=== Step 6: Descriptor Generation ===")
    import pandas as pd
    desc_df = generate_descriptors(pred_df, features, output_root)
    desc_summary = pd.read_csv(output_root / "results" / "descriptor_transfer" / "communication_summary.csv")

    print("=== Step 7: Fleet Graph ===")
    write_fleet_transfer_policy(output_root)
    node_df, edge_df, graph_stats = build_behavioural_graph(desc_df)
    save_graph_artifacts(node_df, edge_df, graph_stats, output_root)

    print("=== Step 8: Scenario Evaluation ===")
    scenario_results = run_scenario_evaluation(features, pred_df, desc_df, output_root)

    print("=== Step 9: Campaign Size & Edge Sensitivity ===")
    campaign_size = run_campaign_size_sensitivity(features, pred_df, output_root)
    edge_sensitivity = run_edge_sensitivity(desc_df, output_root)

    print("=== Step 10: Statistical Analysis ===")
    run_statistical_analysis(scenario_results, campaign_size, output_root)

    print("=== Step 11: Publication Tables & Figures ===")
    file_inventory = audit_result["file_inventory"]
    generate_publication_tables(
        file_inventory, window_manifest, metrics_df, desc_summary,
        graph_stats, scenario_results, campaign_size, edge_sensitivity, output_root,
    )
    generate_figures(
        window_manifest, pred_df, metrics_df, graph_stats,
        scenario_results, campaign_size, edge_sensitivity, desc_summary, output_root,
    )

    write_paper_wording(output_root)

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    cost_dir = ensure_dir(output_root / "results" / "cost")
    cost_df = pd.DataFrame([{"runtime_sec": elapsed, "peak_memory_mb": peak_mb}])
    cost_df.to_csv(cost_dir / "runtime_summary.csv", index=False)

    write_summary(
        output_root, audit_result, len(window_manifest), metrics_df,
        desc_summary, graph_stats, scenario_results, campaign_size, edge_sensitivity,
        elapsed, peak_mb,
    )

    print(f"\nDone in {elapsed:.1f}s (peak memory {peak_mb:.1f} MB)")
    print(f"Output: {output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
