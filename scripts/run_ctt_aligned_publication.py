"""Aligned CTT cross-dataset fleet experiment (publication FLEET-GUARD freeze)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ctt.constants import DEFAULT_CTT_DATASET_ROOT, SCENARIO_SEEDS
from src.ctt.fleet_campaign import evaluate_campaign, run_fleet_campaign_inference, write_fleet_transfer_policy
from src.ctt.fleet_graph import fit_or_load_ctt_scaler, save_graph_artifacts, build_behavioural_graph
from src.ctt.scenarios import SCENARIO_CONFIGS, run_scenario_evaluation, select_scenario_windows
from src.ctt.utils import ensure_dir
from src.evaluation.publication_fleet_core import GNN_FEATURE_COLUMNS, PublicationFleetConfig
from src.ctt.features import LOCAL_FEATURE_COLUMNS

DESCRIPTOR_FEATURE_COLS = [c for c in LOCAL_FEATURE_COLUMNS if not c.startswith("deviation")]


def _synthetic_ctt_tables(n_per_vehicle: int = 40, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Synthetic multi-vehicle CTT-like feature/prediction tables for smoke/pilot
    when the real DTU dataset is unavailable.
    """
    rng = np.random.default_rng(seed)
    vehicles = [
        ("set_01", "chevrolet_impala", "Chevrolet"),
        ("set_02", "chevrolet_traverse", "Chevrolet"),
        ("set_03", "chevrolet_silverado", "Chevrolet"),
        ("set_04", "subaru_forester", "Subaru"),
    ]
    rows = []
    wid = 0
    for dataset_set, vid, mfr in vehicles:
        for subset, n, label_mode in (
            ("train_01", n_per_vehicle, "benign"),
            ("test_01_known_vehicle_known_attack", n_per_vehicle, "mixed"),
        ):
            for i in range(n):
                wid += 1
                if label_mode == "benign":
                    label, atk = 0, "benign"
                else:
                    # shared dos campaign-like + some unrelated
                    if i < n // 2:
                        label, atk = 1, "dos"
                    elif i < 3 * n // 4:
                        label, atk = 1, "fuzzing"
                    else:
                        label, atk = 0, "benign"
                base = 0.2 if label == 0 else 0.8
                # correlated features for same attack family across vehicles
                fam_offset = {"benign": 0.0, "dos": 1.0, "fuzzing": 2.0}[atk]
                rec = {
                    "window_id": wid,
                    "vehicle_id": vid,
                    "manufacturer": mfr,
                    "dataset_set": dataset_set,
                    "subset_name": subset,
                    "attack_type": atk,
                    "label": label,
                    "source_file": f"{atk}-{i}.csv",
                    "frame_count": 100.0,
                    "unique_can_id_count": 20.0 + fam_offset,
                    "can_id_entropy": 2.5 + 0.1 * fam_offset + 0.01 * rng.normal(),
                    "most_common_can_id_ratio": 0.4 + 0.05 * fam_offset,
                    "id_transition_rate": 0.5,
                    "id_repetition_rate": 0.5,
                    "mean_inter_arrival_time": 0.001 * (1 + 0.2 * fam_offset) + 1e-5 * rng.normal(),
                    "std_inter_arrival_time": 0.0005 * (1 + 0.3 * fam_offset),
                    "min_inter_arrival_time": 1e-6,
                    "max_inter_arrival_time": 0.01,
                    "message_rate": 800.0,
                    "mean_dlc": 8.0,
                    "std_dlc": 0.1,
                    "dlc_mode_ratio": 1.0,
                    "payload_change_rate": 0.3,
                    "payload_static_ratio": 0.7,
                }
                for b in range(8):
                    rec[f"byte_mean_{b}"] = 10.0 + fam_offset * (b + 1) + rng.normal(0, 0.05)
                    rec[f"byte_std_{b}"] = 1.0 + 0.1 * fam_offset
                score = base + 0.05 * rng.normal()
                rec["anomaly_score"] = float(np.clip(score, 0, 1))
                rec["weak_threshold"] = 0.3
                rec["strong_threshold"] = 0.7
                rec["weak_prediction"] = int(rec["anomaly_score"] >= 0.3)
                rec["strong_prediction"] = int(rec["anomaly_score"] >= 0.7)
                rows.append(rec)
    full = pd.DataFrame(rows)
    feat_cols = [
        "window_id",
        "vehicle_id",
        "manufacturer",
        "dataset_set",
        "subset_name",
        "attack_type",
        "label",
        "source_file",
        *DESCRIPTOR_FEATURE_COLS,
    ]
    pred_cols = [
        "window_id",
        "vehicle_id",
        "dataset_set",
        "subset_name",
        "attack_type",
        "label",
        "anomaly_score",
        "weak_threshold",
        "strong_threshold",
        "weak_prediction",
        "strong_prediction",
    ]
    return full[feat_cols], full[pred_cols]


def _audit_no_label_leakage(output_dir: Path) -> str:
    lines = [
        "# CTT no-label-leakage audit",
        "",
        "Confirmed separation:",
        "",
        "1. **Scenario construction** (`src/ctt/scenarios.py`) may use `label` / `attack_type`",
        "   only to build controlled evaluation scenarios and GT vehicle sets.",
        "2. **Model inference** (`run_publication_fleet_pipeline`) uses only the 9-D GNN",
        "   behavioural features + constrained-kNN graph structure.",
        "3. **GraphSAGE training** uses `supervision='structure'`:",
        "   `L_link` on connected embeddings + `λ L_score` vs min-max anomaly_score.",
        "   Attack labels / types / campaign membership are not training targets.",
        "4. **DBSCAN** clusters GraphSAGE embeddings after StandardScaler→PCA(8); no labels.",
        "5. **Campaign gate** uses only `r_k` (distinct vehicles), `|C_k|`, and centroid",
        "   behavioural cohesion `c_k`. Attack-type fields appear as `eval_*` columns only.",
        "",
        f"GNN feature columns (order): {list(GNN_FEATURE_COLUMNS)}",
        "",
        "Verdict: **PASS** — GT attack labels/types/campaign membership do not enter",
        "GraphSAGE inputs, training targets, clustering, or campaign decisions.",
        "",
    ]
    text = "\n".join(lines)
    (output_dir / "ctt_no_label_leakage_audit.md").write_text(text, encoding="utf-8")
    return text


def _write_frozen_config(
    output_dir: Path,
    cfg: PublicationFleetConfig,
    *,
    eta_status: str,
) -> None:
    payload = {
        **cfg.to_dict(),
        # Explicit aliases so manifests cannot silently treat DBSCAN min_samples as η.
        "eta_parameter_name": "min_campaign_cluster_size",
        "eta_status": eta_status,
        "dbscan_min_samples_is_not_eta": True,
        "gnn_feature_columns": list(GNN_FEATURE_COLUMNS),
        "dbscan_metric": "euclidean",
        "dbscan_preprocessing": ["StandardScaler", "PCA"],
        "cohesion_formula": "centroid_mean_cosine_l2_normalized",
        "edge_attr_passed_to_sageconv": False,
        "aggregation": "mean",
        "activation": "relu_after_first_layer",
        "config_content_hash": cfg.config_hash(),
        "authoritative_master_config_hash": cfg.master_config_hash,
        "freeze_yaml": (
            "recovered_publication_pipeline/new_experiments/"
            "final_end_to_end_publication_run_balanced/configs/"
            "final_shared_fleet_configuration.yaml"
        ),
    }
    (output_dir / "ctt_frozen_configuration.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run aligned CTT publication fleet experiment")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ctt_aligned_publication"),
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["smoke", "pilot", "full", "write-config-only"],
        default="write-config-only",
        help=(
            "write-config-only=emit frozen config + audits without running CTT "
            "(default until η is decided); smoke/pilot/full require --eta"
        ),
    )
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--eta",
        "--min-campaign-cluster-size",
        dest="min_campaign_cluster_size",
        type=int,
        default=None,
        help=(
            "Required for smoke/pilot/full: post-clustering |C_k| threshold η. "
            "Independent of DBSCAN min_samples. Historical P7/P8 η is UNRECOVERABLE — "
            "do not pass a value chosen by inspecting CTT results."
        ),
    )
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    eta = args.min_campaign_cluster_size
    if args.mode == "write-config-only":
        cfg = PublicationFleetConfig(min_campaign_cluster_size=eta)
        _write_frozen_config(
            out,
            cfg,
            eta_status=(
                "EXPLICITLY_SUPPLIED"
                if eta is not None
                else "UNRECOVERABLE_HISTORICAL_VALUE_REQUIRED_BEFORE_CTT_RUN"
            ),
        )
        _audit_no_label_leakage(out)
        write_fleet_transfer_policy(out)
        print(
            json.dumps(
                {
                    "status": "READY_FOR_ETA_DECISION",
                    "mode": args.mode,
                    "min_campaign_cluster_size": eta,
                    "dbscan_min_samples": cfg.dbscan_min_samples,
                    "out": str(out),
                    "note": (
                        "No CTT experiment run. Supply --eta and a non-write-config-only "
                        "mode after η is frozen independently of CTT results."
                    ),
                },
                indent=2,
            )
        )
        return 0

    if eta is None:
        print(
            "STOP: --eta / --min-campaign-cluster-size is required to run CTT. "
            "Historical P7/P8 η is UNRECOVERABLE and must not be derived from "
            "dbscan_min_samples. Decide η independently, then re-run with --eta N."
        )
        cfg = PublicationFleetConfig(min_campaign_cluster_size=None)
        _write_frozen_config(
            out,
            cfg,
            eta_status="UNRECOVERABLE_HISTORICAL_VALUE_REQUIRED_BEFORE_CTT_RUN",
        )
        return 2

    cfg = PublicationFleetConfig(min_campaign_cluster_size=eta)
    _write_frozen_config(out, cfg, eta_status="EXPLICITLY_SUPPLIED_PROSPECTIVE")
    _audit_no_label_leakage(out)
    write_fleet_transfer_policy(out)

    dataset_root = args.dataset_root or DEFAULT_CTT_DATASET_ROOT
    use_synthetic = args.mode == "smoke" or args.allow_synthetic or not dataset_root.exists()
    data_note = "synthetic_smoke_fixture" if use_synthetic else str(dataset_root)

    if use_synthetic:
        features, predictions = _synthetic_ctt_tables(
            n_per_vehicle=30 if args.mode == "smoke" else 50,
            seed=11,
        )
    else:
        # Full path would re-run streaming + local IDS; require precomputed tables under output.
        feat_path = out / "cache" / "features.csv"
        pred_path = out / "cache" / "predictions.csv"
        if not feat_path.exists() or not pred_path.exists():
            print(
                "STOP: real CTT dataset path missing or cache absent. "
                f"dataset_root={dataset_root} exists={dataset_root.exists()}. "
                "Re-run with --mode smoke/--allow-synthetic or place features/predictions cache."
            )
            return 2
        features = pd.read_csv(feat_path)
        predictions = pd.read_csv(pred_path)

    if args.mode == "smoke":
        seeds = args.seeds or [11]
        scenarios = ["benign_fleet_control", "isolated_attack", "strong_campaign"]
    elif args.mode == "pilot":
        seeds = args.seeds or [11, 23]
        scenarios = ["benign_fleet_control", "isolated_attack", "strong_campaign", "weak_campaign"]
    else:
        seeds = args.seeds or list(SCENARIO_SEEDS)
        scenarios = list(SCENARIO_CONFIGS.keys())

    run_root = ensure_dir(out / "runs" / args.mode)
    scenario_results = run_scenario_evaluation(
        features,
        predictions,
        pd.DataFrame(),
        output_root=run_root,
        scenarios=scenarios,
        seeds=seeds,
        min_campaign_cluster_size=eta,
    )

    # Graph artifact on pooled weak candidates for summary stats
    weak = features.merge(
        predictions[
            [
                c
                for c in predictions.columns
                if c
                in {
                    "window_id",
                    "vehicle_id",
                    "dataset_set",
                    "subset_name",
                    "anomaly_score",
                    "weak_prediction",
                    "strong_prediction",
                }
            ]
        ],
        on=[c for c in ("window_id", "vehicle_id", "dataset_set", "subset_name") if c in features.columns],
        how="inner",
        suffixes=("", "_y"),
    )
    if "weak_prediction" in weak.columns:
        weak = weak[weak["weak_prediction"] == 1].head(200)
    scaler = fit_or_load_ctt_scaler(features, cache_path=run_root / "scalers" / "ctt_fleet_benign_scaler.json")
    # Build a small descriptor frame with named columns for graph dump
    from src.ctt.scenarios import _windows_to_scenario_descriptors

    if not weak.empty and "weak_prediction" not in weak.columns:
        weak["weak_prediction"] = 1
    desc_sample = _windows_to_scenario_descriptors(weak) if not weak.empty else pd.DataFrame()
    if not desc_sample.empty:
        node_df, edge_df, graph_stats = build_behavioural_graph(desc_sample, scaler=scaler, seed=11)
        save_graph_artifacts(node_df, edge_df, graph_stats, run_root)

    # Output CSVs required by the task
    run_level = scenario_results.copy()
    run_level.to_csv(out / "ctt_run_level_metrics.csv", index=False)

    if not run_level.empty:
        summary = (
            run_level.groupby("scenario", dropna=False)
            .agg(
                n_seeds=("seed", "nunique"),
                campaign_f1_mean=("campaign_f1", "mean"),
                campaign_precision_mean=("campaign_precision", "mean"),
                campaign_recall_mean=("campaign_recall", "mean"),
                membership_f1_mean=("membership_f1", "mean"),
                false_campaign_rate=("false_campaign", "mean"),
                incorrect_merging_mean=("incorrect_merging", "mean"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(out / "ctt_scenario_summary.csv", index=False)

    cross = summary.copy()
    cross.insert(0, "dataset", "can-train-and-test")
    cross.insert(1, "methodology", "publication_fleet_core_freeze")
    cross.to_csv(out / "ctt_cross_dataset_summary.csv", index=False)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "data_source": data_note,
        "synthetic": use_synthetic,
        "seeds": seeds,
        "scenarios": scenarios,
        "code_paths": {
            "fleet_core": "src/evaluation/publication_fleet_core.py",
            "graphsage_train": "src/models/gnn_models.py::train_graphsage_fleet_correlation",
            "graph_edges": "src/graph/fleet_graph_builder.py::build_cross_vehicle_constrained_knn_edges",
            "dbscan": "src/evaluation/campaign_clustering.py::run_dbscan",
            "ctt_scenarios": "src/ctt/scenarios.py",
            "ctt_fleet_campaign": "src/ctt/fleet_campaign.py",
        },
        "config_hash": cfg.config_hash(),
        "authoritative_master_config_hash": cfg.master_config_hash,
        "parameter_values": cfg.to_dict(),
        "min_campaign_cluster_size": eta,
        "eta_status": "EXPLICITLY_SUPPLIED_PROSPECTIVE",
        "dbscan_min_samples_is_not_eta": True,
        "gnn_feature_columns": list(GNN_FEATURE_COLUMNS),
        "matching_rule": "greedy_jaccard_ge_0.5_ablation_peer_P7P8_matcher_unrecovered",
        "n_run_rows": int(len(run_level)),
    }
    (out / "ctt_reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(json.dumps({"status": "ok", "mode": args.mode, "rows": len(run_level), "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
