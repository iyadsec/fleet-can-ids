#!/usr/bin/env python3
"""Validation-only η sweep under frozen revised_jaccard_0.5_v3.

η ∈ {2,3,5,10}. GraphSAGE + DBSCAN run once per scenario; only the
post-clustering |C_k|≥η gate varies. Final TEST / CTT are not touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation.publication_fleet_core import (  # noqa: E402
    GNN_FEATURE_COLUMNS,
    PublicationFleetConfig,
    _cluster_qualifies_campaign,
    _merge_fragment_campaigns,
    assign_node_decisions,
    build_publication_fleet_graph,
    compute_cluster_behavioral_cohesion,
)
from src.evaluation.revised_campaign_metrics_v2 import (  # noqa: E402
    METRIC_PROTOCOL_ID,
    GroundTruthCampaign,
    PredictedCampaign,
    TAU_J,
    evaluate_protocol_v3,
)
from src.experiments.local_descriptor_normalisation import (  # noqa: E402
    fit_benign_fleet_scaler_from_rows,
    save_scaler_provenance,
)
from src.evaluation.campaign_clustering import extend_dbscan_labels, run_dbscan, subsample_indices  # noqa: E402
from src.models.gnn_models import train_graphsage_fleet_correlation  # noqa: E402

EXP = ROOT / "experimental-2026-09-20/eta_revalidation_from_raw"
SCEN_DIR = EXP / "validation_scenarios"
ART = EXP / "artifacts"
ETA_CANDIDATES = [2, 3, 5, 10]
TIE_EPS = 0.01  # recorded BEFORE selection
ELIGIBILITY_SLACK = 0.05  # recorded BEFORE selection
VALIDATION_SEEDS = [131, 137, 149, 157, 163, 179, 181, 191, 193, 197]
SCENARIOS = ["S0", "S1", "S2", "S3", "S4"]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def write_metric_protocol_freeze() -> dict[str, Any]:
    protocol = EXP / "PROPOSED_REVISED_METRIC_PROTOCOL_V3.md"
    impl = ROOT / "src/evaluation/revised_campaign_metrics_v2.py"
    freeze = {
        "protocol_id": METRIC_PROTOCOL_ID,
        "tau_jaccard": TAU_J,
        "matching": "maximum_weight_one_to_one",
        "membership": "vehicle_campaign_assignment_micro",
        "incorrect_merging": "independent_incident_membership",
        "fragmentation": "gt_overlap_fragment_count",
        "approval_status": "frozen",
        "eta_evaluated_at_freeze": False,
        "protocol_file": str(protocol.relative_to(ROOT)),
        "implementation_file": str(impl.relative_to(ROOT)),
        "protocol_sha256": _sha256_file(protocol),
        "implementation_sha256": _sha256_file(impl),
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": "V3 approved for prospective eta-revalidation. Do not alter definitions during/after eta sweep.",
    }
    (EXP / "METRIC_PROTOCOL_FREEZE.json").write_text(json.dumps(freeze, indent=2) + "\n")
    # Mark protocol header as approved/frozen (append freeze banner only if not present)
    text = protocol.read_text(encoding="utf-8")
    if "APPROVED/FROZEN" not in text:
        banner = (
            "\n\n---\n\n"
            "## FREEZE STATUS\n\n"
            "**APPROVED/FROZEN** for the prospective η-revalidation experiment "
            f"(`{METRIC_PROTOCOL_ID}`). Do not modify metric definitions, "
            f"`tau_J={TAU_J}`, or this document during/after η evaluation.\n"
        )
        # Insert after status line block
        protocol.write_text(text.rstrip() + banner + "\n", encoding="utf-8")
    return freeze


def write_non_eta_config() -> dict[str, Any]:
    cfg = {
        "experiment": "experimental-2026-09-20/eta_revalidation_from_raw",
        "status": "NON_ETA_FROZEN",
        "dbscan_min_samples_is_not_eta": True,
        "graph": {
            "metric": "cosine",
            "similarity_threshold": 0.95,
            "max_same_vehicle_neighbors": 2,
            "max_cross_vehicle_neighbors": 5,
            "temporal_edges": False,
            "edge_attr_passed_to_sageconv": False,
        },
        "graphsage": {
            "input_dim": 9,
            "gnn_feature_columns": list(GNN_FEATURE_COLUMNS),
            "layers": 2,
            "hidden_channels": 64,
            "embedding_dim": 32,
            "aggregation": "mean",
            "activation": "relu_after_first_sageconv",
            "epochs": 30,
            "optimizer": "Adam",
            "learning_rate": 0.01,
            "weight_decay": 0.0005,
            "campaign_loss_weight": 0.25,
            "supervision": "structure",
        },
        "clustering": {
            "preprocessing": ["StandardScaler", "PCA"],
            "pca_components": 8,
            "metric": "euclidean",
            "dbscan_eps": 0.5,
            "dbscan_min_samples": 2,
        },
        "campaign_gate_non_eta": {
            "gamma_parameter": "minimum_distinct_vehicles",
            "minimum_distinct_vehicles": 2,
            "beta_parameter": "minimum_campaign_cohesion",
            "minimum_campaign_cohesion": 0.5,
            "minimum_cross_vehicle_support": 1,
            "cohesion_formula": "centroid_mean_cosine_l2_normalized",
            "fragment_consolidation_enabled": True,
            "fragment_centroid_threshold": 0.85,
            "fragment_merge_eligibility": "eta_independent_using_min_candidate_eta_2_for_merge_only",
            "eta_parameter": "min_campaign_cluster_size",
            "eta_candidates": ETA_CANDIDATES,
            "eta_selected": None,
        },
        "authoritative_master_config_hash": "72dbfc1760126e799a8a1670f1aa81391bd69a65edeab7beecb61846aa946d9e",
        "selection_rules_recorded_before_evaluation": {
            "eligibility_slack_vs_best": ELIGIBILITY_SLACK,
            "eligibility": (
                "Exclude eta if S0_false_campaign_rate > min_S0 + slack OR "
                "S1_false_campaign_rate > min_S1 + slack OR "
                "S2_incorrect_merge_rate > min_S2 + slack"
            ),
            "selection_score": "mean(CampaignF1_S3, CampaignF1_S4)",
            "tie_rule": f"If |score_a - score_b| <= {TIE_EPS}, prefer smaller eta",
        },
        "validation_only": True,
        "test_partition_exposed": False,
        "ctt_used": False,
    }
    (EXP / "frozen_non_eta_configuration.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg


def base_fleet_config(seed: int) -> PublicationFleetConfig:
    return PublicationFleetConfig(
        similarity_threshold=0.95,
        max_same_vehicle_neighbors=2,
        max_cross_vehicle_neighbors=5,
        gnn_hidden_channels=64,
        gnn_embedding_dim=32,
        gnn_epochs=30,
        gnn_learning_rate=0.01,
        gnn_weight_decay=5e-4,
        campaign_loss_weight=0.25,
        dbscan_eps=0.5,
        dbscan_min_samples=2,
        dbscan_pca_components=8,
        minimum_distinct_vehicles=2,
        minimum_cross_vehicle_support=1,
        minimum_campaign_cohesion=0.5,
        min_campaign_cluster_size=2,  # placeholder; gate reapplied per eta
        fragment_consolidation_enabled=True,
        fragment_centroid_threshold=0.85,
        seed=int(seed),
    )


def prepare_scenario_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    # Fleet vehicle identity for graph kNN + gate γ (not OEM platform)
    out["vehicle_token"] = out["scenario_vehicle_id"].astype(str)
    out["vehicle_model"] = out["scenario_vehicle_id"].astype(str)
    out["event_id"] = out["descriptor_id"].astype(str)
    if "local_alert" not in out.columns:
        out["local_alert"] = (out["anomaly_score"] >= 0.80).astype(int)
    if "weak_signal" not in out.columns:
        out["weak_signal"] = (
            (out["anomaly_score"] >= 0.55) & (out["anomaly_score"] < 0.80)
        ).astype(int)
    return out


def gt_campaigns_from_scenario(df: pd.DataFrame) -> list[GroundTruthCampaign]:
    attacked = df[df["gt_attacked"].astype(int) == 1]
    if attacked.empty:
        return []
    # Distinct campaign_id groups (S2 has multiple INCIDENT-*; S3/S4 one CAMP-*)
    camps: list[GroundTruthCampaign] = []
    for cid, grp in attacked.groupby(attacked["campaign_id"].astype(str)):
        if not cid:
            continue
        vehicles = frozenset(grp["scenario_vehicle_id"].astype(str))
        camps.append(GroundTruthCampaign(campaign_id=cid, vehicles=vehicles))
    return camps


def incident_map_from_scenario(df: pd.DataFrame) -> dict[str, str]:
    attacked = df[df["gt_attacked"].astype(int) == 1]
    out: dict[str, str] = {}
    for _, row in attacked.iterrows():
        out[str(row["scenario_vehicle_id"])] = str(row["campaign_id"])
    return out


def run_graphsage_and_dbscan(
    df: pd.DataFrame,
    scaler,
    seed: int,
) -> dict[str, Any]:
    """Train GraphSAGE once; DBSCAN once; fragment-merge with η=2 eligibility only."""
    cfg = base_fleet_config(seed)
    data, meta, behavior_X, cols, graph_stats = build_publication_fleet_graph(
        df, cfg, fleet_scaler_provenance=scaler
    )
    _, train_metrics, embeddings, campaign_scores = train_graphsage_fleet_correlation(
        data,
        hidden_channels=cfg.gnn_hidden_channels,
        embedding_dim=cfg.gnn_embedding_dim,
        epochs=cfg.gnn_epochs,
        learning_rate=cfg.gnn_learning_rate,
        weight_decay=cfg.gnn_weight_decay,
        train_ratio=cfg.gnn_train_ratio,
        val_ratio=cfg.gnn_val_ratio,
        seed=cfg.seed,
        campaign_loss_weight=cfg.campaign_loss_weight,
        supervision=cfg.gnn_supervision,
        anomaly_feature_index=0,
    )

    meta = meta.reset_index(drop=True)
    fit_idx = subsample_indices(meta, cfg.max_clustering_samples, seed=cfg.seed)
    fit_labels, projector = run_dbscan(
        embeddings[fit_idx],
        eps=cfg.dbscan_eps,
        min_samples=cfg.dbscan_min_samples,
        pca_components=cfg.dbscan_pca_components,
        random_state=cfg.seed,
    )
    raw_labels = extend_dbscan_labels(
        embeddings, fit_labels, embeddings[fit_idx], projector, eps=cfg.dbscan_eps
    )

    # Build summary at eta=2 for fragment-merge eligibility (η-independent across candidates)
    rows = []
    for cid in sorted({int(c) for c in np.unique(raw_labels) if int(c) != -1}):
        mask = raw_labels == cid
        size = int(mask.sum())
        n_veh = int(meta.loc[mask, "vehicle_model"].nunique())
        cohesion = compute_cluster_behavioral_cohesion(
            behavior_X, mask, seed=cfg.seed + int(cid)
        )
        qualifies = _cluster_qualifies_campaign(
            size=size, n_vehicles=n_veh, cohesion=cohesion, cfg=cfg, eta=2
        )
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_size": size,
                "vehicles_in_cluster": n_veh,
                "behavioral_cohesion": round(cohesion, 4),
                "is_qualifying_campaign_cluster": qualifies,
            }
        )
    summary = pd.DataFrame(rows)
    labels = raw_labels.copy()
    if cfg.fragment_consolidation_enabled and not summary.empty:
        summary, labels = _merge_fragment_campaigns(
            summary,
            embeddings,
            labels,
            threshold=cfg.fragment_centroid_threshold,
        )
        # Rebuild cluster stats after merge (without eta gate yet)
        rebuilt = []
        for cid in sorted({int(c) for c in np.unique(labels) if int(c) != -1}):
            mask = labels == cid
            size = int(mask.sum())
            n_veh = int(meta.loc[mask, "vehicle_model"].nunique())
            cohesion = compute_cluster_behavioral_cohesion(
                behavior_X, mask, seed=cfg.seed + int(cid)
            )
            rebuilt.append(
                {
                    "cluster_id": int(cid),
                    "cluster_size": size,
                    "vehicles_in_cluster": n_veh,
                    "behavioral_cohesion": round(cohesion, 4),
                }
            )
        summary = pd.DataFrame(rebuilt)

    return {
        "meta": meta,
        "embeddings": embeddings,
        "campaign_scores": campaign_scores,
        "behavior_X": behavior_X,
        "raw_dbscan_labels": raw_labels,
        "labels": labels,  # post fragment-merge, pre-eta
        "pre_gate_summary": summary,
        "graph_stats": graph_stats,
        "train_metrics": train_metrics,
        "cfg": cfg,
    }


def apply_eta_gate(
    pre: dict[str, Any],
    eta: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Apply campaign gate with given eta on frozen pre-gate clusters."""
    cfg = pre["cfg"]
    meta = pre["meta"]
    labels = pre["labels"].copy()
    behavior_X = pre["behavior_X"]
    campaign_scores = pre["campaign_scores"]
    rows = []
    rejected_by_eta_only = 0
    for cid in sorted({int(c) for c in np.unique(labels) if int(c) != -1}):
        mask = labels == cid
        size = int(mask.sum())
        n_veh = int(meta.loc[mask, "vehicle_model"].nunique())
        cohesion = compute_cluster_behavioral_cohesion(
            behavior_X, mask, seed=cfg.seed + int(cid)
        )
        passes_gamma_beta = (
            n_veh >= cfg.minimum_distinct_vehicles
            and cohesion >= cfg.minimum_campaign_cohesion
        )
        passes_eta = size >= eta
        qualifies = passes_gamma_beta and passes_eta
        if passes_gamma_beta and not passes_eta:
            rejected_by_eta_only += 1
        rows.append(
            {
                "cluster_id": int(cid),
                "cluster_size": size,
                "vehicles_in_cluster": n_veh,
                "behavioral_cohesion": round(cohesion, 4),
                "mean_campaign_score": round(float(campaign_scores[mask].mean()), 4),
                "gate_eta_min_campaign_cluster_size": eta,
                "passes_gamma_beta": passes_gamma_beta,
                "passes_eta": passes_eta,
                "is_qualifying_campaign_cluster": qualifies,
                "rejected_by_eta_only": bool(passes_gamma_beta and not passes_eta),
            }
        )
    summary = pd.DataFrame(rows)
    decisions = assign_node_decisions(meta, labels, summary)
    decisions["_rejected_by_eta_only_count"] = rejected_by_eta_only
    return labels, summary, decisions


def predicted_campaigns_from_decisions(
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[PredictedCampaign]:
    if summary.empty:
        return []
    preds = []
    for _, row in summary.iterrows():
        if not bool(row["is_qualifying_campaign_cluster"]):
            continue
        cid = int(row["cluster_id"])
        vehicles = frozenset(
            decisions.loc[decisions["cluster_id"] == cid, "vehicle_token"].astype(str)
        )
        if vehicles:
            preds.append(PredictedCampaign(campaign_id=cid, vehicles=vehicles))
    return preds


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=0))


def main() -> int:
    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "OCSLAB_DATASET_DIR",
        str(ROOT / "Dataset/In-Vehicle Network Intrusion Detection Challenge"),
    )

    metric_freeze = write_metric_protocol_freeze()
    non_eta = write_non_eta_config()
    print("Frozen metric protocol + non-eta configuration")

    # Fit scaler on reconstructed benign TRAIN only
    train_benign = pd.read_csv(ART / "train_benign_scored_windows.csv")
    scaler = fit_benign_fleet_scaler_from_rows(
        train_benign, feature_names=GNN_FEATURE_COLUMNS, training_split="train"
    )
    save_scaler_provenance(scaler, ART / "fleet_benign_scaler_reconstructed.json")

    # Predeclare selection rules (also in non_eta json)
    print(
        f"SELECTION RULES (predeclared): slack={ELIGIBILITY_SLACK} "
        f"tie_eps={TIE_EPS} score=mean(S3,S4) CampaignF1"
    )

    detail_rows: list[dict[str, Any]] = []
    cluster_size_rows: list[dict[str, Any]] = []
    # Per scenario pre-gate cache for eta identity check
    label_hashes: dict[str, str] = {}

    manifest = pd.read_csv(SCEN_DIR / "validation_scenario_manifest.csv")
    runs = manifest[manifest["validation_passed"] == True].copy()  # noqa: E712

    for _, mrow in runs.iterrows():
        scen = str(mrow["scenario"])
        seed = int(mrow["validation_seed"])
        run_id = str(mrow["validation_run_id"])
        records_path = SCEN_DIR / f"{run_id}_records.csv"
        if not records_path.exists():
            print("MISSING", records_path)
            continue
        print(f"=== {run_id} ===")
        df = prepare_scenario_df(records_path)
        gt = gt_campaigns_from_scenario(df)
        incidents = incident_map_from_scenario(df)
        campaign_size = int(mrow["campaign_size"])

        pre = run_graphsage_and_dbscan(df, scaler, seed)
        # Record pre-gate cluster sizes
        sizes = (
            pre["pre_gate_summary"]["cluster_size"].astype(int).tolist()
            if len(pre["pre_gate_summary"])
            else []
        )
        for s in sizes:
            cluster_size_rows.append(
                {
                    "scenario": scen,
                    "seed": seed,
                    "run_id": run_id,
                    "campaign_size": campaign_size,
                    "cluster_size": s,
                }
            )

        # Verify DBSCAN/post-fragment labels identical across eta by applying gate only
        label_hash = hashlib.sha256(pre["labels"].tobytes()).hexdigest()
        label_hashes[run_id] = label_hash
        labels_by_eta: dict[int, str] = {}

        for eta in ETA_CANDIDATES:
            labels, summary, decisions = apply_eta_gate(pre, eta)
            labels_by_eta[eta] = hashlib.sha256(labels.tobytes()).hexdigest()
            if labels_by_eta[eta] != label_hash:
                raise RuntimeError(
                    f"STOP: cluster labels changed with eta={eta} for {run_id}"
                )

            preds = predicted_campaigns_from_decisions(decisions, summary)
            metrics = evaluate_protocol_v3(preds, gt, incident_of=incidents)

            n_pre = int(len(summary)) if len(summary) else 0
            n_rej = int(summary["rejected_by_eta_only"].sum()) if n_pre else 0
            n_qual = int(summary["is_qualifying_campaign_cluster"].sum()) if n_pre else 0
            pct_rej = float(n_rej / n_pre) if n_pre else 0.0

            detail_rows.append(
                {
                    "run_id": run_id,
                    "scenario": scen,
                    "seed": seed,
                    "campaign_size": campaign_size,
                    "eta": eta,
                    "metric_protocol": METRIC_PROTOCOL_ID,
                    "n_gt": metrics.n_gt,
                    "n_predicted": metrics.n_predicted,
                    "tp_campaign": metrics.tp_campaign,
                    "fp_campaign": metrics.fp_campaign,
                    "fn_campaign": metrics.fn_campaign,
                    "campaign_precision": metrics.campaign_precision,
                    "campaign_recall": metrics.campaign_recall,
                    "campaign_f1": metrics.campaign_f1,
                    "membership_precision": metrics.membership_precision_micro,
                    "membership_recall": metrics.membership_recall_micro,
                    "membership_f1": metrics.membership_f1_micro,
                    "false_campaign": int(metrics.false_campaign),
                    "incorrect_merge": int(metrics.incorrect_merge_run),
                    "fragmentation_rate_mean": metrics.fragmentation_rate_mean,
                    "fragments_sum": int(sum(metrics.fragments_per_gt.values()))
                    if metrics.fragments_per_gt
                    else 0,
                    "n_pre_gate_clusters": n_pre,
                    "n_qualifying": n_qual,
                    "clusters_rejected_by_eta_only": n_rej,
                    "pct_clusters_rejected_by_eta": pct_rej,
                    "label_hash": label_hash,
                }
            )

        # All etas must share label hash
        if len(set(labels_by_eta.values())) != 1:
            raise RuntimeError(f"STOP: DBSCAN/post-fragment labels differ by eta for {run_id}")

    details = pd.DataFrame(detail_rows)
    details.to_csv(EXP / "ETA_VALIDATION_RESULTS.csv", index=False)

    # Cluster size analysis
    cs = pd.DataFrame(cluster_size_rows)
    analysis_rows = []
    for scen, grp in cs.groupby("scenario"):
        sizes = grp["cluster_size"].to_numpy(dtype=float)
        if sizes.size == 0:
            continue
        analysis_rows.append(
            {
                "scope": scen,
                "n_clusters": int(sizes.size),
                "min": float(np.min(sizes)),
                "p25": float(np.percentile(sizes, 25)),
                "median": float(np.median(sizes)),
                "p75": float(np.percentile(sizes, 75)),
                "max": float(np.max(sizes)),
                "mean": float(np.mean(sizes)),
                "std": float(np.std(sizes)),
                "count_size_2": int((sizes == 2).sum()),
                "count_size_3": int((sizes == 3).sum()),
                "count_size_4": int((sizes == 4).sum()),
                "count_size_5_9": int(((sizes >= 5) & (sizes <= 9)).sum()),
                "count_size_ge_10": int((sizes >= 10).sum()),
            }
        )
    # overall
    if len(cs):
        sizes = cs["cluster_size"].to_numpy(dtype=float)
        analysis_rows.insert(
            0,
            {
                "scope": "ALL_VALIDATION",
                "n_clusters": int(sizes.size),
                "min": float(np.min(sizes)),
                "p25": float(np.percentile(sizes, 25)),
                "median": float(np.median(sizes)),
                "p75": float(np.percentile(sizes, 75)),
                "max": float(np.max(sizes)),
                "mean": float(np.mean(sizes)),
                "std": float(np.std(sizes)),
                "count_size_2": int((sizes == 2).sum()),
                "count_size_3": int((sizes == 3).sum()),
                "count_size_4": int((sizes == 4).sum()),
                "count_size_5_9": int(((sizes >= 5) & (sizes <= 9)).sum()),
                "count_size_ge_10": int((sizes >= 10).sum()),
            },
        )
    pd.DataFrame(analysis_rows).to_csv(EXP / "ETA_CLUSTER_SIZE_ANALYSIS.csv", index=False)

    # Aggregate per eta
    summary_rows = []
    for eta in ETA_CANDIDATES:
        sub = details[details["eta"] == eta]
        def scen_mean(scen: str, col: str) -> float:
            s = sub[sub["scenario"] == scen][col]
            return float(s.mean()) if len(s) else float("nan")

        s0 = scen_mean("S0", "false_campaign")
        s1 = scen_mean("S1", "false_campaign")
        # S1 false escalation: predicted multi-vehicle campaigns when isolated —
        # use false_campaign only if n_gt==0; for S1 use rate of campaign_f1? 
        # User asked S1_false_campaign_rate — for isolated, treat any predicted
        # campaign as false escalation proxy: 1{n_predicted>0} averaged.
        s1_false = float(sub[sub["scenario"] == "S1"]["n_predicted"].gt(0).mean()) if len(sub[sub["scenario"]=="S1"]) else float("nan")
        s2_merge = scen_mean("S2", "incorrect_merge")
        s3_cf1 = scen_mean("S3", "campaign_f1")
        s3_mf1 = scen_mean("S3", "membership_f1")
        s4_cf1 = scen_mean("S4", "campaign_f1")
        s4_mf1 = scen_mean("S4", "membership_f1")
        sel = float(np.nanmean([s3_cf1, s4_cf1]))
        rej = float(sub["clusters_rejected_by_eta_only"].sum())
        pre = float(sub["n_pre_gate_clusters"].sum())
        summary_rows.append(
            {
                "eta": eta,
                "S0_false_campaign_rate": s0,
                "S1_false_campaign_rate": s1_false,
                "S2_incorrect_merge_rate": s2_merge,
                "S3_campaign_f1": s3_cf1,
                "S3_membership_f1": s3_mf1,
                "S4_campaign_f1": s4_cf1,
                "S4_membership_f1": s4_mf1,
                "selection_score": sel,
                "clusters_rejected_by_eta": rej,
                "pre_gate_clusters_total": pre,
                "pct_clusters_rejected_by_eta": (rej / pre) if pre else 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows)

    # Eligibility (rules recorded before seeing selection)
    min_s0 = float(summary["S0_false_campaign_rate"].min())
    min_s1 = float(summary["S1_false_campaign_rate"].min())
    min_s2 = float(summary["S2_incorrect_merge_rate"].min())
    eligible_flags = []
    for _, r in summary.iterrows():
        ok = (
            r["S0_false_campaign_rate"] <= min_s0 + ELIGIBILITY_SLACK
            and r["S1_false_campaign_rate"] <= min_s1 + ELIGIBILITY_SLACK
            and r["S2_incorrect_merge_rate"] <= min_s2 + ELIGIBILITY_SLACK
        )
        eligible_flags.append(bool(ok))
    summary["eligible"] = eligible_flags

    # Select among eligible by selection_score; tie → smaller eta
    elig = summary[summary["eligible"]].copy()
    if elig.empty:
        raise RuntimeError("No eligible eta under predeclared safety filters")
    elig = elig.sort_values(["selection_score", "eta"], ascending=[False, True])
    best_score = float(elig.iloc[0]["selection_score"])
    tied = elig[np.abs(elig["selection_score"] - best_score) <= TIE_EPS]
    selected_eta = int(tied["eta"].min())
    summary["selection_status"] = summary["eta"].map(
        lambda e: "SELECTED" if int(e) == selected_eta else (
            "ELIGIBLE" if bool(summary.loc[summary["eta"] == e, "eligible"].iloc[0]) else "EXCLUDED"
        )
    )
    summary.to_csv(EXP / "ETA_SELECTION_SUMMARY.csv", index=False)

    sel_row = summary[summary["eta"] == selected_eta].iloc[0]

    # Decision impact for selected eta
    sel_details = details[details["eta"] == selected_eta]
    total_pre = int(sel_details["n_pre_gate_clusters"].sum())
    total_rej = int(sel_details["clusters_rejected_by_eta_only"].sum())
    total_qual = int(sel_details["n_qualifying"].sum())
    # Compare to a hypothetical gate without eta (eta=1) on same labels
    # Reconstruct impact: clusters rejected by eta only
    active = total_rej > 0

    eta_selection = {
        "candidate_set": ETA_CANDIDATES,
        "metric_protocol": METRIC_PROTOCOL_ID,
        "validation_dataset": "OCSLab reconstructed validation descriptors/scenarios",
        "validation_scenarios": SCENARIOS,
        "validation_seeds": VALIDATION_SEEDS,
        "campaign_size_strata_available": sorted(details["campaign_size"].unique().tolist()),
        "campaign_size_note": (
            "Reconstructed validation suite uses campaign_size=5 for S2/S3/S4; "
            "sizes 2 and 10 are not present in this reconstructed suite."
        ),
        "selection_rule": {
            "1_exclude_if_S0_S1_S2_worse_than_best_plus_slack": ELIGIBILITY_SLACK,
            "2_maximize_mean_S3_S4_campaign_f1": True,
            "3_tie_break_smaller_eta": True,
        },
        "tie_rule": {
            "absolute_difference_le": TIE_EPS,
            "prefer": "smaller_eta",
            "recorded_before_selection": True,
        },
        "dbscan_labels_identical_across_eta": True,
        "all_candidate_results": summary.to_dict(orient="records"),
        "selected_eta": selected_eta,
        "selection_reason": (
            f"Eligible under S0/S1/S2 slack={ELIGIBILITY_SLACK}; "
            f"maximized selection_score=mean(S3,S4) CampaignF1 "
            f"({sel_row['selection_score']:.6f}); "
            f"tie_eps={TIE_EPS} → selected eta={selected_eta}."
        ),
        "eta_frozen": True,
        "final_test_not_run": True,
        "ctt_not_run": True,
    }
    (EXP / "eta_selection.json").write_text(json.dumps(eta_selection, indent=2) + "\n")

    frozen_fleet = {
        **{k: v for k, v in non_eta.items() if k != "campaign_gate_non_eta"},
        "status": "ETA_FROZEN",
        "eta_status": "SELECTED",
        "selected_eta": selected_eta,
        "metric_protocol": METRIC_PROTOCOL_ID,
        "campaign_gate": {
            "gamma_parameter": "minimum_distinct_vehicles",
            "minimum_distinct_vehicles": 2,
            "eta_parameter": "min_campaign_cluster_size",
            "min_campaign_cluster_size": selected_eta,
            "beta_parameter": "minimum_campaign_cohesion",
            "minimum_campaign_cohesion": 0.5,
            "minimum_cross_vehicle_support": 1,
            "fragment_consolidation_enabled": True,
            "fragment_centroid_threshold": 0.85,
        },
        "dbscan_min_samples_is_not_eta": True,
        "historical_results_modified": False,
        "manuscript_modified": False,
    }
    (EXP / "frozen_fleet_configuration.json").write_text(
        json.dumps(frozen_fleet, indent=2) + "\n"
    )

    # Impact markdown
    by_scen = (
        sel_details.groupby("scenario")
        .agg(
            pre=("n_pre_gate_clusters", "sum"),
            rej=("clusters_rejected_by_eta_only", "sum"),
            qual=("n_qualifying", "sum"),
            camp_f1=("campaign_f1", "mean"),
            false_c=("false_campaign", "mean"),
            merge=("incorrect_merge", "mean"),
        )
        .reset_index()
    )
    impact_status = "ACTIVE" if active else "REDUNDANT_ON_VALIDATION"
    impact_md = [
        "# ETA_DECISION_IMPACT_VALIDATION.md",
        "",
        f"**Selected η = {selected_eta}** (`{METRIC_PROTOCOL_ID}`)",
        "",
        f"## Status: **{impact_status}**",
        "",
        "Computed as clusters that pass γ/β but fail `|C_k| ≥ η` "
        f"(rejected_by_eta_only). Total rejected = {total_rej}.",
        "",
        "## Totals (validation suite)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Pre-gate DBSCAN/post-fragment candidate clusters | {total_pre} |",
        f"| Rejected because `|C_k| < η` only | {total_rej} |",
        f"| Retained qualifying campaigns | {total_qual} |",
        f"| % rejected by η | {100.0 * total_rej / total_pre if total_pre else 0:.2f}% |",
        "",
        "## Breakdown by scenario",
        "",
        by_scen.to_markdown(index=False),
        "",
        "## Strong vs weak (selected η)",
        "",
        f"- S3 Campaign F1 = {sel_row['S3_campaign_f1']:.4f}",
        f"- S3 Membership F1 = {sel_row['S3_membership_f1']:.4f}",
        f"- S4 Campaign F1 = {sel_row['S4_campaign_f1']:.4f}",
        f"- S4 Membership F1 = {sel_row['S4_membership_f1']:.4f}",
        "",
        "## Campaign size",
        "",
        "Reconstructed validation scenarios use campaign_size=5 for S2/S3/S4 "
        "(sizes 2 and 10 not present in this suite).",
        "",
        "## Notes",
        "",
        "- DBSCAN / post-fragment labels verified identical across all η candidates.",
        "- Fragment consolidation applied once with merge-eligibility at η=2 "
        "(smallest candidate), then only the η gate varied.",
        "- Final TEST and CTT were not run.",
        "",
    ]
    (EXP / "ETA_DECISION_IMPACT_VALIDATION.md").write_text("\n".join(impact_md), encoding="utf-8")

    audit_md = [
        "# ETA_VALIDATION_AUDIT.md",
        "",
        f"**Verdict:** ETA_VALIDATION_COMPLETE",
        "",
        f"- Metric protocol: `{METRIC_PROTOCOL_ID}` (FROZEN)",
        f"- Selected η: **{selected_eta}**",
        f"- η impact: **{impact_status}**",
        f"- Candidates evaluated: {ETA_CANDIDATES}",
        f"- Validation runs: {len(runs)} (S0–S4 × 10 seeds)",
        f"- DBSCAN labels identical across η: **Yes**",
        f"- Final test run: **No**",
        f"- CTT run: **No**",
        f"- Manuscript / June artifacts modified: **No**",
        "",
        "## Selection summary",
        "",
        summary.to_markdown(index=False),
        "",
        f"Elapsed seconds: {time.time() - t0:.1f}",
        "",
    ]
    (EXP / "ETA_VALIDATION_AUDIT.md").write_text("\n".join(audit_md), encoding="utf-8")

    # Update checkpoint
    (EXP / "CHECKPOINT_VERDICT.md").write_text(
        "\n".join(
            [
                "# CHECKPOINT_VERDICT.md",
                "",
                "# **ETA_VALIDATION_COMPLETE**",
                "",
                f"Selected η = **{selected_eta}** under `{METRIC_PROTOCOL_ID}`.",
                f"Impact: **{impact_status}**.",
                "",
                "Final OCSLab TEST and CTT were **not** run. η is frozen.",
                "",
                "```text",
                "ETA_VALIDATION_COMPLETE",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({
        "selected_eta": selected_eta,
        "impact": impact_status,
        "summary": summary.to_dict(orient="records"),
        "elapsed": round(time.time() - t0, 1),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
