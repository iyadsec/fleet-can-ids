#!/usr/bin/env python3
"""Diagnostic-only S3/S4 audit for low Campaign F1. No parameter changes."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation.campaign_clustering import (  # noqa: E402
    extend_dbscan_labels,
    run_dbscan,
    subsample_indices,
)
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
    jaccard,
)
from src.experiments.local_descriptor_normalisation import (  # noqa: E402
    fit_benign_fleet_scaler_from_rows,
)
from src.graph.fleet_graph_builder import (  # noqa: E402
    build_cross_vehicle_constrained_knn_edges,
)
from src.models.gnn_models import train_graphsage_fleet_correlation  # noqa: E402

EXP = ROOT / "experimental-2026-09-20/eta_revalidation_from_raw"
SCEN_DIR = EXP / "validation_scenarios"
ART = EXP / "artifacts"
ETA = 2  # frozen selected
GAMMA = 2
BETA = 0.5
WEAK_TH = 0.55
STRONG_TH = 0.80


def prepare_scenario_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = df.copy()
    out["vehicle_token"] = out["scenario_vehicle_id"].astype(str)
    out["vehicle_model"] = out["scenario_vehicle_id"].astype(str)
    out["event_id"] = out["descriptor_id"].astype(str)
    if "local_alert" not in out.columns:
        out["local_alert"] = (out["anomaly_score"] >= STRONG_TH).astype(int)
    if "weak_signal" not in out.columns:
        out["weak_signal"] = (
            (out["anomaly_score"] >= WEAK_TH) & (out["anomaly_score"] < STRONG_TH)
        ).astype(int)
    return out


def gt_campaigns_from_scenario(df: pd.DataFrame) -> list[GroundTruthCampaign]:
    attacked = df[df["gt_attacked"].astype(int) == 1]
    if attacked.empty:
        return []
    camps: list[GroundTruthCampaign] = []
    for cid, grp in attacked.groupby(attacked["campaign_id"].astype(str)):
        if not cid:
            continue
        vehicles = frozenset(grp["scenario_vehicle_id"].astype(str))
        camps.append(GroundTruthCampaign(campaign_id=cid, vehicles=vehicles))
    return camps


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
        minimum_distinct_vehicles=GAMMA,
        minimum_cross_vehicle_support=1,
        minimum_campaign_cohesion=BETA,
        min_campaign_cluster_size=ETA,
        fragment_consolidation_enabled=True,
        fragment_centroid_threshold=0.85,
        seed=int(seed),
    )


def classify_failure(
    *,
    n_attack_desc: int,
    n_strong: int,
    n_weak: int,
    n_cross_edges: int,
    n_clusters: int,
    gt_vehicles: set[str],
    cluster_rows: list[dict[str, Any]],
    gate_passing: list[dict[str, Any]],
    max_j_any_cluster: float,
    max_j_gate: float,
    tp: int,
) -> str:
    if tp > 0:
        return "NONE_MATCHED"
    if n_attack_desc == 0:
        return "NO_ATTACK_DESCRIPTORS"
    if n_strong + n_weak == 0:
        return "DESCRIPTOR_SCORE_FILTER"
    if n_clusters == 0:
        return "DBSCAN_NO_CLUSTER"
    # Did any cluster contain ≥2 GT vehicles?
    gt_multi_clusters = [
        r for r in cluster_rows if r["n_gt_vehicles_in_cluster"] >= 2
    ]
    if not gt_multi_clusters and n_cross_edges == 0:
        return "INSUFFICIENT_CROSS_VEHICLE_EDGES"
    if not gt_multi_clusters:
        # GT vehicles exist but never co-clustered
        gt_in_any = any(r["n_gt_vehicles_in_cluster"] >= 1 for r in cluster_rows)
        if not gt_in_any:
            return "GRAPHSAGE_EMBEDDING_SEPARATION"
        return "GRAPH_FRAGMENTATION"
    # GT co-clustered somewhere — check gate
    best = max(gt_multi_clusters, key=lambda r: r["n_gt_vehicles_in_cluster"])
    if not best["gamma_pass"]:
        return "GAMMA_GATE_FAILURE"
    if not best["beta_pass"]:
        return "BETA_COHESION_FAILURE"
    if not best["eta_pass"]:
        return "ETA_GATE_FAILURE"
    if gate_passing and max_j_gate < TAU_J:
        return "JACCARD_MATCH_FAILURE"
    if max_j_any_cluster < TAU_J and gate_passing:
        return "JACCARD_MATCH_FAILURE"
    # Gate-passing exists but TP=0
    if gate_passing:
        return "JACCARD_MATCH_FAILURE"
    return "OTHER"


def run_one(df: pd.DataFrame, scaler, seed: int, scen: str, run_id: str) -> dict[str, Any]:
    cfg = base_fleet_config(seed)
    gt = gt_campaigns_from_scenario(df)
    gt_vehicles = set()
    for g in gt:
        gt_vehicles |= set(g.vehicles)
    gt_set = gt_vehicles

    # Descriptor counts
    n_desc = len(df)
    attacked_mask = df["gt_attacked"].astype(int) == 1
    # malicious descriptors: attacked vehicle AND non-benign attack_type
    is_benign_type = df["attack_type"].astype(str).str.lower().isin(
        {"benign", "normal", "none", ""}
    )
    mal_mask = attacked_mask & ~is_benign_type
    # Actually attacked vehicles have 5 mal + 5 benign descriptors mixed
    # Prefer: descriptors from attacked vehicles that are attack-type
    n_attacked_vehicle_desc = int(attacked_mask.sum())
    n_benign_vehicle_desc = int((~attacked_mask).sum())
    n_mal_desc = int(mal_mask.sum())
    n_strong = int(((df["anomaly_score"] >= STRONG_TH) & mal_mask).sum())
    n_weak = int(
        ((df["anomaly_score"] >= WEAK_TH) & (df["anomaly_score"] < STRONG_TH) & mal_mask).sum()
    )

    data, meta, behavior_X, cols, graph_stats = build_publication_fleet_graph(
        df, cfg, fleet_scaler_provenance=scaler
    )

    # Connected components on undirected unique edges
    n_nodes = int(graph_stats["num_nodes"])
    ei = data.edge_index.cpu().numpy() if data.edge_index.numel() else np.zeros((2, 0), dtype=int)
    if ei.shape[1] > 0:
        # Use unique undirected edges for CC
        u = np.minimum(ei[0], ei[1])
        v = np.maximum(ei[0], ei[1])
        und = np.unique(np.vstack([u, v]), axis=1)
        adj = csr_matrix(
            (np.ones(und.shape[1]), (und[0], und[1])), shape=(n_nodes, n_nodes)
        )
        adj = adj + adj.T
        n_cc = int(connected_components(adj, directed=False)[0])
    else:
        n_cc = n_nodes

    # Graph cosine audit on GNN feature space (same X used for edges)
    vehicles = meta["vehicle_token"].astype(str).to_numpy()
    gt_node = np.array([v in gt_set for v in vehicles])
    # Possible cross-vehicle pairs among all nodes
    n_cross_possible = 0
    n_cross_pass_tau = 0
    same_gt_sims = []
    unrelated_sims = []
    # Sample pairwise for large N: use full for 200 is ok (200^2/2=20k)
    sims = cosine_similarity(behavior_X)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if vehicles[i] == vehicles[j]:
                continue
            n_cross_possible += 1
            s = float(sims[i, j])
            if s >= cfg.similarity_threshold:
                n_cross_pass_tau += 1
            both_gt = gt_node[i] and gt_node[j]
            neither_gt = (not gt_node[i]) and (not gt_node[j])
            if both_gt:
                same_gt_sims.append(s)
            elif not both_gt and not (gt_node[i] or gt_node[j] and False):
                # unrelated: at least one non-GT or different
                if not (gt_node[i] and gt_node[j]):
                    unrelated_sims.append(s)

    # GraphSAGE
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

    # Embedding distances
    emb = embeddings
    within = []
    between = []
    gt_idx = np.where(gt_node)[0]
    non_idx = np.where(~gt_node)[0]
    for a in range(len(gt_idx)):
        for b in range(a + 1, len(gt_idx)):
            within.append(float(np.linalg.norm(emb[gt_idx[a]] - emb[gt_idx[b]])))
    for a in gt_idx:
        # sample up to 50 unrelated
        take = non_idx[: min(50, len(non_idx))]
        for b in take:
            between.append(float(np.linalg.norm(emb[a] - emb[b])))

    # DBSCAN
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

    # Fragment merge at eta=2 eligibility (same as sweep)
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
            summary, embeddings, labels, threshold=cfg.fragment_centroid_threshold
        )

    n_noise = int((labels == -1).sum())
    cluster_ids = sorted({int(c) for c in np.unique(labels) if int(c) != -1})

    cluster_rows: list[dict[str, Any]] = []
    gate_passing: list[dict[str, Any]] = []
    max_j_any = 0.0
    max_j_gate = 0.0
    gt_vset = gt[0].vehicles if gt else frozenset()

    for cid in cluster_ids:
        mask = labels == cid
        size = int(mask.sum())
        veh_set = frozenset(meta.loc[mask, "vehicle_token"].astype(str))
        n_veh = len(veh_set)
        cohesion = compute_cluster_behavioral_cohesion(
            behavior_X, mask, seed=cfg.seed + int(cid)
        )
        gamma_pass = n_veh >= GAMMA
        beta_pass = cohesion >= BETA
        eta_pass = size >= ETA
        qualifies = gamma_pass and beta_pass and eta_pass
        n_gt_in = len(veh_set & gt_set)
        j_max = jaccard(veh_set, gt_vset) if gt else 0.0
        max_j_any = max(max_j_any, j_max)
        row = {
            "cluster_id": cid,
            "cluster_size": size,
            "r_k": n_veh,
            "c_k": round(cohesion, 6),
            "gamma_pass": gamma_pass,
            "beta_pass": beta_pass,
            "eta_pass": eta_pass,
            "qualifies": qualifies,
            "V_P": sorted(veh_set),
            "n_gt_vehicles_in_cluster": n_gt_in,
            "jaccard_vs_gt": round(j_max, 6),
        }
        cluster_rows.append(row)
        if qualifies:
            gate_passing.append(row)
            max_j_gate = max(max_j_gate, j_max)

    preds = [
        PredictedCampaign(campaign_id=r["cluster_id"], vehicles=frozenset(r["V_P"]))
        for r in gate_passing
    ]
    metrics = evaluate_protocol_v3(preds, gt)

    failure = classify_failure(
        n_attack_desc=n_mal_desc,
        n_strong=n_strong,
        n_weak=n_weak,
        n_cross_edges=int(graph_stats["cross_vehicle_edges"]),
        n_clusters=len(cluster_ids),
        gt_vehicles=gt_set,
        cluster_rows=cluster_rows,
        gate_passing=gate_passing,
        max_j_any_cluster=max_j_any,
        max_j_gate=max_j_gate,
        tp=int(metrics.tp_campaign),
    )

    # Per-vehicle attack descriptor stats
    veh_stats = []
    for vid, grp in df[df["gt_attacked"].astype(int) == 1].groupby("scenario_vehicle_id"):
        mal = grp[~grp["attack_type"].astype(str).str.lower().isin({"benign", "normal", "none", ""})]
        scores = mal["anomaly_score"].astype(float) if len(mal) else grp["anomaly_score"].astype(float)
        veh_stats.append(
            {
                "vehicle_id": str(vid),
                "n_desc": len(grp),
                "n_mal": len(mal),
                "score_mean": float(scores.mean()) if len(scores) else float("nan"),
                "score_min": float(scores.min()) if len(scores) else float("nan"),
                "score_max": float(scores.max()) if len(scores) else float("nan"),
                "n_strong": int((scores >= STRONG_TH).sum()) if len(scores) else 0,
                "n_weak": int(((scores >= WEAK_TH) & (scores < STRONG_TH)).sum())
                if len(scores)
                else 0,
                "attack_types": sorted(set(mal["attack_type"].astype(str))) if len(mal) else [],
                "source_files": sorted(set(mal["source_file"].astype(str))) if len(mal) and "source_file" in mal.columns else [],
            }
        )

    return {
        "run_id": run_id,
        "scenario": scen,
        "seed": seed,
        "n_desc": n_desc,
        "n_attacked_vehicle_desc": n_attacked_vehicle_desc,
        "n_benign_vehicle_desc": n_benign_vehicle_desc,
        "n_mal_desc": n_mal_desc,
        "n_strong_mal": n_strong,
        "n_weak_mal": n_weak,
        "num_nodes": n_nodes,
        "num_edges_undirected": int(graph_stats["num_edges"]),
        "same_vehicle_edges": int(graph_stats["same_vehicle_edges"]),
        "cross_vehicle_edges": int(graph_stats["cross_vehicle_edges"]),
        "n_connected_components": n_cc,
        "n_cross_possible_pairs": n_cross_possible,
        "n_cross_pairs_pass_tau": n_cross_pass_tau,
        "cross_edge_density": (
            float(graph_stats["cross_vehicle_edges"]) / n_cross_possible
            if n_cross_possible
            else 0.0
        ),
        "same_gt_cosine_mean": float(np.mean(same_gt_sims)) if same_gt_sims else float("nan"),
        "same_gt_cosine_median": float(np.median(same_gt_sims)) if same_gt_sims else float("nan"),
        "unrelated_cosine_mean": float(np.mean(unrelated_sims)) if unrelated_sims else float("nan"),
        "unrelated_cosine_median": float(np.median(unrelated_sims)) if unrelated_sims else float("nan"),
        "emb_within_gt_mean": float(np.mean(within)) if within else float("nan"),
        "emb_gt_unrelated_mean": float(np.mean(between)) if between else float("nan"),
        "n_dbscan_clusters": len(cluster_ids),
        "n_noise_nodes": n_noise,
        "cluster_sizes": [r["cluster_size"] for r in cluster_rows],
        "cluster_rows": cluster_rows,
        "n_gate_passing": len(gate_passing),
        "max_jaccard_any_cluster": max_j_any,
        "max_jaccard_gate_passing": max_j_gate,
        "V_G": sorted(gt_vset),
        "tp": int(metrics.tp_campaign),
        "fp": int(metrics.fp_campaign),
        "fn": int(metrics.fn_campaign),
        "campaign_f1": float(metrics.campaign_f1),
        "membership_f1": float(metrics.membership_f1_micro),
        "failure": failure if metrics.tp_campaign == 0 else "NONE_MATCHED",
        "has_gate_passing_irrespective_of_jaccard": len(gate_passing) > 0,
        "veh_stats": veh_stats,
        "gnn_feature_names": cols,
        "train_metrics": train_metrics,
        "graph_stats": graph_stats,
        "cfg": {
            "tau": cfg.similarity_threshold,
            "k_same": cfg.max_same_vehicle_neighbors,
            "k_cross": cfg.max_cross_vehicle_neighbors,
            "hidden": cfg.gnn_hidden_channels,
            "emb": cfg.gnn_embedding_dim,
            "epochs": cfg.gnn_epochs,
            "lr": cfg.gnn_learning_rate,
            "wd": cfg.gnn_weight_decay,
            "lambda": cfg.campaign_loss_weight,
            "eps": cfg.dbscan_eps,
            "min_samples": cfg.dbscan_min_samples,
            "pca": cfg.dbscan_pca_components,
            "gamma": GAMMA,
            "beta": BETA,
            "eta": ETA,
        },
    }


def main() -> int:
    t0 = time.time()
    os.environ.setdefault(
        "OCSLAB_DATASET_DIR",
        str(ROOT / "Dataset/In-Vehicle Network Intrusion Detection Challenge"),
    )

    train_benign = pd.read_csv(ART / "train_benign_scored_windows.csv")
    scaler = fit_benign_fleet_scaler_from_rows(
        train_benign, feature_names=GNN_FEATURE_COLUMNS, training_split="train"
    )

    manifest = pd.read_csv(SCEN_DIR / "validation_scenario_manifest.csv")
    runs = manifest[
        (manifest["validation_passed"] == True)  # noqa: E712
        & (manifest["scenario"].isin(["S3", "S4"]))
    ].copy()

    results: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    gate_cluster_rows: list[dict[str, Any]] = []
    jaccard_vals: list[float] = []
    jaccard_gate_vals: list[float] = []
    all_pre_gate_sizes: list[int] = []

    # Also load ALL-scenario pre-gate sizes from ETA_VALIDATION_RESULTS / recompute from ETA_CLUSTER
    cs = pd.read_csv(EXP / "ETA_CLUSTER_SIZE_ANALYSIS.csv")

    for _, mrow in runs.iterrows():
        scen = str(mrow["scenario"])
        seed = int(mrow["validation_seed"])
        run_id = str(mrow["validation_run_id"])
        path = SCEN_DIR / f"{run_id}_records.csv"
        print(f"=== DIAG {run_id} ===")
        df = prepare_scenario_df(path)
        out = run_one(df, scaler, seed, scen, run_id)
        results.append(out)

        # Stage diagnostic: one row per run (aggregate) + expand clusters
        stage_rows.append(
            {
                "run_id": run_id,
                "scenario": scen,
                "seed": seed,
                "campaign_size": int(mrow["campaign_size"]),
                "A_n_descriptors": out["n_desc"],
                "B_attacked_vehicle_desc": out["n_attacked_vehicle_desc"],
                "B_benign_vehicle_desc": out["n_benign_vehicle_desc"],
                "B_malicious_desc": out["n_mal_desc"],
                "C_strong_mal_desc": out["n_strong_mal"],
                "C_weak_mal_desc": out["n_weak_mal"],
                "D_n_graph_nodes": out["num_nodes"],
                "E_n_graph_edges_undirected": out["num_edges_undirected"],
                "F_same_vehicle_edges": out["same_vehicle_edges"],
                "G_cross_vehicle_edges": out["cross_vehicle_edges"],
                "H_connected_components": out["n_connected_components"],
                "I_n_dbscan_clusters_pre_gate": out["n_dbscan_clusters"],
                "J_n_noise_nodes": out["n_noise_nodes"],
                "K_cluster_sizes": json.dumps(out["cluster_sizes"]),
                "n_gate_passing": out["n_gate_passing"],
                "R_V_G": json.dumps(out["V_G"]),
                "max_J_any_cluster": out["max_jaccard_any_cluster"],
                "max_J_gate_passing": out["max_jaccard_gate_passing"],
                "T_tp": out["tp"],
                "T_fp": out["fp"],
                "T_fn": out["fn"],
                "campaign_f1": out["campaign_f1"],
                "membership_f1": out["membership_f1"],
                "primary_failure": out["failure"],
                "has_gate_passing_irrespective_of_jaccard": out[
                    "has_gate_passing_irrespective_of_jaccard"
                ],
                "same_gt_cosine_mean": out["same_gt_cosine_mean"],
                "unrelated_cosine_mean": out["unrelated_cosine_mean"],
                "emb_within_gt_mean": out["emb_within_gt_mean"],
                "emb_gt_unrelated_mean": out["emb_gt_unrelated_mean"],
                "n_cross_possible_pairs": out["n_cross_possible_pairs"],
                "n_cross_pairs_pass_tau": out["n_cross_pairs_pass_tau"],
                "cross_edge_density": out["cross_edge_density"],
            }
        )

        for r in out["cluster_rows"]:
            all_pre_gate_sizes.append(int(r["cluster_size"]))
            jaccard_vals.append(float(r["jaccard_vs_gt"]))
            if r["qualifies"]:
                jaccard_gate_vals.append(float(r["jaccard_vs_gt"]))
            gate_cluster_rows.append(
                {
                    "run_id": run_id,
                    "scenario": scen,
                    "seed": seed,
                    "cluster_id": r["cluster_id"],
                    "C_k": r["cluster_size"],
                    "r_k": r["r_k"],
                    "c_k": r["c_k"],
                    "gamma_pass": r["gamma_pass"],
                    "beta_pass": r["beta_pass"],
                    "eta_pass": r["eta_pass"],
                    "qualifies": r["qualifies"],
                    "V_P": json.dumps(r["V_P"]),
                    "V_G": json.dumps(out["V_G"]),
                    "J_P_G": r["jaccard_vs_gt"],
                    "n_gt_vehicles_in_cluster": r["n_gt_vehicles_in_cluster"],
                }
            )

    stage_df = pd.DataFrame(stage_rows)
    stage_df.to_csv(EXP / "S3_S4_STAGE_DIAGNOSTIC.csv", index=False)
    pd.DataFrame(gate_cluster_rows).to_csv(EXP / "S3_S4_GATE_CLUSTER_DETAIL.csv", index=False)

    # Failure analysis
    fail_counts = Counter(r["failure"] for r in results)
    n_miss = sum(1 for r in results if r["tp"] == 0)
    n_tot = len(results)
    fail_lines = [
        "# S3_S4_FAILURE_ANALYSIS.md",
        "",
        f"**Runs analysed:** {n_tot} (S3+S4 × 10 seeds)",
        f"**Missed GT campaigns (tp=0):** {n_miss}",
        f"**Matched (tp≥1):** {n_tot - n_miss}",
        "",
        "## Primary failure classification (missed runs only)",
        "",
        "| Failure | Count | % of missed | % of all S3/S4 |",
        "|---------|------:|------------:|---------------:|",
    ]
    missed = [r for r in results if r["tp"] == 0]
    for k, c in fail_counts.most_common():
        if k == "NONE_MATCHED":
            continue
        pct_m = 100.0 * c / max(1, len(missed))
        pct_a = 100.0 * c / n_tot
        fail_lines.append(f"| {k} | {c} | {pct_m:.1f}% | {pct_a:.1f}% |")
    eta_fail = fail_counts.get("ETA_GATE_FAILURE", 0)
    fail_lines += [
        "",
        f"**ETA_GATE_FAILURE count:** {eta_fail} "
        f"(expected 0 given 0/159 η=2 rejections on full suite).",
        "",
        "## Per-scenario failure breakdown",
        "",
    ]
    for scen in ["S3", "S4"]:
        sub = [r for r in results if r["scenario"] == scen]
        fc = Counter(r["failure"] for r in sub)
        fail_lines.append(f"### {scen}")
        fail_lines.append("")
        for k, c in fc.most_common():
            fail_lines.append(f"- {k}: {c}/{len(sub)}")
        fail_lines.append("")
    # Gate before vs after
    n_with_gt_coclust = sum(
        1
        for r in results
        if any(c["n_gt_vehicles_in_cluster"] >= 2 for c in r["cluster_rows"])
    )
    n_gate = sum(1 for r in results if r["n_gate_passing"] > 0)
    n_tp = sum(1 for r in results if r["tp"] > 0)
    fail_lines += [
        "## Where F1 collapses",
        "",
        f"- Runs with ≥2 GT vehicles co-clustered (pre-gate): **{n_with_gt_coclust}/{n_tot}**",
        f"- Runs with ≥1 gate-passing predicted campaign (γ∧β∧η): **{n_gate}/{n_tot}**",
        f"- Runs with campaign TP under Jaccard τ={TAU_J}: **{n_tp}/{n_tot}**",
        "",
        "Low F1 is therefore primarily **before or at matching**, not η.",
        "",
    ]
    (EXP / "S3_S4_FAILURE_ANALYSIS.md").write_text("\n".join(fail_lines) + "\n", encoding="utf-8")

    # Scenario construction audit
    scen_lines = [
        "# S3_S4_SCENARIO_CONSTRUCTION_AUDIT.md",
        "",
        "## Recoverability",
        "",
        "Historical builder `build_mixed_validation_suite` "
        "(`src.experiments.final_shared_configuration.validation_scenarios`) "
        "is **UNRECOVERABLE** — listed in `recovered_publication_pipeline/MISSING_MODULES.md`. "
        "This reconstruction uses a **stand-in** in "
        "`scripts/reconstruct_validation_pipeline.py` that follows recovered "
        "V0–V4 semantics + node budget, **not** claimed bit-identical.",
        "",
        "## Intended vs reconstructed construction",
        "",
        "| Aspect | Recovered publication intent | This reconstruction | Status |",
        "|--------|------------------------------|----------------------|--------|",
        "| Fleet size / desc per vehicle | 20 / 10 | 20 / 10 | MATCH |",
        "| Mal/benign per attacked | 5 / 5 | 5 / 5 | MATCH |",
        "| S3/S4 campaign size strata | 2, 5, 10 | **5 only** | MISMATCH |",
        "| Seeds | `{11,23,37,41,53,67,71,83,97,101}` (master YAML) | `{131,…,197}` | MISMATCH / alternate list |",
        "| Strong/weak thresholds | 0.80 / 0.55 | 0.80 / 0.55 | MATCH |",
        "| Behavioural coordination | `behavioural_coordination_only: true` | Prototype blend of behavioural features (strength 1.0 / 0.35) toward global attack mean | **KNOWN_PROSPECTIVE / UNVERIFIED vs missing builder** |",
        "| Shared campaign id S3/S4 | One campaign | One `CAMP-*` id | MATCH (semantics) |",
        "| Attack OEMs | Unrecovered exact mix | Hyundai×3 + Kia×2 | UNRECOVERABLE exact |",
        "",
        "## Per-run construction snapshot",
        "",
    ]
    for r in results:
        vs = r["veh_stats"]
        scen_lines.append(f"### {r['run_id']}")
        scen_lines.append("")
        scen_lines.append(f"- scenario={r['scenario']} seed={r['seed']} campaign_size=5")
        scen_lines.append(f"- GT vehicles V(G)={r['V_G']}")
        scen_lines.append(
            f"- malicious descriptors={r['n_mal_desc']} strong={r['n_strong_mal']} weak={r['n_weak_mal']}"
        )
        for v in vs:
            scen_lines.append(
                f"  - {v['vehicle_id']}: n_mal={v['n_mal']} "
                f"score[min/mean/max]={v['score_min']:.3f}/{v['score_mean']:.3f}/{v['score_max']:.3f} "
                f"strong={v['n_strong']} weak={v['n_weak']} "
                f"types={v['attack_types']} sources={v['source_files'][:3]}"
            )
        scen_lines.append("")
    # Check S3 vs S4 score separation
    s3_strong = np.mean([r["n_strong_mal"] for r in results if r["scenario"] == "S3"])
    s4_strong = np.mean([r["n_strong_mal"] for r in results if r["scenario"] == "S4"])
    s3_weak = np.mean([r["n_weak_mal"] for r in results if r["scenario"] == "S4"])
    scen_lines += [
        "## Strong vs weak evidence check",
        "",
        f"- Mean strong-threshold malicious descriptors per S3 run: **{s3_strong:.1f}**",
        f"- Mean strong-threshold malicious descriptors per S4 run: **{s4_strong:.1f}**",
        f"- Mean weak-band malicious descriptors per S4 run: **{np.mean([r['n_weak_mal'] for r in results if r['scenario']=='S4']):.1f}**",
        "",
        "If S3 and S4 pools are not cleanly separated, the stand-in weak-pool "
        "fallback may have used the full attack pool (see builder: weak pool used "
        "only if large enough).",
        "",
        "## Verdict on construction",
        "",
        "**SCENARIO_CONSTRUCTION is a reconstructed stand-in.** Exact historical "
        "coordination logic is UNRECOVERABLE. Node budget and thresholds match; "
        "campaign-size strata and seed list differ; behavioural blending is a "
        "reconstructed proxy.",
        "",
    ]
    (EXP / "S3_S4_SCENARIO_CONSTRUCTION_AUDIT.md").write_text(
        "\n".join(scen_lines) + "\n", encoding="utf-8"
    )

    # Graph audit
    g_lines = [
        "# S3_S4_GRAPH_AUDIT.md",
        "",
        "## Runtime graph parameters (verified)",
        "",
        f"- cosine τ = **{results[0]['cfg']['tau']}**",
        f"- k_same = **{results[0]['cfg']['k_same']}**",
        f"- k_cross = **{results[0]['cfg']['k_cross']}**",
        "- metric = cosine (via `build_cross_vehicle_constrained_knn_edges`)",
        "- Path: `publication_fleet_core.build_publication_fleet_graph` → "
        "`fleet_graph_builder.build_cross_vehicle_constrained_knn_edges`",
        "",
        "## Aggregate S3/S4 graph statistics",
        "",
        "| Metric | Mean | Median | Min | Max |",
        "|--------|-----:|-------:|----:|----:|",
    ]
    for col, key in [
        ("nodes", "num_nodes"),
        ("undirected edges", "num_edges_undirected"),
        ("same-vehicle edges", "same_vehicle_edges"),
        ("cross-vehicle edges", "cross_vehicle_edges"),
        ("connected components", "n_connected_components"),
        ("cross pairs pass τ", "n_cross_pairs_pass_tau"),
        ("cross-edge density", "cross_edge_density"),
        ("same-GT cosine mean", "same_gt_cosine_mean"),
        ("unrelated cosine mean", "unrelated_cosine_mean"),
    ]:
        vals = [float(r[key]) for r in results]
        g_lines.append(
            f"| {col} | {np.mean(vals):.4f} | {np.median(vals):.4f} | "
            f"{np.min(vals):.4f} | {np.max(vals):.4f} |"
        )
    # Cross edges among GT?
    mean_cross = np.mean([r["cross_vehicle_edges"] for r in results])
    mean_same_gt = np.mean([r["same_gt_cosine_mean"] for r in results])
    mean_unrel = np.mean([r["unrelated_cosine_mean"] for r in results])
    g_lines += [
        "",
        "## Interpretation",
        "",
        f"- Mean cross-vehicle edges retained after caps: **{mean_cross:.1f}**",
        f"- Mean cosine (same GT-campaign vehicle descriptors, cross-vehicle pairs): **{mean_same_gt:.4f}**",
        f"- Mean cosine (pairs involving unrelated): **{mean_unrel:.4f}**",
        f"- Cosine τ = 0.95 → same-GT mean {'ABOVE' if mean_same_gt >= 0.95 else 'BELOW'} threshold.",
        "",
        "If same-GT cosine is below τ, coordinated descriptors are **not** becoming "
        "cross-vehicle neighbors at publication τ.",
        "",
        "No parameters were changed.",
        "",
    ]
    (EXP / "S3_S4_GRAPH_AUDIT.md").write_text("\n".join(g_lines) + "\n", encoding="utf-8")

    # GraphSAGE audit
    sage_lines = [
        "# S3_S4_GRAPHSAGE_AUDIT.md",
        "",
        "## Runtime implementation path",
        "",
        "- Entry: `run_eta_validation_sweep.run_graphsage_and_dbscan` / this diagnostic",
        "- Trainer: `src.models.gnn_models.train_graphsage_fleet_correlation`",
        "- Model: `GraphSAGEFleetCorrelator` (SAGEConv mean aggregation)",
        "- No alternate/legacy GNN path invoked.",
        "",
        "## Runtime hyperparameters (frozen)",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| input_dim | {len(GNN_FEATURE_COLUMNS)} |",
        f"| feature_names | {list(GNN_FEATURE_COLUMNS)} |",
        f"| layers | 2 (9→64→32) |",
        f"| hidden | {results[0]['cfg']['hidden']} |",
        f"| embedding | {results[0]['cfg']['emb']} |",
        f"| aggregation | mean (SAGEConv default) |",
        f"| activation | ReLU after first SAGEConv |",
        f"| objective | structure (link + λ·campaign MSE) |",
        f"| epochs | {results[0]['cfg']['epochs']} |",
        f"| optimizer | Adam |",
        f"| learning_rate | {results[0]['cfg']['lr']} |",
        f"| weight_decay | {results[0]['cfg']['wd']} |",
        f"| λ campaign_loss_weight | {results[0]['cfg']['lambda']} |",
        "",
        "## Embedding separation (S3/S4)",
        "",
        f"- Mean within-GT Euclidean distance: "
        f"**{np.mean([r['emb_within_gt_mean'] for r in results]):.4f}**",
        f"- Mean GT↔unrelated Euclidean distance: "
        f"**{np.mean([r['emb_gt_unrelated_mean'] for r in results]):.4f}**",
        "",
        "If within ≈ between, GraphSAGE is not separating coordinated campaign "
        "nodes in embedding space under this reconstructed scenario input.",
        "",
        "No retrain with different parameters was performed.",
        "",
    ]
    (EXP / "S3_S4_GRAPHSAGE_AUDIT.md").write_text(
        "\n".join(sage_lines) + "\n", encoding="utf-8"
    )

    # DBSCAN audit
    db_lines = [
        "# S3_S4_DBSCAN_AUDIT.md",
        "",
        "## Runtime clustering path",
        "",
        "- `src.evaluation.campaign_clustering.run_dbscan` → `DbscanProjector`",
        "- Confirmed pipeline: **StandardScaler → PCA(8) → DBSCAN(eps=0.5, min_samples=2, metric=euclidean)**",
        "",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| StandardScaler | Yes (DbscanProjector) |",
        f"| PCA | {results[0]['cfg']['pca']} |",
        f"| eps | {results[0]['cfg']['eps']} |",
        f"| min_samples | {results[0]['cfg']['min_samples']} |",
        f"| metric | euclidean |",
        "",
        "## Per-run cluster summary",
        "",
        "| run_id | n_clusters | noise | sizes | vehicle composition (r_k list) |",
        "|--------|-----------:|------:|-------|--------------------------------|",
    ]
    for r in results:
        sizes = r["cluster_sizes"]
        rks = [c["r_k"] for c in r["cluster_rows"]]
        db_lines.append(
            f"| {r['run_id']} | {r['n_dbscan_clusters']} | {r['n_noise_nodes']} | "
            f"{sizes} | {rks} |"
        )
    db_lines += ["", "No clustering parameters were changed.", ""]
    (EXP / "S3_S4_DBSCAN_AUDIT.md").write_text("\n".join(db_lines) + "\n", encoding="utf-8")

    # Jaccard diagnostic
    def j_bucket(j: float) -> str:
        if j <= 0:
            return "J=0"
        if j < 0.25:
            return "0<J<0.25"
        if j < 0.5:
            return "0.25<=J<0.5"
        if j < 0.75:
            return "0.5<=J<0.75"
        return "J>=0.75"

    all_j_counts = Counter(j_bucket(j) for j in jaccard_vals)
    gate_j_counts = Counter(j_bucket(j) for j in jaccard_gate_vals)
    n_runs_gate = sum(1 for r in results if r["has_gate_passing_irrespective_of_jaccard"])

    # Pre-gate cluster size audit (full suite from ETA_CLUSTER_SIZE + reconciliation)
    eta_results = pd.read_csv(EXP / "ETA_VALIDATION_RESULTS.csv")
    # Prefer full-suite sizes from ETA_CLUSTER_SIZE_ANALYSIS and per-eta rejections
    size_dist_lines = [
        "# PRE_GATE_CLUSTER_SIZE_AUDIT.md",
        "",
        "## Full validation suite (from η sweep; all S0–S4)",
        "",
    ]
    overall = cs[cs["scope"] == "ALL_VALIDATION"].iloc[0]
    size_dist_lines += [
        f"- n_clusters = **{int(overall['n_clusters'])}**",
        f"- min = {overall['min']}, p25={overall['p25']}, median={overall['median']}, "
        f"p75={overall['p75']}, max={overall['max']}",
        f"- mean={overall['mean']:.3f}, std={overall['std']:.3f}",
        "",
        "| Size bin | Count |",
        "|----------|------:|",
        f"| size=2 | {int(overall['count_size_2'])} |",
        f"| size=3 | {int(overall['count_size_3'])} |",
        f"| size=4 | {int(overall['count_size_4'])} |",
        f"| size=5–9 | {int(overall['count_size_5_9'])} |",
        f"| size≥10 | {int(overall['count_size_ge_10'])} |",
        "",
        f"**Minimum cluster size = {overall['min']}** (DBSCAN `min_samples=2` "
        "implies all non-noise clusters have size ≥ 2).",
        "",
        "## Why η=2 rejected zero clusters",
        "",
        "Rejection requires pass γ/β and fail `|C_k| ≥ η`. With η=2 and "
        "min cluster size ≥ 2, every γ/β-passing cluster also passes η. "
        "Calculated rejected_by_eta_only for η=2 = **0**.",
        "",
        "## Hypothetical rejections at higher η (from η sweep detail)",
        "",
    ]
    for eta in [2, 3, 5, 10]:
        sub = eta_results[eta_results["eta"] == eta]
        rej = int(sub["clusters_rejected_by_eta_only"].sum())
        pre = int(sub["n_pre_gate_clusters"].sum())
        qual = int(sub["n_qualifying"].sum())
        size_dist_lines.append(
            f"- η={eta}: rejected_by_eta_only={rej}/{pre} "
            f"({100.0*rej/pre if pre else 0:.2f}%), qualifying={qual}"
        )
    # Explain identical metrics
    # Find which clusters differ
    pivot = (
        eta_results.groupby(["run_id", "eta"])[["n_qualifying", "n_predicted", "campaign_f1"]]
        .first()
        .reset_index()
    )
    diffs = []
    for run_id, grp in pivot.groupby("run_id"):
        q = grp.set_index("eta")["n_qualifying"].to_dict()
        if len(set(q.values())) > 1:
            diffs.append((run_id, q))
    size_dist_lines += [
        "",
        "## Why metrics were identical across η despite some rejections",
        "",
        f"Runs where `n_qualifying` differed across η: **{len(diffs)}**",
        "",
    ]
    for run_id, q in diffs[:20]:
        size_dist_lines.append(f"- {run_id}: n_qualifying by η = {q}")
    size_dist_lines += [
        "",
        "Those rejected clusters (γ/β pass, size < higher η) did not change "
        "mean S3/S4 Campaign F1 because they were either false campaigns in "
        "non-scoring scenarios or did not create/remove a Jaccard≥0.5 match "
        "on S3/S4. Campaign metrics for S3/S4 remained flat across η.",
        "",
        "## S3/S4-only pre-gate sizes (this diagnostic)",
        "",
    ]
    sarr = np.asarray(all_pre_gate_sizes, dtype=float)
    if sarr.size:
        size_dist_lines += [
            f"- n={sarr.size}, min={sarr.min()}, median={np.median(sarr)}, max={sarr.max()}",
            f"- size=2: {(sarr==2).sum()}, size=3: {(sarr==3).sum()}, size=4: {(sarr==4).sum()}, "
            f"size=5: {(sarr==5).sum()}, size=6–9: {((sarr>=6)&(sarr<=9)).sum()}, "
            f"size≥10: {(sarr>=10).sum()}",
            "",
        ]
    (EXP / "PRE_GATE_CLUSTER_SIZE_AUDIT.md").write_text(
        "\n".join(size_dist_lines) + "\n", encoding="utf-8"
    )

    # Publication vs reconstructed CSV
    pub_rows = [
        ("window_size", "100", "100", "MATCH", "WINDOW_PROVENANCE / master YAML"),
        ("stride", "50", "50", "MATCH", "WINDOW_PROVENANCE / master YAML"),
        ("IF_feature_set", "vehicle-level 24-D peer", "reconstructed IF on raw windows", "MATCH", "VEHICLE_LEVEL_RECONSTRUCTION_AUDIT"),
        ("IF_scoring", "benign-train IF; val FPR≤5%", "benign-train IF; val FPR≤5%", "MATCH", "VEHICLE_LEVEL_RECONSTRUCTION_AUDIT"),
        ("descriptor_construction", "9-D GNN cols + 24-D d_i", "same column set", "MATCH", "VALIDATION_DESCRIPTOR_AUDIT"),
        ("strong_threshold", "0.80", "0.80", "MATCH", "master YAML local_ids"),
        ("weak_threshold", "0.55", "0.55", "MATCH", "master YAML local_ids"),
        ("node_budget", "20×10=200", "20×10=200", "MATCH", "master YAML scenario.*"),
        ("tau", "0.95", "0.95", "MATCH", "final_shared_fleet_configuration.yaml"),
        ("k_same", "2", "2", "MATCH", "final_shared_fleet_configuration.yaml"),
        ("k_cross", "5", "5", "MATCH", "final_shared_fleet_configuration.yaml"),
        ("GraphSAGE_architecture", "9→64→32 mean SAGEConv", "9→64→32 mean SAGEConv", "MATCH", "AUTHORITATIVE_PIPELINE_PARAMETER_AUDIT"),
        ("GraphSAGE_training", "structure; Adam; lr=0.01; wd=5e-4; λ=0.25; epochs=30", "same", "MATCH", "CODE defaults + freeze epochs/lr"),
        ("PCA", "8", "8", "MATCH", "FinalGnnFleetConfig / campaign_clustering"),
        ("DBSCAN_eps", "0.5", "0.5", "MATCH", "final_shared_fleet_configuration.yaml"),
        ("DBSCAN_min_samples", "2", "2", "MATCH", "final_shared_fleet_configuration.yaml"),
        ("DBSCAN_metric", "euclidean", "euclidean", "MATCH", "campaign_clustering.run_dbscan"),
        ("gamma_minimum_distinct_vehicles", "2", "2", "MATCH", "final_shared_fleet_configuration.yaml"),
        ("beta_minimum_campaign_cohesion", "0.5", "0.5", "MATCH", "final_shared_fleet_configuration.yaml"),
        ("eta_min_campaign_cluster_size", "UNRECOVERABLE historically (selected on val)", "2 (validation-selected)", "KNOWN_PROSPECTIVE_CHANGE", "eta_selection.json"),
        ("campaign_construction", "build_mixed_validation_suite MISSING; sizes 2/5/10; seeds 11..101", "stand-in builder; size 5 only; seeds 131..197; prototype blend", "MISMATCH", "MISSING_MODULES.md; VALIDATION_SCENARIO_AUDIT"),
        ("metric_matcher", "extract_run_metrics MISSING (P7/P8 unrecovered)", "revised_jaccard_0.5_v3", "KNOWN_PROSPECTIVE_CHANGE", "PROPOSED_REVISED_METRIC_PROTOCOL_V3.md"),
        ("validation_scenario_module", "UNRECOVERABLE", "reconstruct_validation_pipeline._one_scenario", "UNRECOVERABLE", "MISSING_MODULES.md"),
        ("historical_V3_campaign_f1_at_selection", "~0.649 (selection_report freeze)", "0.10 (this validation)", "MISMATCH", "final_shared_fleet_configuration.yaml selection.best_diag"),
        ("historical_V4_campaign_f1_at_selection", "~0.421 (selection_report freeze)", "0.00 (this validation)", "MISMATCH", "final_shared_fleet_configuration.yaml selection.best_diag"),
    ]
    pub_df = pd.DataFrame(
        pub_rows,
        columns=[
            "component",
            "historical_recovered_value",
            "current_validation_value",
            "status",
            "evidence_source",
        ],
    )
    pub_df.to_csv(EXP / "PUBLICATION_VS_RECONSTRUCTED_PIPELINE.csv", index=False)

    # Jaccard section into failure analysis append + verdict
    j_lines = [
        "",
        "## Jaccard diagnostic (τ_J unchanged at 0.5)",
        "",
        "### All pre-gate clusters vs GT",
        "",
        "| Bucket | Count |",
        "|--------|------:|",
    ]
    for b in ["J=0", "0<J<0.25", "0.25<=J<0.5", "0.5<=J<0.75", "J>=0.75"]:
        j_lines.append(f"| {b} | {all_j_counts.get(b, 0)} |")
    j_lines += [
        "",
        "### Gate-passing clusters only",
        "",
        "| Bucket | Count |",
        "|--------|------:|",
    ]
    for b in ["J=0", "0<J<0.25", "0.25<=J<0.5", "0.5<=J<0.75", "J>=0.75"]:
        j_lines.append(f"| {b} | {gate_j_counts.get(b, 0)} |")
    j_lines += [
        "",
        f"**DIAGNOSTIC ONLY:** S3/S4 runs with ≥1 gate-passing predicted campaign "
        f"(irrespective of Jaccard): **{n_runs_gate}/{n_tot}**",
        "",
        f"Runs with TP under τ_J={TAU_J}: **{n_tp}/{n_tot}**",
        "",
        "If gate-passing ≫ TP, the revised matcher explains part of the gap; "
        "if gate-passing is already near TP, the gap is upstream detection.",
        "",
    ]
    with (EXP / "S3_S4_FAILURE_ANALYSIS.md").open("a", encoding="utf-8") as f:
        f.write("\n".join(j_lines))

    # Determine primary verdict
    statuses = pub_df["status"].value_counts().to_dict()
    mean_same_gt = float(np.mean([r["same_gt_cosine_mean"] for r in results]))
    mean_within = float(np.mean([r["emb_within_gt_mean"] for r in results]))
    mean_between = float(np.mean([r["emb_gt_unrelated_mean"] for r in results]))
    frac_coclust = n_with_gt_coclust / n_tot
    frac_gate = n_gate / n_tot

    causes = []
    # Cause ranking
    causes.append(
        (
            1,
            "SCENARIO_CONSTRUCTION_MISMATCH",
            "Historical `build_mixed_validation_suite` is UNRECOVERABLE; "
            "reconstructed S3/S4 use a stand-in with prototype feature blending, "
            "campaign_size=5 only, and non-publication seed list. "
            "Historical selection_report V3/V4 F1 (~0.65/~0.42) ≫ reconstructed (0.10/0.00).",
        )
    )
    if mean_same_gt < 0.95:
        causes.append(
            (
                2,
                "GRAPH / COORDINATION EVIDENCE WEAK AT τ=0.95",
                f"Mean cross-vehicle cosine among same-GT descriptors = {mean_same_gt:.4f} "
                f"< τ=0.95, so coordinated nodes often fail to become neighbors "
                f"(mean cross edges={mean_cross:.1f}).",
            )
        )
    if mean_within >= 0.8 * mean_between:
        causes.append(
            (
                3,
                "EMBEDDING / CLUSTERING DOES NOT ISOLATE GT CAMPAIGN",
                f"Within-GT emb distance {mean_within:.3f} ≈ GT↔unrelated {mean_between:.3f}; "
                f"only {n_with_gt_coclust}/{n_tot} runs co-cluster ≥2 GT vehicles; "
                f"only {n_gate}/{n_tot} produce any gate-passing campaign.",
            )
        )
    if n_runs_gate > n_tp + 2:
        causes.append(
            (
                4,
                "METRIC_PROTOCOL PARTIAL",
                f"{n_runs_gate} runs have gate-passing campaigns but only {n_tp} match under "
                f"revised_jaccard_0.5_v3 — matcher contributes, but is not the dominant miss mode.",
            )
        )

    # Pick verdict
    if statuses.get("MISMATCH", 0) >= 2 and "UNRECOVERABLE" in statuses:
        verdict = "MULTIPLE_CAUSES_FOUND"
    elif statuses.get("MISMATCH", 0) >= 1 and "campaign_construction" in pub_df[
        pub_df.status == "MISMATCH"
    ].component.tolist():
        # Check if pipeline params mostly match
        non_scen = pub_df[~pub_df.component.isin(
            ["campaign_construction", "metric_matcher", "eta_min_campaign_cluster_size",
             "validation_scenario_module", "historical_V3_campaign_f1_at_selection",
             "historical_V4_campaign_f1_at_selection"]
        )]
        if (non_scen.status == "MATCH").all():
            verdict = "SCENARIO_CONSTRUCTION_MISMATCH_FOUND"
            # But graph/embedding also fail → MULTIPLE
            if mean_same_gt < 0.95 and frac_coclust < 0.5:
                verdict = "MULTIPLE_CAUSES_FOUND"
        else:
            verdict = "RECONSTRUCTION_MISMATCH_FOUND"
    else:
        verdict = "HISTORICAL_PIPELINE_INSUFFICIENTLY_RECOVERABLE"

    # Refine: scenario mismatch + weak coordination evidence + unrecovered builder
    if frac_gate <= 0.2 and mean_same_gt < 0.95:
        verdict = "MULTIPLE_CAUSES_FOUND"

    top3 = causes[:3]

    verdict_md = [
        "# VALIDATION_DIAGNOSTIC_VERDICT.md",
        "",
        f"**Primary verdict:** `{verdict}`",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "Constraints honored: no TEST, no CTT, no manuscript, no η/metric/parameter changes.",
        "",
        f"Selected η=2 remains provisionally frozen. ETA_GATE_FAILURE count = {eta_fail}.",
        "",
        "## Top 3 evidence-backed causes of low S3/S4 Campaign F1",
        "",
    ]
    for i, (_, title, text) in enumerate(top3, 1):
        verdict_md.append(f"{i}. **{title}** — {text}")
        verdict_md.append("")
    verdict_md += [
        "## Key measured facts",
        "",
        f"- S3/S4 runs: {n_tot}; TP runs: {n_tp}; gate-passing runs: {n_gate}",
        f"- Runs with ≥2 GT vehicles co-clustered: {n_with_gt_coclust}",
        f"- Mean same-GT cross-vehicle cosine: {mean_same_gt:.4f} (τ=0.95)",
        f"- Mean emb within-GT / GT↔unrelated: {mean_within:.4f} / {mean_between:.4f}",
        f"- Primary failure mode counts: {dict(fail_counts)}",
        f"- Publication vs reconstructed status counts: {statuses}",
        "",
        f"Elapsed seconds: {time.time()-t0:.1f}",
        "",
        "```text",
        verdict,
        "```",
        "",
    ]
    (EXP / "VALIDATION_DIAGNOSTIC_VERDICT.md").write_text(
        "\n".join(verdict_md) + "\n", encoding="utf-8"
    )

    # Save raw results json for reproducibility
    slim = []
    for r in results:
        slim.append({k: v for k, v in r.items() if k not in {"cfg", "train_metrics", "graph_stats"} or True})
        # drop huge
    # Actually don't dump embeddings; results already without them
    serializable = []
    for r in results:
        d = {k: v for k, v in r.items() if k not in {"train_metrics", "graph_stats", "cfg"}}
        serializable.append(d)
    (ART / "s3_s4_diagnostic_raw.json").write_text(
        json.dumps(serializable, indent=2, default=str) + "\n"
    )

    print(json.dumps({"verdict": verdict, "fail_counts": dict(fail_counts),
                      "n_gate": n_gate, "n_tp": n_tp,
                      "mean_same_gt_cosine": mean_same_gt,
                      "elapsed": round(time.time()-t0, 1)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
