#!/usr/bin/env python3
"""Cosine-collapse diagnostic for real_ocslab (read-only; no experiment changes)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path("/workspace")
HERE = ROOT / "experimental-reviewer-ablation" / "real_ocslab"
sys.path[:0] = [str(ROOT), str(ROOT / "experimental-reviewer-ablation")]

from methods import build_shared_graph  # noqa: E402
from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS  # noqa: E402

CACHE = HERE / "scenario_cache"
ART = HERE / "artifacts"
OUT = HERE / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 23, 37, 41, 53, 67, 71, 83, 97, 101]
SCENARIOS = ["strong_campaign", "weak_campaign"]
TAU = 0.95

GNN9 = [
    "anomaly_score",
    "frame_count",
    "message_rate",
    "burstiness",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "can_id_entropy",
    "most_common_can_id_ratio",
    "payload_entropy",
]
D24 = list(BEHAVIOURAL_FEATURE_COLUMNS)

SCALER_PATH = (
    ROOT
    / "recovered_publication_pipeline"
    / "new_experiments"
    / "final_end_to_end_publication_run_balanced"
    / "scalers"
    / "fleet_benign_scaler.json"
)


def l2n(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def cosmat(X: np.ndarray) -> np.ndarray:
    Xn = l2n(X)
    return Xn @ Xn.T


def pair_vals(sim: np.ndarray, a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    ia = np.where(a)[0]
    if b is None:
        if len(ia) < 2:
            return np.asarray([], dtype=np.float64)
        return sim[np.ix_(ia, ia)][np.triu_indices(len(ia), 1)]
    ib = np.where(b)[0]
    if len(ia) == 0 or len(ib) == 0:
        return np.asarray([], dtype=np.float64)
    return sim[np.ix_(ia, ib)].ravel()


def stats(vals) -> dict:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return {
            "n_pairs": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
            "pct_ge_0_95": float("nan"),
        }
    return {
        "n_pairs": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "p25": float(np.percentile(vals, 25)),
        "median": float(np.median(vals)),
        "p75": float(np.percentile(vals, 75)),
        "max": float(vals.max()),
        "pct_ge_0_95": float(100.0 * np.mean(vals >= TAU)),
    }


def mag_report(X: np.ndarray, names: list[str]) -> list[dict]:
    X = np.asarray(X, dtype=np.float64)
    sq = X**2
    tot = sq.sum(axis=1, keepdims=True)
    share = sq / np.maximum(tot, 1e-12)
    rows = []
    for j, name in enumerate(names):
        c = X[:, j]
        rows.append(
            {
                "feature": name,
                "min": float(c.min()),
                "max": float(c.max()),
                "mean": float(c.mean()),
                "std": float(c.std()),
                "median": float(np.median(c)),
                "mean_abs": float(np.mean(np.abs(c))),
                "mean_share_of_sq_norm": float(share[:, j].mean()),
                "median_share_of_sq_norm": float(np.median(share[:, j])),
            }
        )
    rows.sort(key=lambda r: r["mean_share_of_sq_norm"], reverse=True)
    return rows


def load_hist_scaler(path: Path) -> dict:
    d = json.loads(path.read_text())
    feats = list(d.get("fitted_feature_names") or d.get("feature_names"))
    return {
        "raw": d,
        "features": feats,
        "means": d["means"],
        "stds": d.get("stds") or d.get("scales"),
        "scaler_id": d.get("scaler_id"),
        "fit_row_count": d.get("fit_row_count"),
        "training_split": d.get("training_split"),
        "attack_labels_used": d.get("attack_labels_used"),
    }


def apply_hist(X: np.ndarray, names: list[str], sc: dict) -> np.ndarray:
    out = np.zeros((len(X), len(sc["features"])), dtype=np.float64)
    idx = {n: i for i, n in enumerate(names)}
    for j, f in enumerate(sc["features"]):
        out[:, j] = (X[:, idx[f]] - float(sc["means"][f])) / (float(sc["stds"][f]) + 1e-9)
    return out


def fit_train_benign(all_scored: pd.DataFrame, names: list[str]) -> dict:
    missing = [c for c in names if c not in all_scored.columns]
    if missing:
        raise KeyError(f"all_scored missing {missing}")
    tr = all_scored[(all_scored["split"] == "train") & (all_scored["label"].astype(int) == 0)]
    M = tr[names].fillna(0.0).to_numpy(dtype=np.float64)
    means = M.mean(axis=0)
    stds = M.std(axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    return {"means": means, "stds": stds, "n_fit": int(len(tr)), "names": names}


def apply_fit(X: np.ndarray, sc: dict) -> np.ndarray:
    return (X - sc["means"]) / (sc["stds"] + 1e-9)


def graph_stats(df: pd.DataFrame, X: np.ndarray, seed: int) -> dict:
    meta = df.copy()
    meta["event_id"] = meta["event_id"].astype(str)
    meta["vehicle_model"] = meta["vehicle_id"].astype(str)
    g = build_shared_graph(
        X,
        meta["vehicle_id"].astype(str).tolist(),
        meta,
        similarity_threshold=TAU,
        max_same_vehicle_neighbors=2,
        max_cross_vehicle_neighbors=5,
        seed=seed,
    )
    nxg = g["nx_graph"]
    meta_g = g["meta"].reset_index(drop=True)
    eid_to_i = {str(e): i for i, e in enumerate(meta_g["event_id"].astype(str))}
    gt = meta_g["gt_campaign_id"].astype(int).to_numpy()
    is_atk = meta_g["is_attack"].astype(int).to_numpy()
    camp = gt >= 0
    benign = is_atk == 0
    unrelated = (is_atk == 1) & (~camp)
    sim = cosmat(X)
    n = len(X)
    triu = sim[np.triu_indices(n, 1)]

    def pge(a, b=None):
        v = pair_vals(sim, a, b)
        if v.size == 0:
            return {"n_pairs": 0, "n_ge_0_95": 0, "pct_ge_0_95": float("nan")}
        return {
            "n_pairs": int(v.size),
            "n_ge_0_95": int((v >= TAU).sum()),
            "pct_ge_0_95": float(100.0 * (v >= TAU).mean()),
        }

    same = cross = cc = cb = cu = 0
    vid = meta_g["vehicle_id"].astype(str).to_numpy()
    for u, v in nxg.edges():
        iu, iv = eid_to_i[str(u)], eid_to_i[str(v)]
        if vid[iu] == vid[iv]:
            same += 1
        else:
            cross += 1
        if camp[iu] and camp[iv]:
            cc += 1
        elif (camp[iu] and benign[iv]) or (camp[iv] and benign[iu]):
            cb += 1
        elif (camp[iu] and unrelated[iv]) or (camp[iv] and unrelated[iu]):
            cu += 1

    camp_nodes = [
        str(e) for e in meta_g.loc[camp, "event_id"].astype(str) if str(e) in nxg
    ]
    sub = nxg.subgraph(camp_nodes).copy()
    sub.remove_edges_from(nx.selfloop_edges(sub))
    comps = list(nx.connected_components(sub)) if sub.number_of_nodes() else []
    sizes = sorted((len(c) for c in comps), reverse=True)
    return {
        "n_nodes": n,
        "candidate_undirected_pairs": int(triu.size),
        "pct_all_pairs_ge_0_95": float(100.0 * (triu >= TAU).mean()),
        "camp_camp_pairs": pge(camp),
        "camp_benign_pairs": pge(camp, benign),
        "camp_unrelated_pairs": pge(camp, unrelated),
        "retained_undirected_edges": int(nxg.number_of_edges()),
        "same_vehicle_edges": int(same),
        "cross_vehicle_edges": int(cross),
        "campaign_to_campaign_edges": int(cc),
        "campaign_to_benign_edges": int(cb),
        "campaign_to_unrelated_edges": int(cu),
        "campaign_connected_components": int(len(comps)),
        "largest_campaign_component": int(sizes[0]) if sizes else 0,
        "full_graph_components": int(nx.number_connected_components(nxg)),
        "largest_full_component": int(
            max((len(c) for c in nx.connected_components(nxg)), default=0)
        ),
    }


def main() -> int:
    pool = pd.read_csv(ART / "test_window_pool.csv")
    all_scored = pd.read_csv(ART / "all_scored_windows.csv")

    # Ensure derived GNN cols exist on all_scored for R1 fit
    if "message_rate" not in all_scored.columns and "frame_count" in all_scored.columns:
        all_scored["message_rate"] = all_scored["frame_count"]
    if "burstiness" not in all_scored.columns:
        all_scored["burstiness"] = all_scored["std_inter_arrival_time"] / (
            all_scored["mean_inter_arrival_time"].abs() + 1e-9
        )

    for c in GNN9 + D24:
        if c not in pool.columns and c in all_scored.columns:
            # shouldn't happen for pool
            pass
    miss_pool_24 = [c for c in D24 if c not in pool.columns]
    miss_pool_9 = [c for c in GNN9 if c not in pool.columns]
    if miss_pool_24:
        raise KeyError(f"pool missing 24-D cols: {miss_pool_24}")
    if miss_pool_9:
        # payload_entropy / message_rate / burstiness / anomaly_score should be present
        raise KeyError(f"pool missing 9-D cols: {miss_pool_9}")

    sc = load_hist_scaler(SCALER_PATH)
    r1_9 = fit_train_benign(all_scored, GNN9)
    r1_24 = fit_train_benign(all_scored, D24)

    items = {}
    mag9_rows = []
    mag24_rows = []

    for seed in SEEDS:
        urel = pd.read_csv(CACHE / f"unrelated_incidents_seed{seed}.csv")
        urel_m = urel[urel["is_attack"].astype(int) == 1].copy()
        miss_u = [c for c in GNN9 if c not in urel_m.columns]
        if miss_u:
            raise KeyError(f"unrelated missing {miss_u}")
        X9u = urel_m[GNN9].to_numpy(dtype=np.float64)
        X24u = pool.set_index("window_uid").reindex(urel_m["source_window_uid"].astype(str))[D24].fillna(0.0).to_numpy(dtype=np.float64)

        for scenario in SCENARIOS:
            df = pd.read_csv(CACHE / f"{scenario}_seed{seed}.csv")
            miss = [c for c in GNN9 if c not in df.columns]
            if miss:
                raise KeyError(f"{scenario} seed{seed} missing {miss}")
            X9 = df[GNN9].to_numpy(dtype=np.float64)
            X24_df = pool.set_index("window_uid").reindex(df["source_window_uid"].astype(str))[D24]
            if X24_df.isna().any().any():
                nmiss = int(X24_df.isna().any(axis=1).sum())
                raise RuntimeError(f"24-D join miss {nmiss} for {scenario} seed{seed}")
            X24 = X24_df.to_numpy(dtype=np.float64)
            camp = (df["gt_campaign_id"] >= 0).to_numpy()
            benign = (df["is_attack"].astype(int) == 0).to_numpy()
            items[(scenario, seed)] = {
                "df": df,
                "X9": X9,
                "X24": X24,
                "X9u": X9u,
                "X24u": X24u,
                "camp": camp,
                "benign": benign,
            }
            mag9_rows.append(X9)
            mag24_rows.append(X24)

    def agg_raw(dim: str) -> dict:
        key = "X9" if dim == "9d" else "X24"
        ku = "X9u" if dim == "9d" else "X24u"
        out = {}
        for scenario in SCENARIOS:
            A, B, C, D, E = [], [], [], [], []
            for seed in SEEDS:
                it = items[(scenario, seed)]
                X, Xu = it[key], it[ku]
                sim = cosmat(X)
                A.append(pair_vals(sim, it["camp"]))
                B.append(pair_vals(sim, it["camp"], it["benign"]))
                D.append(pair_vals(sim, it["benign"]))
                campX = l2n(X[it["camp"]])
                unX = l2n(Xu)
                if len(campX) and len(unX):
                    C.append((campX @ unX.T).ravel())
                if len(Xu) >= 2:
                    su = cosmat(Xu)
                    E.append(su[np.triu_indices(len(Xu), 1)])
            out[scenario] = {
                "A_camp_camp": stats(np.concatenate(A)),
                "B_camp_benign": stats(np.concatenate(B)),
                "C_camp_unrelated": stats(np.concatenate(C) if C else []),
                "D_benign_benign": stats(np.concatenate(D)),
                "E_unrelated_unrelated": stats(np.concatenate(E) if E else []),
            }
        return out

    cosine_9 = agg_raw("9d")
    cosine_24 = agg_raw("24d")
    mag_9 = mag_report(np.vstack(mag9_rows), GNN9)
    mag_24 = mag_report(np.vstack(mag24_rows), D24)
    pd.DataFrame(mag_9).to_csv(OUT / "feature_magnitudes_9d.csv", index=False)
    pd.DataFrame(mag_24).to_csv(OUT / "feature_magnitudes_24d.csv", index=False)

    def agg_t9(kind: str) -> dict:
        out = {}
        for scenario in SCENARIOS:
            A, B, C = [], [], []
            for seed in SEEDS:
                it = items[(scenario, seed)]
                X, Xu = it["X9"], it["X9u"]
                if kind == "R0":
                    Xt, Xut = X, Xu
                elif kind == "R1":
                    Xt, Xut = apply_fit(X, r1_9), apply_fit(Xu, r1_9)
                else:
                    Xt, Xut = apply_hist(X, GNN9, sc), apply_hist(Xu, GNN9, sc)
                sim = cosmat(Xt)
                A.append(pair_vals(sim, it["camp"]))
                B.append(pair_vals(sim, it["camp"], it["benign"]))
                campX = l2n(Xt[it["camp"]])
                unX = l2n(Xut)
                if len(campX) and len(unX):
                    C.append((campX @ unX.T).ravel())
            out[scenario] = {
                "within_campaign": stats(np.concatenate(A)),
                "campaign_vs_benign": stats(np.concatenate(B)),
                "campaign_vs_unrelated": stats(np.concatenate(C) if C else []),
            }
        return out

    def agg_t24(kind: str) -> dict:
        out = {}
        for scenario in SCENARIOS:
            A, B, C = [], [], []
            for seed in SEEDS:
                it = items[(scenario, seed)]
                X, Xu = it["X24"], it["X24u"]
                if kind == "R0":
                    Xt, Xut = X, Xu
                else:
                    Xt, Xut = apply_fit(X, r1_24), apply_fit(Xu, r1_24)
                sim = cosmat(Xt)
                A.append(pair_vals(sim, it["camp"]))
                B.append(pair_vals(sim, it["camp"], it["benign"]))
                campX = l2n(Xt[it["camp"]])
                unX = l2n(Xut)
                if len(campX) and len(unX):
                    C.append((campX @ unX.T).ravel())
            out[scenario] = {
                "within_campaign": stats(np.concatenate(A)),
                "campaign_vs_benign": stats(np.concatenate(B)),
                "campaign_vs_unrelated": stats(np.concatenate(C) if C else []),
            }
        return out

    repr9 = {
        "R0_raw_9d": agg_t9("R0"),
        "R1_train_benign_standardscaler_9d": agg_t9("R1"),
        "R2_historical_fleet_benign_scaler_9d": agg_t9("R2"),
    }
    repr24 = {
        "R0_raw_24d": agg_t24("R0"),
        "R1_train_benign_standardscaler_24d": agg_t24("R1"),
        "R2_historical_scaler": {
            "status": "NOT_APPLICABLE",
            "reason": "Historical fleet_benign_scaler is 9-D GNN features, not 24-D behavioural columns",
            "scaler_features": sc["features"],
            "d24_features": D24,
        },
    }

    graph11 = {}
    for scenario in SCENARIOS:
        it = items[(scenario, 11)]
        X0 = it["X9"]
        X1 = apply_fit(X0, r1_9)
        X2 = apply_hist(X0, GNN9, sc)
        print(f"graph {scenario}", flush=True)
        graph11[scenario] = {
            "R0_raw_9d": graph_stats(it["df"], X0, 11),
            "R1_train_benign_standardscaler_9d": graph_stats(it["df"], X1, 11),
            "R2_historical_fleet_benign_scaler_9d": graph_stats(it["df"], X2, 11),
        }

    hyundai = sorted((ROOT / "Dataset" / "ocslab_pipeline" / "Hyundai").iterdir())
    kia = sorted((ROOT / "Dataset" / "ocslab_pipeline" / "Kia").iterdir())
    chev = sorted((ROOT / "Dataset" / "ocslab_pipeline" / "Chevrolet").iterdir())

    report = {
        "cosine_distributions_raw": {"g_i_9d": cosine_9, "d_i_24d": cosine_24},
        "feature_magnitudes": {"g_i_9d_top": mag_9[:5], "d_i_24d_top": mag_24[:8]},
        "pipeline_trace": {
            "pr22_path": [
                "scenario_builder_real.pack_frozen -> X = df[FEATURE_NAMES] raw 9-D",
                "run_real_ablation.build_shared_graph(frozen['X'], ...)",
                "methods.build_shared_graph -> build_scenario_graph_from_features(X,...)",
                "build_cross_vehicle_constrained_knn_edges(metric='cosine', threshold=0.95)",
                "NO fleet scaler / StandardScaler before cosine",
                "Cosine L2-normalizes each vector but does NOT correct per-dimension scale imbalance",
            ],
            "cosine_calculated_on": "A_raw_unscaled_feature_values_then_implicit_per_vector_L2",
            "publication_default_view": "behavior_only_vehicle_normalized (requires fleet_scaler_provenance)",
            "publication_scaler_applied_before_cosine": True,
        },
        "historical_scaler": {
            "path": str(SCALER_PATH),
            "scaler_type": "z-score (benign-train means/stds; std floored to 1.0 for near-constant features)",
            "scaler_id": sc["scaler_id"],
            "n_input_features": len(sc["features"]),
            "feature_ordering": sc["features"],
            "fit_row_count": sc["fit_row_count"],
            "training_split": sc["training_split"],
            "attack_labels_used": sc["attack_labels_used"],
            "means": sc["means"],
            "stds": sc["stds"],
            "matches_pr22_gnn_feature_names": sc["features"] == GNN9,
            "constant_like_features_with_std_floored_to_1": [
                f for f in sc["features"] if abs(float(sc["stds"][f]) - 1.0) < 1e-12
            ],
            "historical_application": (
                "Publication prepare_fleet_similarity_matrix applies apply_fleet_scaler "
                "BEFORE cosine when view is behavior_only_vehicle_normalized. "
                "PR#22 feeds raw 9-D into cosine kNN."
            ),
        },
        "r1_scaler_fit": {
            "gnn9": {
                "n_fit": r1_9["n_fit"],
                "population": "all_scored split=train label=0",
                "means": {n: float(v) for n, v in zip(GNN9, r1_9["means"])},
                "stds": {n: float(v) for n, v in zip(GNN9, r1_9["stds"])},
            },
            "d24": {
                "n_fit": r1_24["n_fit"],
                "population": "all_scored split=train label=0",
                "means": {n: float(v) for n, v in zip(D24, r1_24["means"])},
                "stds": {n: float(v) for n, v in zip(D24, r1_24["stds"])},
            },
        },
        "controlled_transforms_9d": repr9,
        "controlled_transforms_24d": repr24,
        "graph_density_seed11": graph11,
        "dataset_mismatch": {
            "paper_platforms": ["Hyundai Sonata", "Kia Soul", "Chevrolet Spark"],
            "pr22_physical_tags": ["OCSLab_Car", "Challenge_D", "Challenge_S"],
            "OCSLab_Car": "HCRL classic Car-Hacking single-vehicle capture; OEM not asserted by loader",
            "Challenge_D": "Car Hacking Challenge Preliminary Training track D",
            "Challenge_S": "Car Hacking Challenge Preliminary Training track S",
            "oem_named_traces_accessible_in_workspace": {
                "Hyundai_dir": [p.name for p in hyundai],
                "Kia_dir": [p.name for p in kia],
                "Chevrolet_dir": [p.name for p in chev],
                "note": (
                    "Dataset/ocslab_pipeline/{Hyundai,Kia,Chevrolet} exist, but filenames do not "
                    "contain Sonata/Soul/Spark. Recovered publication manifests reference "
                    "Attack_free_HY_Sonata_* / KIA_Soul_* / CHEVROLET_Spark_* which are NOT the "
                    "sources used by PR#22 build_scored_pool.py."
                ),
            },
            "pr22_uses_paper_three_vehicle_datasets": False,
        },
    }

    (OUT / "cosine_collapse_audit.json").write_text(json.dumps(report, indent=2))

    rows = []
    for dim, block in [("9d", cosine_9), ("24d", cosine_24)]:
        for scen, buckets in block.items():
            for bname, st in buckets.items():
                rows.append({"dim": dim, "scenario": scen, "bucket": bname, **st})
    pd.DataFrame(rows).to_csv(OUT / "cosine_distributions_raw.csv", index=False)

    trows = []
    for tname, block in repr9.items():
        for scen, buckets in block.items():
            for bname, st in buckets.items():
                trows.append(
                    {"transform": tname, "dim": "9d", "scenario": scen, "bucket": bname, **st}
                )
    for tname, block in repr24.items():
        if not isinstance(block, dict) or "strong_campaign" not in block:
            continue
        for scen, buckets in block.items():
            for bname, st in buckets.items():
                trows.append(
                    {"transform": tname, "dim": "24d", "scenario": scen, "bucket": bname, **st}
                )
    pd.DataFrame(trows).to_csv(OUT / "cosine_distributions_transforms.csv", index=False)

    grows = []
    for scen, reps in graph11.items():
        for rname, st in reps.items():
            flat = {"scenario": scen, "representation": rname}
            for k, v in st.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        flat[f"{k}__{kk}"] = vv
                else:
                    flat[k] = v
            grows.append(flat)
    pd.DataFrame(grows).to_csv(OUT / "graph_density_seed11.csv", index=False)

    summary = {
        "raw9_strong_B": cosine_9["strong_campaign"]["B_camp_benign"],
        "raw24_strong_B": cosine_24["strong_campaign"]["B_camp_benign"],
        "raw9_strong_A": cosine_9["strong_campaign"]["A_camp_camp"],
        "raw24_strong_A": cosine_24["strong_campaign"]["A_camp_camp"],
        "mag9_top3": mag_9[:3],
        "mag24_top5": mag_24[:5],
        "R0_within_pct": repr9["R0_raw_9d"]["strong_campaign"]["within_campaign"]["pct_ge_0_95"],
        "R0_vsben_pct": repr9["R0_raw_9d"]["strong_campaign"]["campaign_vs_benign"]["pct_ge_0_95"],
        "R1_within_pct": repr9["R1_train_benign_standardscaler_9d"]["strong_campaign"][
            "within_campaign"
        ]["pct_ge_0_95"],
        "R1_vsben_pct": repr9["R1_train_benign_standardscaler_9d"]["strong_campaign"][
            "campaign_vs_benign"
        ]["pct_ge_0_95"],
        "R2_within_pct": repr9["R2_historical_fleet_benign_scaler_9d"]["strong_campaign"][
            "within_campaign"
        ]["pct_ge_0_95"],
        "R2_vsben_pct": repr9["R2_historical_fleet_benign_scaler_9d"]["strong_campaign"][
            "campaign_vs_benign"
        ]["pct_ge_0_95"],
        "R0_weak_vsben_pct": repr9["R0_raw_9d"]["weak_campaign"]["campaign_vs_benign"][
            "pct_ge_0_95"
        ],
        "R2_weak_vsben_pct": repr9["R2_historical_fleet_benign_scaler_9d"]["weak_campaign"][
            "campaign_vs_benign"
        ]["pct_ge_0_95"],
        "R1_24_vsben_pct": repr24["R1_train_benign_standardscaler_24d"]["strong_campaign"][
            "campaign_vs_benign"
        ]["pct_ge_0_95"],
        "R0_24_vsben_pct": repr24["R0_raw_24d"]["strong_campaign"]["campaign_vs_benign"][
            "pct_ge_0_95"
        ],
        "graph_R0": {
            "edges": graph11["strong_campaign"]["R0_raw_9d"]["retained_undirected_edges"],
            "camp_benign_edges": graph11["strong_campaign"]["R0_raw_9d"][
                "campaign_to_benign_edges"
            ],
            "camp_camp_edges": graph11["strong_campaign"]["R0_raw_9d"][
                "campaign_to_campaign_edges"
            ],
            "cc": graph11["strong_campaign"]["R0_raw_9d"]["campaign_connected_components"],
            "largest": graph11["strong_campaign"]["R0_raw_9d"]["largest_campaign_component"],
            "pct_all_ge": graph11["strong_campaign"]["R0_raw_9d"]["pct_all_pairs_ge_0_95"],
        },
        "graph_R2": {
            "edges": graph11["strong_campaign"]["R2_historical_fleet_benign_scaler_9d"][
                "retained_undirected_edges"
            ],
            "camp_benign_edges": graph11["strong_campaign"][
                "R2_historical_fleet_benign_scaler_9d"
            ]["campaign_to_benign_edges"],
            "camp_camp_edges": graph11["strong_campaign"][
                "R2_historical_fleet_benign_scaler_9d"
            ]["campaign_to_campaign_edges"],
            "cc": graph11["strong_campaign"]["R2_historical_fleet_benign_scaler_9d"][
                "campaign_connected_components"
            ],
            "largest": graph11["strong_campaign"]["R2_historical_fleet_benign_scaler_9d"][
                "largest_campaign_component"
            ],
            "pct_all_ge": graph11["strong_campaign"]["R2_historical_fleet_benign_scaler_9d"][
                "pct_all_pairs_ge_0_95"
            ],
        },
        "scaler_stds": sc["stds"],
        "scaler_means_frame_message": {
            "frame_count": sc["means"]["frame_count"],
            "message_rate": sc["means"]["message_rate"],
        },
    }
    print(json.dumps(summary, indent=2))
    print("WROTE", OUT / "cosine_collapse_audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
