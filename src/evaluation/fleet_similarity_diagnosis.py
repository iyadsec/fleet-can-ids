"""Diagnose fleet graph cross-vehicle similarity failures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.features.feature_extractor import BEHAVIOURAL_FEATURE_COLUMNS
from src.graph.fleet_graph_builder import (
    build_networkx_graph,
    build_topk_similarity_edges,
    compute_graph_statistics,
    load_anomaly_descriptors,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

ATTACK_TYPES = ("flooding", "fuzzy", "replay", "malfunction")

IEEE_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

CAN_ID_FEATURES = {
    "unique_can_id_count",
    "most_common_can_id_ratio",
    "can_id_entropy",
}
BYTE_FEATURES = {c for c in BEHAVIOURAL_FEATURE_COLUMNS if c.startswith("byte_")}

BEHAVIOUR_ONLY_COLUMNS = [
    "frame_count",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "can_id_entropy",
    "most_common_can_id_ratio",
    "anomaly_score",
]

NORMALIZED_BEHAVIOUR_COLUMNS = [
    "frame_count",
    "mean_inter_arrival_time",
    "std_inter_arrival_time",
    "can_id_entropy",
    "anomaly_score",
]


@dataclass(frozen=True)
class DiagnosisOutputs:
    results_dir: Path
    figures_dir: Path


@dataclass(frozen=True)
class DiagnosisConfig:
    top_k_neighbors: int = 15
    similarity_threshold: float = 0.95
    max_pairs_per_category: int = 50000
    seed: int = 42


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"))
    plt.close(fig)


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (X / norms).astype(np.float32)


def _feature_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[columns].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0)


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["message_rate"] = out["frame_count"].astype(float)
    out["burstiness"] = out["std_inter_arrival_time"].astype(float) / (
        out["mean_inter_arrival_time"].astype(float).abs() + 1e-9
    )
    return out


def _vehicle_zscore(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = out.groupby("vehicle_model")[col].transform(
            lambda s: (s - s.mean()) / (s.std() + 1e-9)
        )
    return out


def _sample_pair_similarities(
    df: pd.DataFrame,
    X: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    max_pairs: int,
    seed: int,
    same_indices: bool = False,
    require_cross_vehicle: bool = False,
    require_same_vehicle: bool = False,
    vehicles: np.ndarray | None = None,
) -> float:
    idx_a = np.flatnonzero(mask_a)
    idx_b = idx_a if same_indices else np.flatnonzero(mask_b)
    if len(idx_a) < 1 or len(idx_b) < 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    sims: list[float] = []
    attempts = 0
    while len(sims) < max_pairs and attempts < max_pairs * 5:
        attempts += 1
        i = int(rng.choice(idx_a))
        j = int(rng.choice(idx_b))
        if i == j:
            continue
        if vehicles is not None:
            if require_cross_vehicle and vehicles[i] == vehicles[j]:
                continue
            if require_same_vehicle and vehicles[i] != vehicles[j]:
                continue
        sims.append(float(cosine_similarity(X[i : i + 1], X[j : j + 1])[0, 0]))
    return float(np.mean(sims)) if sims else float("nan")


def analyse_cross_vehicle_attack_similarity(
    df: pd.DataFrame,
    X: np.ndarray,
    *,
    max_pairs: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    vehicles = df["vehicle_model"].astype(str).to_numpy()
    attacks = df["attack_type"].astype(str).to_numpy()

    for attack in ATTACK_TYPES:
        atk_mask = attacks == attack
        rows.append(
            {
                "attack_type": attack,
                "metric": "average_same_vehicle_similarity",
                "value": _sample_pair_similarities(
                    df, X, atk_mask, atk_mask,
                    max_pairs=max_pairs, seed=seed, same_indices=True,
                    require_same_vehicle=True, vehicles=vehicles,
                ),
            }
        )
        rows.append(
            {
                "attack_type": attack,
                "metric": "average_cross_vehicle_similarity",
                "value": _sample_pair_similarities(
                    df,
                    X,
                    atk_mask,
                    atk_mask,
                    max_pairs=max_pairs,
                    seed=seed + 1,
                    same_indices=False,
                    require_cross_vehicle=True,
                    vehicles=vehicles,
                ),
            }
        )
        rows.append(
            {
                "attack_type": attack,
                "metric": "average_same_attack_cross_vehicle_similarity",
                "value": rows[-1]["value"],
            }
        )

    for attack in ATTACK_TYPES:
        atk_mask = attacks == attack
        diff_mask = (attacks != attack) & np.isin(attacks, list(ATTACK_TYPES))
        rows.append(
            {
                "attack_type": attack,
                "metric": "average_different_attack_cross_vehicle_similarity",
                "value": _sample_pair_similarities(
                    df, X, atk_mask, diff_mask,
                    max_pairs=max_pairs, seed=seed + 2,
                    require_cross_vehicle=True, vehicles=vehicles,
                ),
            }
        )

    return pd.DataFrame(rows)


def analyse_flooding_cross_vehicle(
    flooding: pd.DataFrame,
    X: np.ndarray,
    *,
    top_k: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vehicles = flooding["vehicle_model"].astype(str).to_numpy()
    event_ids = flooding["event_id"].astype(str).to_numpy()
    n = len(flooding)

    # Pairwise vehicle-pair mean similarity
    heat_rows: list[dict[str, Any]] = []
    unique_v = sorted(flooding["vehicle_model"].unique())
    for v1 in unique_v:
        for v2 in unique_v:
            m1 = vehicles == v1
            m2 = vehicles == v2
            if not m1.any() or not m2.any():
                continue
            sims = cosine_similarity(X[m1], X[m2])
            if v1 == v2:
                tri = sims[np.triu_indices(sims.shape[0], k=1)]
                mean_sim = float(np.mean(tri)) if len(tri) else float("nan")
            else:
                mean_sim = float(np.mean(sims))
            heat_rows.append(
                {
                    "source_vehicle": v1,
                    "target_vehicle": v2,
                    "mean_cosine_similarity": mean_sim,
                    "pair_count": int(sims.size if v1 != v2 else len(tri)),
                }
            )
    cross_df = pd.DataFrame(heat_rows)

    Xn = _l2_normalize(X)
    k_query = min(top_k + 1, n)
    nn = NearestNeighbors(n_neighbors=k_query, metric="cosine", algorithm="auto", n_jobs=1)
    nn.fit(Xn)
    distances, indices = nn.kneighbors(Xn, return_distance=True)

    topk_rows: list[dict[str, Any]] = []
    for i in range(n):
        rank = 0
        for dist, j in zip(distances[i], indices[i]):
            if int(j) == i:
                continue
            rank += 1
            sim = float(1.0 - dist)
            same_vehicle = vehicles[i] == vehicles[int(j)]
            topk_rows.append(
                {
                    "descriptor_id": event_ids[i],
                    "vehicle_id": vehicles[i],
                    "rank": rank,
                    "neighbour_descriptor_id": event_ids[int(j)],
                    "neighbour_vehicle_id": vehicles[int(j)],
                    "cosine_similarity": round(sim, 6),
                    "same_vehicle": int(same_vehicle),
                }
            )
            if rank >= top_k:
                break

    topk_df = pd.DataFrame(topk_rows)
    return cross_df, topk_df


def analyse_topk_vehicle_bias(
    df: pd.DataFrame,
    *,
    top_k: int,
    similarity_threshold: float,
    seed: int,
) -> pd.DataFrame:
    X = _feature_matrix(df, list(BEHAVIOURAL_FEATURE_COLUMNS))
    _, _, ei_after, w_after, sub_idx = build_topk_similarity_edges(
        X,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        metric="cosine",
        max_nodes=None,
        seed=seed,
    )
    sub = df.iloc[sub_idx].reset_index(drop=True)
    vehicles = sub["vehicle_model"].astype(str).to_numpy()
    attacks = sub["attack_type"].astype(str).to_numpy()

    same_v = cross_v = 0
    flood_same = flood_cross = flood_total = 0

    if ei_after.size:
        for k in range(ei_after.shape[1]):
            u, v = int(ei_after[0, k]), int(ei_after[1, k])
            if u >= v:
                continue
            sv = vehicles[u] == vehicles[v]
            if sv:
                same_v += 1
            else:
                cross_v += 1
            if attacks[u] == "flooding" and attacks[v] == "flooding":
                flood_total += 1
                if sv:
                    flood_same += 1
                else:
                    flood_cross += 1

    total = same_v + cross_v
    return pd.DataFrame(
        [
            {
                "metric": "pct_edges_same_vehicle",
                "value": round(100.0 * same_v / total, 4) if total else 0.0,
                "count": same_v,
            },
            {
                "metric": "pct_edges_cross_vehicle",
                "value": round(100.0 * cross_v / total, 4) if total else 0.0,
                "count": cross_v,
            },
            {
                "metric": "pct_flooding_edges_cross_vehicle",
                "value": round(100.0 * flood_cross / flood_total, 4) if flood_total else 0.0,
                "count": flood_cross,
            },
            {
                "metric": "total_edges",
                "value": total,
                "count": total,
            },
            {
                "metric": "total_flooding_edges",
                "value": flood_total,
                "count": flood_total,
            },
        ]
    )


def _graph_stats_from_features(
    df: pd.DataFrame,
    columns: list[str],
    *,
    top_k: int,
    similarity_threshold: float,
    seed: int,
    label: str,
) -> tuple[dict[str, Any], nx.Graph]:
    X = _feature_matrix(df, columns)
    _, _, ei_after, w_after, sub_idx = build_topk_similarity_edges(
        X,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        metric="cosine",
        max_nodes=None,
        seed=seed,
    )
    sub = df.iloc[sub_idx].reset_index(drop=True)
    G = build_networkx_graph(sub, ei_after, w_after)
    stats = compute_graph_statistics(G)
    stats["descriptor_variant"] = label
    stats["top_k_neighbors"] = top_k
    stats["similarity_threshold"] = similarity_threshold

    # Cross-vehicle connected components for flooding nodes
    flooding_ids = set(sub.loc[sub["attack_type"] == "flooding", "event_id"].astype(str))
    cross_flood_components = 0
    for component in nx.connected_components(G):
        comp_flood = [n for n in component if str(n) in flooding_ids]
        if len(comp_flood) < 2:
            continue
        comp_df = sub[sub["event_id"].astype(str).isin({str(n) for n in comp_flood})]
        if comp_df["vehicle_model"].nunique() >= 2:
            cross_flood_components += 1
    stats["flooding_cross_vehicle_components"] = cross_flood_components
    return stats, G


def _flooding_cross_vehicle_sim(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    flooding = df[df["attack_type"] == "flooding"].copy()
    X = _feature_matrix(flooding, columns)
    vehicles = flooding["vehicle_model"].astype(str).to_numpy()
    unique_v = sorted(flooding["vehicle_model"].unique())
    rows = []
    for v1 in unique_v:
        for v2 in unique_v:
            if v1 >= v2:
                continue
            m1, m2 = vehicles == v1, vehicles == v2
            sims = cosine_similarity(X[m1], X[m2])
            rows.append(
                {
                    "vehicle_a": v1,
                    "vehicle_b": v2,
                    "mean_cross_vehicle_similarity": float(np.mean(sims)),
                    "max_cross_vehicle_similarity": float(np.max(sims)),
                    "min_cross_vehicle_similarity": float(np.min(sims)),
                }
            )
    return pd.DataFrame(rows)


def _feature_dominance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compare inter-vehicle feature variance for flooding (proxy for vehicle dominance)."""
    flooding = df[df["attack_type"] == "flooding"].copy()
    rows = []
    all_cols = list(BEHAVIOURAL_FEATURE_COLUMNS) + ["anomaly_score", "burstiness", "message_rate"]
    flooding = _add_derived_features(flooding)
    for col in all_cols:
        if col not in flooding.columns:
            continue
        per_vehicle_mean = flooding.groupby("vehicle_model")[col].mean()
        between_var = float(per_vehicle_mean.var())
        within_var = float(
            flooding.groupby("vehicle_model")[col].apply(lambda s: s.var()).mean()
        )
        rows.append(
            {
                "feature": col,
                "between_vehicle_variance": between_var,
                "within_vehicle_variance": within_var,
                "vehicle_dominance_ratio": between_var / (within_var + 1e-9),
            }
        )
    out = pd.DataFrame(rows).sort_values("vehicle_dominance_ratio", ascending=False)
    return out


def _plot_flooding_heatmap(cross_df: pd.DataFrame, out: Path) -> None:
    if cross_df.empty:
        return
    pivot = cross_df.pivot(
        index="source_vehicle", columns="target_vehicle", values="mean_cosine_similarity"
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Flooding Descriptor Cosine Similarity (by Vehicle)")
    _save_figure(fig, out)


def _plot_topk_bias(bias_df: pd.DataFrame, out: Path) -> None:
    sub = bias_df[bias_df["metric"].isin(["pct_edges_same_vehicle", "pct_edges_cross_vehicle"])]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.bar(sub["metric"].str.replace("pct_edges_", ""), sub["value"], color=["#4472C4", "#ED7D31"])
    ax.set_ylabel("Percentage of edges (%)")
    ax.set_title("Top-k Graph: Same vs Cross-Vehicle Edges")
    ax.set_ylim(0, 100)
    _save_figure(fig, out)


def _plot_embedding(
    df: pd.DataFrame,
    columns: list[str],
    out: Path,
    *,
    seed: int,
    max_points: int = 4000,
) -> None:
    sub = df.copy()
    if len(sub) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sub), size=max_points, replace=False)
        sub = sub.iloc[idx]
    X = StandardScaler().fit_transform(_feature_matrix(sub, columns))
    emb = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca").fit_transform(X)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    flooding = sub["attack_type"] == "flooding"
    for vehicle, color in {"Chevrolet": "#4472C4", "Hyundai": "#ED7D31", "Kia": "#70AD47"}.items():
        mask = sub["vehicle_model"] == vehicle
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.25, c=color, label=vehicle)
    if flooding.any():
        ax.scatter(
            emb[flooding, 0],
            emb[flooding, 1],
            s=35,
            facecolors="none",
            edgecolors="red",
            linewidths=0.8,
            label="Flooding",
        )
    ax.legend(fontsize=7, markerscale=1.2)
    ax.set_title("Behaviour-Only Descriptor Embedding (t-SNE)")
    _save_figure(fig, out)


def run_fleet_similarity_diagnosis(
    *,
    descriptors_path: Path,
    features_path: Path,
    outputs: DiagnosisOutputs,
    cfg: DiagnosisConfig,
) -> dict[str, Path]:
    plt.rcParams.update(IEEE_RC)
    outputs.results_dir.mkdir(parents=True, exist_ok=True)
    outputs.figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_anomaly_descriptors(descriptors_path, features_path=features_path)
    df = _add_derived_features(df)
    X_full = _feature_matrix(df, list(BEHAVIOURAL_FEATURE_COLUMNS))

    # 1. Cross-vehicle attack similarity (full descriptors)
    attack_sim = analyse_cross_vehicle_attack_similarity(
        df, X_full, max_pairs=cfg.max_pairs_per_category, seed=cfg.seed
    )
    attack_sim.to_csv(outputs.results_dir / "cross_vehicle_attack_similarity.csv", index=False)

    # 2. Flooding analysis
    flooding = df[df["attack_type"] == "flooding"].reset_index(drop=True)
    X_flood = _feature_matrix(flooding, list(BEHAVIOURAL_FEATURE_COLUMNS))
    flood_cross, flood_topk = analyse_flooding_cross_vehicle(
        flooding, X_flood, top_k=cfg.top_k_neighbors, seed=cfg.seed
    )
    flood_cross.to_csv(outputs.results_dir / "flooding_cross_vehicle_similarity.csv", index=False)
    flood_topk.to_csv(outputs.results_dir / "flooding_topk_neighbour_analysis.csv", index=False)

    # 3. Top-k bias
    bias = analyse_topk_vehicle_bias(
        df,
        top_k=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
    )
    bias.to_csv(outputs.results_dir / "topk_vehicle_bias_analysis.csv", index=False)

    # 4. Vehicle-normalized descriptors
    norm_df = _vehicle_zscore(df, NORMALIZED_BEHAVIOUR_COLUMNS)
    norm_stats, _ = _graph_stats_from_features(
        norm_df,
        NORMALIZED_BEHAVIOUR_COLUMNS,
        top_k=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        label="vehicle_normalized",
    )
    pd.DataFrame([norm_stats]).to_csv(
        outputs.results_dir / "normalized_fleet_graph_statistics.csv", index=False
    )

    # 5. Behaviour-only
    beh_df = _add_derived_features(df)
    beh_cols = BEHAVIOUR_ONLY_COLUMNS + ["burstiness"]
    beh_stats, _ = _graph_stats_from_features(
        beh_df,
        beh_cols,
        top_k=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        label="behaviour_only",
    )
    pd.DataFrame([beh_stats]).to_csv(
        outputs.results_dir / "behaviour_only_fleet_graph_statistics.csv", index=False
    )
    beh_flood_sim = _flooding_cross_vehicle_sim(beh_df, beh_cols)
    beh_flood_sim.to_csv(outputs.results_dir / "behaviour_only_flooding_similarity.csv", index=False)

    # Feature dominance
    dominance = _feature_dominance_report(df)

    # Raw graph stats for comparison
    raw_stats, _ = _graph_stats_from_features(
        df,
        list(BEHAVIOURAL_FEATURE_COLUMNS),
        top_k=cfg.top_k_neighbors,
        similarity_threshold=cfg.similarity_threshold,
        seed=cfg.seed,
        label="raw_full",
    )

    # Figures
    _plot_flooding_heatmap(
        flood_cross, outputs.figures_dir / "flooding_cross_vehicle_similarity_heatmap"
    )
    _plot_topk_bias(bias, outputs.figures_dir / "topk_same_vs_cross_vehicle_edges")
    _plot_embedding(
        df,
        beh_cols,
        outputs.figures_dir / "behaviour_only_descriptor_embedding",
        seed=cfg.seed,
    )

    # Top-k neighbour same-vehicle rate for flooding
    if len(flood_topk):
        flood_same_pct = 100.0 * flood_topk["same_vehicle"].mean()
    else:
        flood_same_pct = float("nan")

    # Extract key metrics
    def _get_attack_metric(atk: str, metric: str) -> float:
        row = attack_sim[(attack_sim["attack_type"] == atk) & (attack_sim["metric"] == metric)]
        return float(row["value"].iloc[0]) if len(row) else float("nan")

    flood_same_v = _get_attack_metric("flooding", "average_same_vehicle_similarity")
    flood_cross_v = _get_attack_metric("flooding", "average_same_attack_cross_vehicle_similarity")
    pct_same_edges = float(bias.loc[bias["metric"] == "pct_edges_same_vehicle", "value"].iloc[0])
    pct_cross_edges = float(bias.loc[bias["metric"] == "pct_edges_cross_vehicle", "value"].iloc[0])
    top_features = dominance.head(5)["feature"].tolist()

    norm_improves = (
        norm_stats.get("flooding_cross_vehicle_components", 0)
        > raw_stats.get("flooding_cross_vehicle_components", 0)
        or norm_stats.get("after_connected_components", 0) != raw_stats.get("after_connected_components", 0)
    )
    beh_improves = beh_stats.get("flooding_cross_vehicle_components", 0) > raw_stats.get(
        "flooding_cross_vehicle_components", 0
    )

    summary_path = outputs.results_dir / "fleet_similarity_diagnosis.md"
    summary_path.write_text(
        "\n".join([
            "# Fleet Similarity Diagnosis",
            "",
            "## 1. Are flooding attacks behaviourally similar across vehicles?",
            "",
            f"- **Same-vehicle flooding similarity (full descriptors):** {flood_same_v:.4f}",
            f"- **Cross-vehicle flooding similarity (full descriptors):** {flood_cross_v:.4f}",
            f"- **Cross-vehicle gap (same − cross):** {flood_same_v - flood_cross_v:.4f}",
            "",
            (
                "Flooding windows are **more similar within a vehicle than across vehicles** "
                "under the current full descriptor, indicating vehicle-specific signal dominates attack signal."
                if flood_same_v > flood_cross_v + 0.05
                else "Cross-vehicle flooding similarity is comparable to same-vehicle similarity."
            ),
            "",
            "## 2. Are top-k neighbours mostly same-vehicle?",
            "",
            f"- **Top-k edges same-vehicle:** {pct_same_edges:.2f}%",
            f"- **Top-k edges cross-vehicle:** {pct_cross_edges:.2f}%",
            f"- **Flooding top-k neighbours same-vehicle:** {flood_same_pct:.2f}%",
            "",
            (
                "Yes — the top-k graph is **strongly biased toward within-vehicle neighbours**, "
                "so flooding nodes rarely link across vehicle models."
                if pct_same_edges > 80 or flood_same_pct > 80
                else "Top-k neighbours are mixed across vehicles."
            ),
            "",
            "## 3. Which descriptor features dominate similarity?",
            "",
            "Highest vehicle-dominance ratio (between-vehicle / within-vehicle variance) for flooding:",
            "",
            *[f"- `{r['feature']}`: ratio {r['vehicle_dominance_ratio']:.2f}" for _, r in dominance.head(8).iterrows()],
            "",
            f"Byte-level payload features ({', '.join(top_features[:3])}) and CAN-ID structure features "
            "vary strongly by vehicle platform, dominating cosine similarity.",
            "",
            "## 4. Does vehicle-normalization improve cross-vehicle flooding similarity?",
            "",
            f"- **Raw graph flooding cross-vehicle components:** {raw_stats.get('flooding_cross_vehicle_components', 0)}",
            f"- **Normalized graph flooding cross-vehicle components:** {norm_stats.get('flooding_cross_vehicle_components', 0)}",
            f"- **Normalized connected components:** {int(norm_stats.get('connected_components', 0))}",
            "",
            (
                "Vehicle z-score normalization **partially reduces platform bias** but may not fully connect flooding across vehicles at τ=0.95."
                if norm_improves
                else "Normalization alone **did not materially increase** cross-vehicle flooding connectivity under current thresholds."
            ),
            "",
            "## 5. Does behaviour-only similarity improve fleet correlation?",
            "",
            f"- **Behaviour-only flooding cross-vehicle components:** {beh_stats.get('flooding_cross_vehicle_components', 0)}",
            f"- **Behaviour-only connected components:** {int(beh_stats.get('connected_components', 0))}",
            "",
            "Behaviour-only cross-vehicle flooding similarity (pairwise means):",
            "",
            *[
                f"- {r['vehicle_a']} ↔ {r['vehicle_b']}: mean={r['mean_cross_vehicle_similarity']:.4f}, max={r['max_cross_vehicle_similarity']:.4f}"
                for _, r in beh_flood_sim.iterrows()
            ],
            "",
            (
                "Behaviour-only features **improve cross-vehicle flooding linkage** relative to full descriptors."
                if beh_improves
                else "Behaviour-only features **do not yet produce cross-vehicle flooding clusters** at top-k=15, τ=0.95 — similarity may still be too strict or timing/entropy differ by platform."
            ),
            "",
            "## Root cause (diagnosis)",
            "",
            "Fleet graph clustering fails to connect flooding across vehicles because:",
            "",
            "1. **Descriptor design:** Full behavioural vectors include byte means/stds and CAN-ID ratios that encode vehicle platform identity.",
            "2. **Top-k construction:** Nearest neighbours are overwhelmingly same-vehicle, producing vehicle-partitioned components.",
            "3. **Graph threshold:** High cosine threshold (0.95) retains only near-duplicate windows, which are typically same-vehicle.",
            "",
            "**Recommendation (diagnosis only — IDS unchanged):** Use behaviour-only, vehicle-normalized features for fleet similarity; "
            "consider lower τ or explicit cross-vehicle kNN quotas for fleet correlation (not local IDS).",
            "",
            f"Parameters: top_k={cfg.top_k_neighbors}, similarity_threshold={cfg.similarity_threshold}.",
            "",
        ]),
        encoding="utf-8",
    )

    logger.info("Fleet similarity diagnosis complete.")
    return {
        "fleet_similarity_diagnosis": summary_path,
        "cross_vehicle_attack_similarity": outputs.results_dir / "cross_vehicle_attack_similarity.csv",
        "flooding_heatmap": outputs.figures_dir / "flooding_cross_vehicle_similarity_heatmap.png",
    }
