#!/usr/bin/env python3
"""Run corrected real-OCSLab reviewer ablation (methodology-faithful).

Order:
  1) dataset gate
  2) scored pool (build if missing)
  3) scaler check
  4) seed-11 scenarios + cosine validation (STOP if collapse)
  5) if pilot OK → all 10 seeds M1–M4
"""

from __future__ import annotations

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
from sklearn.metrics.pairwise import cosine_similarity

EXP = Path(__file__).resolve().parent
PARENT = EXP.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PARENT))
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
from build_scenarios import build_scenario_from_pool  # noqa: E402
from fleet_scaler import (  # noqa: E402
    FEATURE_NAMES,
    apply_fleet_scaler_matrix,
    load_fleet_benign_scaler,
    verify_scaler_compatibility,
)
from scenario_builder_corrected import (  # noqa: E402
    freeze_scenario,
    load_frozen_scenario,
    pack_frozen,
)


def pair_stats(Xa: np.ndarray, Xb: np.ndarray | None = None) -> dict[str, float]:
    if Xb is None:
        if len(Xa) < 2:
            return {"n_pairs": 0, "mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan, "pct_ge_0_95": np.nan}
        S = cosine_similarity(Xa)
        iu = np.triu_indices(len(Xa), k=1)
        v = S[iu]
    else:
        if len(Xa) == 0 or len(Xb) == 0:
            return {"n_pairs": 0, "mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan, "pct_ge_0_95": np.nan}
        S = cosine_similarity(Xa, Xb)
        v = S.ravel()
    return {
        "n_pairs": int(v.size),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "pct_ge_0_95": float(100.0 * np.mean(v >= 0.95)),
    }


def cosine_validation_seed11(
    pool: pd.DataFrame,
    all_windows: pd.DataFrame,
    cfg: dict[str, Any],
    scaler: dict,
) -> tuple[bool, pd.DataFrame, list[str]]:
    lines: list[str] = ["# Cosine Validation (seed 11)", ""]
    rows: list[dict[str, Any]] = []
    cache = EXP / "scenario_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ok = True
    for scenario in ("strong_campaign", "weak_campaign"):
        try:
            df, notes = build_scenario_from_pool(
                scenario,
                11,
                pool,
                n_vehicles=int(cfg["fleet"]["n_vehicles"]),
                windows_per_vehicle=int(cfg["fleet"]["windows_per_vehicle"]),
                campaign_size=int(cfg["fleet"]["campaign_size"]),
                strong_thr=float(cfg["local_ids"]["strong_threshold"]),
                weak_thr=float(cfg["local_ids"]["weak_threshold"]),
                coordination_strength=float(cfg["campaign_semantics"]["coordination_strength"]),
                prototype_pool=all_windows,
            )
        except Exception as exc:  # noqa: BLE001
            lines.append(f"## {scenario}: UNSUPPORTED — {exc}")
            rows.append({"scenario": scenario, "status": "unsupported", "error": str(exc)})
            if scenario == "strong_campaign":
                ok = False
            continue

        freeze_scenario(df, cache, notes=notes)
        X_raw = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
        X = apply_fleet_scaler_matrix(X_raw, scaler)
        camp = df["gt_campaign_id"].to_numpy() == 0
        # unrelated: other malicious non-campaign
        mal = df["is_attack"].to_numpy() == 1
        ben = ~mal & (df["gt_campaign_id"].to_numpy() == -1)
        # For campaign scenarios, unrelated bucket uses independent malicious if present
        unr = mal & (df["gt_campaign_id"].to_numpy() != 0)

        buckets = {
            "campaign_campaign": pair_stats(X[camp]),
            "campaign_benign": pair_stats(X[camp], X[ben]) if ben.any() else pair_stats(X[camp][:0]),
            "campaign_unrelated": pair_stats(X[camp], X[unr]) if unr.any() else {
                "n_pairs": 0, "mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan, "pct_ge_0_95": np.nan
            },
        }
        lines.append(f"## {scenario}")
        lines.append(f"- notes: {json.dumps(notes)}")
        for name, st in buckets.items():
            lines.append(
                f"- {name}: mean={st['mean']:.4f} median={st['median']:.4f} "
                f"min={st['min']:.4f} max={st['max']:.4f} pct>=0.95={st['pct_ge_0_95']:.2f} "
                f"(n={st['n_pairs']})"
            )
            rows.append({"scenario": scenario, "bucket": name, **st, "status": "ok"})

        # Collapse gate: all three buckets essentially 100% >= 0.95 (PR#22 failure mode)
        pcts = [buckets["campaign_campaign"]["pct_ge_0_95"], buckets["campaign_benign"]["pct_ge_0_95"]]
        if all(np.isfinite(p) and p >= 99.0 for p in pcts):
            lines.append("**FAIL: cosine still globally collapsed (≥99% of camp-camp and camp-benign ≥0.95).**")
            ok = False
        else:
            lines.append("PASS: not globally collapsed like PR #22.")
        lines.append("")

    (EXP / "COSINE_VALIDATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cdf = pd.DataFrame(rows)
    cdf.to_csv(EXP / "artifacts" / "cosine_validation_seed11.csv", index=False)
    return ok, cdf, lines


def evaluate_method(method: str, scenario: str, frozen: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
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
                "n_gt_campaigns": int(len(set(frozen["gt_campaign_id"].tolist()) - {-1})),
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


def run_seed(
    cfg: dict[str, Any],
    seed: int,
    pool: pd.DataFrame,
    all_windows: pd.DataFrame,
    scaler: dict,
    cache_dir: Path,
    verify_lines: list[str],
    manifest_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    set_all_seeds(seed)
    fleet = cfg["fleet"]
    gcfg = cfg["graph"]
    lids = cfg["local_ids"]
    rows: list[dict[str, Any]] = []

    for scenario in cfg["scenarios"]:
        try:
            df, notes = build_scenario_from_pool(
                scenario,
                seed,
                pool,
                n_vehicles=int(fleet["n_vehicles"]),
                windows_per_vehicle=int(fleet["windows_per_vehicle"]),
                campaign_size=int(fleet["campaign_size"]),
                strong_thr=float(lids["strong_threshold"]),
                weak_thr=float(lids["weak_threshold"]),
                coordination_strength=float(cfg["campaign_semantics"]["coordination_strength"]),
                prototype_pool=all_windows,
            )
        except Exception as exc:  # noqa: BLE001
            verify_lines.append(f"SKIP seed={seed} scenario={scenario} reason={exc}")
            rows.append(
                {
                    "seed": seed,
                    "method": "ALL",
                    "scenario": scenario,
                    "campaign_f1": np.nan,
                    "membership_f1": np.nan,
                    "incorrect_merge_rate": np.nan,
                    "campaign_metrics_status": f"unsupported:{exc}",
                }
            )
            continue

        freeze_scenario(df, cache_dir, notes=notes)
        df = load_frozen_scenario(cache_dir, scenario, seed)
        frozen = pack_frozen(df)
        assert int(df["synthetic"].sum()) == 0

        X_scaled = apply_fleet_scaler_matrix(frozen["X"], scaler)
        frozen_scaled = dict(frozen)
        frozen_scaled["X"] = X_scaled

        shared_graph = build_shared_graph(
            X_scaled,
            frozen["vehicles"],
            frozen["meta"],
            similarity_threshold=float(gcfg["similarity_threshold"]),
            max_same_vehicle_neighbors=int(gcfg["max_same_vehicle_neighbors"]),
            max_cross_vehicle_neighbors=int(gcfg["max_cross_vehicle_neighbors"]),
            seed=seed,
        )
        shared_graph_b = build_shared_graph(
            X_scaled,
            frozen["vehicles"],
            frozen["meta"],
            similarity_threshold=float(gcfg["similarity_threshold"]),
            max_same_vehicle_neighbors=int(gcfg["max_same_vehicle_neighbors"]),
            max_cross_vehicle_neighbors=int(gcfg["max_cross_vehicle_neighbors"]),
            seed=seed,
        )
        assert shared_graph["edge_sig"] == shared_graph_b["edge_sig"]
        assert np.allclose(shared_graph["X"], shared_graph_b["X"])

        # M2 clusters scaled g_i (same representation as graph features), matching
        # baseline harness which clusters the method input features — not raw 24-D.
        m1 = run_m1_local(frozen_scaled, cfg)
        m2 = run_m2_descriptor(frozen_scaled, cfg, seed)
        m3 = run_m3_gcn(frozen_scaled, cfg, seed, shared_graph)
        m4 = run_m4_graphsage(frozen_scaled, cfg, seed, shared_graph)

        assert m3["edge_sig"] == m4["edge_sig"] == shared_graph["edge_sig"]
        verify_lines.append(
            f"PASS seed={seed} scenario={scenario} nodes={len(df)} "
            f"edges={len(shared_graph['edge_sig'])} graph_m3==graph_m4 "
            f"scaler_before_cosine=1 synthetic=0 "
            f"notes={notes.get('sampling_adjustments', [])}"
        )
        for method, result in [
            ("M1_local_if", m1),
            ("M2_descriptor_clustering", m2),
            ("M3_gcn", m3),
            ("M4_graphsage", m4),
        ]:
            ev = evaluate_method(method, scenario, frozen_scaled, result)
            ev["seed"] = seed
            rows.append(ev)

        for _, r in df.iterrows():
            manifest_rows.append(
                {
                    "seed": seed,
                    "scenario": scenario,
                    "event_id": r["event_id"],
                    "vehicle_id": r["vehicle_id"],
                    "source_window_uid": r["source_window_uid"],
                    "gt_campaign_id": r["gt_campaign_id"],
                    "is_attack": r["is_attack"],
                    "physical_vehicle": r.get("physical_vehicle"),
                    "source_trace": r.get("source_trace"),
                }
            )
    return rows


def summarize(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "campaign_f1",
        "membership_f1",
        "incorrect_merge_rate",
        "campaign_precision",
        "campaign_recall",
        "fragmentation",
        "false_campaign_rate",
    ]
    rows = []
    for (method, scenario), g in raw.groupby(["method", "scenario"]):
        rec: dict[str, Any] = {"method": method, "scenario": scenario, "n_seeds": len(g)}
        for m in metrics:
            if m not in g.columns:
                continue
            v = pd.to_numeric(g[m], errors="coerce")
            rec[f"{m}_mean"] = float(v.mean())
            rec[f"{m}_std"] = float(v.std(ddof=0)) if len(v) else np.nan
        rows.append(rec)
    summary = pd.DataFrame(rows)

    # Ablation table requested format
    def cell(method: str, scenario: str, metric: str) -> str:
        sub = summary[(summary.method == method) & (summary.scenario == scenario)]
        if sub.empty or f"{metric}_mean" not in sub.columns:
            return "N/A"
        mu = sub.iloc[0][f"{metric}_mean"]
        sd = sub.iloc[0][f"{metric}_std"]
        if pd.isna(mu):
            return "N/A"
        return f"{mu:.3f} ± {sd:.3f}"

    table_rows = []
    for method in [
        "M1_local_if",
        "M2_descriptor_clustering",
        "M3_gcn",
        "M4_graphsage",
    ]:
        table_rows.append(
            {
                "Method": method,
                "Strong Campaign F1": cell(method, "strong_campaign", "campaign_f1"),
                "Weak Campaign F1": cell(method, "weak_campaign", "campaign_f1"),
                "Strong Membership F1": cell(method, "strong_campaign", "membership_f1"),
                "Weak Membership F1": cell(method, "weak_campaign", "membership_f1"),
                "Independent Merge Rate": cell(
                    method, "unrelated_incidents", "incorrect_merge_rate"
                ),
            }
        )
    return summary, pd.DataFrame(table_rows)


def main() -> int:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    (EXP / "artifacts").mkdir(parents=True, exist_ok=True)

    rc = subprocess.call([sys.executable, str(EXP / "check_dataset_gate.py")])
    if rc != 0:
        print("DATASET GATE FAILED", file=sys.stderr)
        return 2

    scaler_info = verify_scaler_compatibility()
    (EXP / "artifacts" / "scaler_check.json").write_text(json.dumps(scaler_info, indent=2))
    if not scaler_info["matches_feature_names"]:
        return 3
    scaler = load_fleet_benign_scaler()

    pool_path = EXP / "artifacts" / "test_window_pool.csv"
    all_path = EXP / "artifacts" / "all_scored_windows.csv"
    if not pool_path.exists() or not all_path.exists():
        from build_scored_pool_corrected import build as build_pool

        build_pool()
    pool = pd.read_csv(pool_path)
    all_windows = pd.read_csv(all_path)
    assert set(pool["vehicle_model"].unique()) >= {"Hyundai", "Kia", "Chevrolet"}

    ok, _, cos_lines = cosine_validation_seed11(pool, all_windows, cfg, scaler)
    print("\n".join(cos_lines))
    if not ok:
        print("STOPPED: cosine validation failed.", file=sys.stderr)
        (EXP / "verification_report.txt").write_text(
            "FAIL cosine validation / strong scenario unsupported\n" + "\n".join(cos_lines),
            encoding="utf-8",
        )
        return 4

    cache = EXP / "scenario_cache"
    verify_lines: list[str] = [
        f"timestamp={datetime.now(timezone.utc).isoformat()}",
        f"python={platform.python_version()}",
        "scaler_before_cosine=True",
        f"beta={cfg['campaign_gate']['cohesion_threshold']}",
        f"fragment={cfg['campaign_gate']['fragment_merge_threshold']}",
        "cosine_validation=PASS",
    ]
    manifest_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []

    # Pilot already built seed-11 scenarios during cosine validation; run methods for all seeds
    for seed in cfg["seeds"]:
        print(f"=== seed {seed} ===", flush=True)
        all_rows.extend(
            run_seed(cfg, seed, pool, all_windows, scaler, cache, verify_lines, manifest_rows)
        )

    raw = pd.DataFrame(all_rows)
    raw.to_csv(EXP / "raw_results.csv", index=False)
    summary, table = summarize(raw)
    summary.to_csv(EXP / "summary_results.csv", index=False)
    table.to_csv(EXP / "ablation_table.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(EXP / "scenario_manifest.csv", index=False)
    (EXP / "verification_report.txt").write_text("\n".join(verify_lines) + "\n", encoding="utf-8")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
