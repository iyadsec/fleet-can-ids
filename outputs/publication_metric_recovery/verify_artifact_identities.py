#!/usr/bin/env python3
"""Verification-only: algebraic identities + P7/P8 aggregation from frozen campaign_metrics.

Does NOT re-implement extract_run_metrics / run_refinement_fcgnn.
Does NOT overwrite publication outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BAL = ROOT / "experimental-2026-06-23" / "01_primary_ocslab_balanced"
OUT = ROOT / "outputs" / "publication_metric_recovery" / "verification"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cm = pd.read_csv(BAL / "results" / "campaign_metrics.csv")
    p7 = pd.read_csv(BAL / "tables" / "table_P7_strong_campaign_results.csv")
    p8 = pd.read_csv(BAL / "tables" / "table_P8_weak_campaign_results.csv")

    report: dict = {"checks": [], "status": "ok"}

    # 1) Aggregation reproduces P7/P8 F1 headline numbers
    for label, sk, tab, expected_f1 in [
        ("P7", "strong_campaign", p7, [0.5333333333333333, 0.7333333333333333, 1.0]),
        ("P8", "weak_campaign", p8, [0.06666666666666667, 0.5, 0.7166666666666667]),
    ]:
        agg = (
            cm[cm["scenario_key"] == sk]
            .groupby("campaign_size", as_index=False)
            .mean(numeric_only=True)
            .sort_values("campaign_size")
        )
        f1_ok = np.allclose(agg["campaign_f1"].to_numpy(), tab["campaign_f1"].to_numpy(), atol=1e-12)
        headline_ok = np.allclose(tab["campaign_f1"].to_numpy(), expected_f1, atol=1e-9)
        cols = [
            "campaign_precision",
            "campaign_recall",
            "campaign_f1",
            "membership_precision",
            "membership_recall",
            "membership_f1",
            "fragmentation_rate",
            "fragments_per_true_campaign",
            "incorrect_merging_rate",
        ]
        all_cols_ok = all(
            np.allclose(agg[c].to_numpy(), tab[c].to_numpy(), atol=1e-12, equal_nan=True) for c in cols
        )
        report["checks"].append(
            {
                "name": f"{label}_aggregation_reproduces_table",
                "pass": bool(f1_ok and headline_ok and all_cols_ok),
                "f1_table": tab["campaign_f1"].tolist(),
                "f1_from_campaign_metrics_mean": agg["campaign_f1"].tolist(),
            }
        )

    # 2) Algebraic identities on strong+weak rows (artifact-derived; not source recovery)
    s = cm[cm["scenario_key"].isin(["strong_campaign", "weak_campaign"])].copy()
    s["tp_count"] = np.minimum(s["n_predicted_campaign_clusters"], s["n_true_campaign_clusters"])
    s["fp_count"] = np.maximum(s["n_predicted_campaign_clusters"] - s["n_true_campaign_clusters"], 0)
    s["fn_count"] = np.maximum(s["n_true_campaign_clusters"] - s["n_predicted_campaign_clusters"], 0)
    prec_count = np.where(
        s["n_predicted_campaign_clusters"] > 0,
        s["tp_count"] / s["n_predicted_campaign_clusters"],
        0.0,
    )
    rec_count = np.where(
        s["n_true_campaign_clusters"] > 0,
        s["tp_count"] / s["n_true_campaign_clusters"],
        0.0,
    )
    identities = {
        "precision_eq_min_npred_ntrue_over_npred": bool(np.allclose(s["campaign_precision"], prec_count)),
        "recall_eq_min_npred_ntrue_over_ntrue": bool(np.allclose(s["campaign_recall"], rec_count)),
        "fp_eq_max_npred_minus_ntrue_0": bool(
            (s["false_campaign_cluster_count"] == s["fp_count"]).all()
        ),
        "fn_eq_max_ntrue_minus_npred_0": bool((s["missed_campaign_count"] == s["fn_count"]).all()),
        "detection_rate_eq_recall": bool(
            np.allclose(s["campaign_detection_rate"], s["campaign_recall"])
        ),
        "completeness_eq_recall": bool(np.allclose(s["completeness"], s["campaign_recall"])),
        "membership_purity_eq_membership_precision": bool(
            np.allclose(s["membership_purity"], s["membership_precision"])
        ),
        "membership_recall_eq_attacked_ok_over_true_size": bool(
            np.allclose(
                s["membership_recall"],
                s["attacked_vehicles_correctly_included"] / s["true_campaign_size"],
            )
        ),
        "membership_precision_eq_attacked_ok_over_pred_size_when_pred_gt0": bool(
            np.allclose(
                s.loc[s["predicted_campaign_size"] > 0, "membership_precision"],
                (
                    s.loc[s["predicted_campaign_size"] > 0, "attacked_vehicles_correctly_included"]
                    / s.loc[s["predicted_campaign_size"] > 0, "predicted_campaign_size"]
                ),
            )
        ),
        "pred_size_eq_attacked_ok_plus_benign": bool(
            np.allclose(
                s["predicted_campaign_size"],
                s["attacked_vehicles_correctly_included"] + s["benign_vehicles_included"],
            )
        ),
        "true_size_eq_attacked_ok_plus_missed": bool(
            np.allclose(
                s["true_campaign_size"],
                s["attacked_vehicles_correctly_included"] + s["attacked_vehicles_missed"],
            )
        ),
        "frag_per_eq_n_predicted": bool(
            (s["fragments_per_true_campaign"] == s["n_predicted_campaign_clusters"]).all()
        ),
        "frag_rate_eq_npred_gt_ntrue": bool(
            (
                s["fragmentation_rate"]
                == (s["n_predicted_campaign_clusters"] > s["n_true_campaign_clusters"]).astype(float)
            ).all()
        ),
        "f1_eq_2pr_over_p_plus_r": bool(
            np.allclose(
                s["campaign_f1"],
                np.where(
                    (s["campaign_precision"] + s["campaign_recall"]) > 0,
                    2
                    * s["campaign_precision"]
                    * s["campaign_recall"]
                    / (s["campaign_precision"] + s["campaign_recall"]),
                    0.0,
                ),
            )
        ),
        "matched_implies_attacked_ok_gt0": bool(
            not (
                ((s["n_true_campaign_clusters"] - s["missed_campaign_count"]) > 0)
                & (s["attacked_vehicles_correctly_included"] == 0)
            ).any()
        ),
    }
    report["artifact_algebraic_identities"] = identities
    report["checks"].append(
        {
            "name": "artifact_algebraic_identities_all_hold",
            "pass": all(identities.values()),
            "n_strong_weak_rows": int(len(s)),
        }
    )

    # 3) Incorrect merging on unrelated
    u = cm[cm["scenario_key"] == "unrelated_incidents"]
    merge_mean = float(u["incorrect_merging_rate"].mean())
    report["checks"].append(
        {
            "name": "unrelated_incorrect_merging_mean_is_0_4",
            "pass": abs(merge_mean - 0.4) < 1e-12,
            "mean": merge_mean,
            "per_seed": u[["seed", "incorrect_merging_rate", "n_predicted_campaign_clusters"]].to_dict(
                "records"
            ),
            "note": "Rate = mean over 10 seeds of binary incorrect_merging; equals P6 0.400",
        }
    )

    # 4) Reproduction of executor: impossible without missing modules
    report["full_pipeline_reproduction"] = {
        "attempted": False,
        "reason": (
            "extract_run_metrics / run_refinement_fcgnn / SharedFleetConfiguration / "
            "_run_single_test never committed; cannot recompute fields from descriptors"
        ),
        "evidence_grade": "UNRECOVERABLE",
    }

    report["status"] = "pass" if all(c["pass"] for c in report["checks"]) else "fail"
    (OUT / "reproduction_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "out": str(OUT / "reproduction_check.json")}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
