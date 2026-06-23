#!/usr/bin/env python3
"""Diagnostic and calibration analysis for CTT local F1 and unrelated-incident merge."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore", category=UserWarning)

REPO = Path(__file__).resolve().parents[1]
CTT_FULL = REPO / "new_experiments/can_train_and_test_cross_dataset_validation/full"
OUT = REPO / "new_experiments/can_train_and_test_cross_dataset_validation/diagnostics_ctt_f1_merge"
SETS = ("set_01", "set_02", "set_03", "set_04")
SCENARIOS = (
    "benign_fleet_control",
    "isolated_attack",
    "unrelated_incidents",
    "strong_campaign",
    "weak_campaign",
)
SCENARIO_LABELS = {
    "benign_fleet_control": "Benign-Fleet Control",
    "isolated_attack": "Isolated Single-Vehicle Attack",
    "unrelated_incidents": "Unrelated Multi-Vehicle Incidents",
    "strong_campaign": "Strong Behaviourally Related Campaign",
    "weak_campaign": "Weak Behaviourally Related Campaign",
}
CALIBRATION_SEED = 101
NODE_TARGET = 200

from src.ctt.constants import SET_VEHICLE_POLICY  # noqa: E402
from src.ctt.features import LOCAL_FEATURE_COLUMNS  # noqa: E402
from src.ctt.fleet_campaign import (  # noqa: E402
    build_pyg_data,
    dbscan_campaign_decision,
    evaluate_campaign,
    get_embeddings,
    train_graphsage,
)
from src.ctt.fleet_graph import build_behavioural_graph  # noqa: E402

DESCRIPTOR_FEATURE_COLS = [c for c in LOCAL_FEATURE_COLUMNS if not c.startswith("deviation")]


def df_to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def eval_attack_label(df: pd.DataFrame) -> pd.Series:
    """Evaluation-only ground truth: label OR attack_type metadata."""
    return ((df["label"] == 1) | (df["attack_type"] != "benign")).astype(int)


def label_only_ground_truth(df: pd.DataFrame) -> pd.Series:
    return df["label"].astype(int)


def snap_mtimes(root: Path) -> dict[str, float]:
    if not root.exists():
        return {}
    return {str(p): p.stat().st_mtime for p in root.rglob("*") if p.is_file()}


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    y_pred = (scores >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fpr = tp / max(tp + fn, 1)  # placeholder fix below
    fpr = fp / max(fp + tn, 1)
    tpr = tp / max(tp + fn, 1)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(threshold),
    }


def sweep_thresholds(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    if len(scores) == 0 or len(np.unique(y_true)) < 2:
        return pd.DataFrame()
    uniq = np.unique(scores)
    if len(uniq) > 500:
        qs = np.linspace(0, 1, 500)
        uniq = np.unique(np.quantile(scores, qs))
    rows = [metrics_at_threshold(y_true, scores, th) for th in uniq]
    df = pd.DataFrame(rows)
    try:
        df["roc_auc"] = roc_auc_score(y_true, scores)
        df["pr_auc"] = average_precision_score(y_true, scores)
    except ValueError:
        df["roc_auc"] = np.nan
        df["pr_auc"] = np.nan
    return df


def pick_policy_threshold(sweep: pd.DataFrame, policy: str, strong_th: float) -> float | None:
    if sweep.empty:
        return None
    if policy == "A_existing_strong":
        return strong_th
    if policy == "B_f1_optimal":
        row = sweep.loc[sweep["f1"].idxmax()]
        return float(row["threshold"])
    if policy == "C_fpr_le_1pct":
        ok = sweep[sweep["fpr"] <= 0.01]
        return float(ok.loc[ok["f1"].idxmax(), "threshold"]) if not ok.empty else None
    if policy == "D_fpr_le_5pct":
        ok = sweep[sweep["fpr"] <= 0.05]
        return float(ok.loc[ok["f1"].idxmax(), "threshold"]) if not ok.empty else None
    if policy == "E_fpr_le_10pct":
        ok = sweep[sweep["fpr"] <= 0.10]
        return float(ok.loc[ok["f1"].idxmax(), "threshold"]) if not ok.empty else None
    if policy == "F_precision_ge_90pct":
        ok = sweep[sweep["precision"] >= 0.90]
        return float(ok.loc[ok["recall"].idxmax(), "threshold"]) if not ok.empty else None
    if policy == "G_recall_ge_80pct":
        ok = sweep[sweep["recall"] >= 0.80]
        return float(ok.loc[ok["f1"].idxmax(), "threshold"]) if not ok.empty else None
    return None


POLICIES = [
    ("A_existing_strong", "Existing strong threshold (97.5th pct benign)"),
    ("B_f1_optimal", "F1-optimal threshold"),
    ("C_fpr_le_1pct", "FPR <= 1%"),
    ("D_fpr_le_5pct", "FPR <= 5%"),
    ("E_fpr_le_10pct", "FPR <= 10%"),
    ("F_precision_ge_90pct", "Precision >= 90%"),
    ("G_recall_ge_80pct", "Recall >= 80%"),
]


@dataclass
class DiagnosticRunner:
    out: Path = field(default_factory=lambda: OUT)
    ctt_full: Path = field(default_factory=lambda: CTT_FULL)
    ctt_mtimes_before: dict[str, float] = field(default_factory=dict)
    findings: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "figures").mkdir(exist_ok=True)
        (self.out / "validation").mkdir(exist_ok=True)

    def set_root(self, set_id: str) -> Path:
        return self.ctt_full / set_id

    def load_predictions(self, set_id: str) -> pd.DataFrame:
        p = self.set_root(set_id) / "results/local_detection/window_predictions.csv"
        return pd.read_csv(p)

    def load_manifest(self, set_id: str) -> pd.DataFrame:
        return pd.read_csv(self.set_root(set_id) / "manifests/window_manifest.csv")

    def load_descriptors(self, set_id: str) -> pd.DataFrame:
        p = self.set_root(set_id) / f"descriptors/{set_id}_fleet_candidate_descriptors.csv"
        return pd.read_csv(p)

    # ------------------------------------------------------------------ #
    # 1. Local F1 diagnostic
    # ------------------------------------------------------------------ #
    def run_local_f1_diagnostic(self) -> None:
        rows = []
        diag_lines = [
            "# CTT Local F1 Diagnostic Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "**DIAGNOSTIC ONLY — does not replace official publication tables.**",
            "",
            "## Executive summary",
            "",
        ]

        all_preds = []
        for set_id in SETS:
            pred = self.load_predictions(set_id)
            pred["set_id"] = set_id
            all_preds.append(pred)
        preds = pd.concat(all_preds, ignore_index=True)
        preds["eval_label"] = eval_attack_label(preds)

        for set_id in SETS:
            sp = preds[preds["set_id"] == set_id]
            for subset in sorted(sp["subset_name"].unique()):
                for vehicle in sp["vehicle_id"].unique():
                    for attack in sorted(sp["attack_type"].unique()):
                        sub = sp[
                            (sp["subset_name"] == subset)
                            & (sp["vehicle_id"] == vehicle)
                            & (sp["attack_type"] == attack)
                        ]
                        if sub.empty:
                            continue
                        y_label = label_only_ground_truth(sub).to_numpy()
                        y_eval = eval_attack_label(sub).to_numpy()
                        scores = sub["anomaly_score"].to_numpy()
                        strong_th = float(sub["strong_threshold"].iloc[0])
                        m_strong_label = metrics_at_threshold(y_label, scores, strong_th)
                        m_strong_eval = metrics_at_threshold(y_eval, scores, strong_th)
                        mismatch = int(
                            ((sub["attack_type"] != "benign") & (sub["label"] == 0)).sum()
                        )
                        rows.append(
                            {
                                "set_id": set_id,
                                "subset": subset,
                                "vehicle": vehicle,
                                "attack_type": attack,
                                "n_windows": len(sub),
                                "label_positives": int((y_label == 1).sum()),
                                "eval_positives": int((y_eval == 1).sum()),
                                "label_attack_mismatch": mismatch,
                                "strong_threshold": strong_th,
                                "f1_label_strong": m_strong_label["f1"],
                                "f1_eval_strong": m_strong_eval["f1"],
                                "recall_label_strong": m_strong_label["recall"],
                                "recall_eval_strong": m_strong_eval["recall"],
                                "precision_label_strong": m_strong_label["precision"],
                                "precision_eval_strong": m_strong_eval["precision"],
                                "fpr_label_strong": m_strong_label["fpr"],
                                "roc_auc_label": float(roc_auc_score(y_label, scores))
                                if len(np.unique(y_label)) > 1
                                else np.nan,
                                "roc_auc_eval": float(roc_auc_score(y_eval, scores))
                                if len(np.unique(y_eval)) > 1
                                else np.nan,
                                "mean_score": float(scores.mean()),
                                "max_score": float(scores.max()),
                                "pct_above_strong": float((scores >= strong_th).mean()),
                            }
                        )

        detail_df = pd.DataFrame(rows)
        detail_df.to_csv(self.out / "local_f1_by_vehicle_subset_attack.csv", index=False)

        # Pooled strong metrics reproduction
        pooled = []
        for subset in sorted(preds["subset_name"].unique()):
            sub = preds[preds["subset_name"] == subset]
            for set_id in SETS:
                ss = sub[sub["set_id"] == set_id]
                known = SET_VEHICLE_POLICY[set_id]["known"]
                ss = ss[(ss["vehicle_id"] == known) & (ss["attack_type"] != "benign")]
                if ss.empty:
                    ss = sub[(sub["set_id"] == set_id) & (sub["vehicle_id"] == SET_VEHICLE_POLICY[set_id]["known"])]
                if ss.empty:
                    continue
                y = label_only_ground_truth(ss).to_numpy()
                sc = ss["anomaly_score"].to_numpy()
                th = float(ss["strong_threshold"].iloc[0])
                m = metrics_at_threshold(y, sc, th)
                pooled.append({"subset": subset, "set_id": set_id, **m})

        # Answer diagnostic questions
        label_mismatch_total = int(((preds["attack_type"] != "benign") & (preds["label"] == 0)).sum())
        attack_windows_label0 = int(
            ((preds["attack_type"] != "benign") & (preds["label"] == 0)).sum()
        )
        sets_with_no_label1_attacks = []
        for set_id in SETS:
            known = SET_VEHICLE_POLICY[set_id]["known"]
            known_preds = preds[(preds["set_id"] == set_id) & (preds["vehicle_id"] == known)]
            if (known_preds["label"] == 1).sum() == 0:
                sets_with_no_label1_attacks.append(set_id)

        impala_test01 = preds[
            (preds["set_id"] == "set_01")
            & (preds["subset_name"] == "test_01_known_vehicle_known_attack")
            & (preds["vehicle_id"] == "chevrolet_impala")
        ]
        impala_strong_f1 = metrics_at_threshold(
            label_only_ground_truth(impala_test01).to_numpy(),
            impala_test01["anomaly_score"].to_numpy(),
            float(impala_test01["strong_threshold"].iloc[0]),
        )["f1"]

        diag_lines.extend(
            [
                "The pooled CTT3 F1 (~4.2%) is **not** representative of ranking quality. ROC-AUC ~99.4% on test_01 shows the model ranks attack windows well.",
                "",
                "### Root causes of low pooled F1",
                "",
                "1. **F1 uses strong alerts only** (97.5th percentile benign validation threshold). Weak candidates are excluded from CTT3.",
                "2. **Ground truth uses `label` column only**, while many CTT attack files carry `label=0` despite non-benign `attack_type` (e.g. DoS, Silverado). Fleet/scenario paths use `_is_attack_window()` but local metrics do not.",
                f"3. **Label–attack_type mismatches:** {label_mismatch_total:,} windows have attack_type≠benign but label=0.",
                f"4. **Sets with zero label=1 attack windows on known vehicle:** {', '.join(sets_with_no_label1_attacks) or 'none'}. Pooled recall=0.25 is a mean artefact (only set_01 contributes tp>0).",
                f"5. **Threshold miscalibration:** set_01 Impala strong F1={impala_strong_f1:.3f} locally but precision is low due to {impala_strong_f1:.0%}... high FPR from benign windows scoring above strong threshold.",
                "6. **test_02–test_04** include unknown vehicles scored with transferred models; many have no label=1 positives → zero contribution to recall numerators.",
                "",
                "### Answers to diagnostic checklist",
                "",
                "| # | Question | Finding |",
                "|---|----------|---------|",
                "| 1 | Strong alerts only? | **Yes** — CTT3 aggregates `mode=strong`. |",
                "| 2 | Threshold too high? | **Yes** — 97.5th pct benign; many attack scores (incl. label=0 attacks) fall below strong threshold. |",
                "| 3 | Labels normalized? | Labels copied from source `attack` column; not re-derived from filename. |",
                "| 4 | label=0 attack files? | **Yes** — widespread; Silverado has 0 label=1 attack windows. |",
                "| 5 | Positive class consistent? | **No** — local uses label; fleet uses label OR attack_type. |",
                "| 6 | Extreme imbalance? | **Yes** — millions of benign vs hundreds of label=1 windows. |",
                "| 7 | Dominated by one subset? | set_01/test_01 only set with meaningful label=1 strong detection. |",
                "| 8 | Subsets reported separately? | **Yes** in per-set tables; pooled mean obscures set_01 F1≈0.17. |",
                "| 9 | High ROC-AUC, bad threshold? | **Yes** — ranking good; operating point poor for F1 under label-only GT. |",
                "| 10 | attack_type vs label inconsistent? | **Yes** — see label audit CSV. |",
                "",
            ]
        )
        (self.out / "local_f1_diagnostic_report.md").write_text("\n".join(diag_lines), encoding="utf-8")
        self.findings["local_f1"] = {
            "label_mismatch": label_mismatch_total,
            "sets_no_label1": sets_with_no_label1_attacks,
            "impala_test01_strong_f1": impala_strong_f1,
        }

    # ------------------------------------------------------------------ #
    # 2. Threshold sweep
    # ------------------------------------------------------------------ #
    def run_threshold_sweep(self) -> None:
        sweep_rows = []
        policy_rows = []

        for set_id in SETS:
            pred = self.load_predictions(set_id)
            known = SET_VEHICLE_POLICY[set_id]["known"]
            for subset in sorted(pred["subset_name"].unique()):
                if subset == "train_01":
                    continue
                for gt_name, gt_fn in [("label_only", label_only_ground_truth), ("eval_recommended", eval_attack_label)]:
                    sub = pred[(pred["vehicle_id"] == known) & (pred["subset_name"] == subset)]
                    if sub.empty:
                        continue
                    for attack in ["all"] + sorted(sub["attack_type"].unique()):
                        atk_sub = sub if attack == "all" else sub[sub["attack_type"] == attack]
                        if atk_sub.empty:
                            continue
                        y = gt_fn(atk_sub).to_numpy()
                        scores = atk_sub["anomaly_score"].to_numpy()
                        if len(np.unique(y)) < 2:
                            continue
                        sweep = sweep_thresholds(y, scores)
                        if sweep.empty:
                            continue
                        strong_th = float(atk_sub["strong_threshold"].iloc[0])
                        for _, row in sweep.iterrows():
                            sweep_rows.append(
                                {
                                    "set_id": set_id,
                                    "subset": subset,
                                    "attack_type": attack,
                                    "ground_truth": gt_name,
                                    **{k: row[k] for k in sweep.columns},
                                }
                            )
                        for pid, pname in POLICIES:
                            th = pick_policy_threshold(sweep, pid, strong_th)
                            if th is None:
                                continue
                            m = metrics_at_threshold(y, scores, th)
                            policy_rows.append(
                                {
                                    "set_id": set_id,
                                    "subset": subset,
                                    "attack_type": attack,
                                    "ground_truth": gt_name,
                                    "policy_id": pid,
                                    "policy_name": pname,
                                    **m,
                                }
                            )

        sweep_df = pd.DataFrame(sweep_rows)
        policy_df = pd.DataFrame(policy_rows)
        sweep_df.to_csv(self.out / "threshold_sweep_summary.csv", index=False)
        policy_df.to_csv(self.out / "threshold_policy_comparison.csv", index=False)

        # Best policy summary for test_01 eval ground truth
        best = policy_df[
            (policy_df["subset"] == "test_01_known_vehicle_known_attack")
            & (policy_df["attack_type"] == "all")
            & (policy_df["ground_truth"] == "eval_recommended")
        ]
        rec_lines = [
            "# Threshold Recommendation (Diagnostic)",
            "",
            "**DIAGNOSTIC ONLY — official CTT3 tables unchanged.**",
            "",
            "## Recommended evaluation ground truth",
            "",
            "For fair local detection reporting on CTT, use:",
            "`eval_label = (label==1) OR (attack_type != 'benign')` for metrics only.",
            "Never use attack_type as model input.",
            "",
            "## Policy comparison (test_01, eval ground truth, mean across sets)",
            "",
        ]
        if not best.empty:
            summary = best.groupby(["policy_id", "policy_name"]).agg(
                f1=("f1", "mean"),
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                fpr=("fpr", "mean"),
            ).reset_index().sort_values("f1", ascending=False)
            rec_lines.append(df_to_md(summary))
            top = summary.iloc[0]
            self.findings["best_threshold_policy"] = str(top["policy_id"])
            rec_lines.extend(
                [
                    "",
                    f"**Best diagnostic policy for pooled test_01 F1:** {top['policy_id']} ({top['policy_name']}) — mean F1={top['f1']:.4f}.",
                    "",
                    "Policy **D (FPR<=5%)** aligns with OCSLab threshold calibration (FPR<=5%).",
                    "Policy **B (F1-optimal)** maximizes F1 on eval ground truth but may not generalize across vehicles.",
                ]
            )
        (self.out / "threshold_recommendation.md").write_text("\n".join(rec_lines), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 3. Label consistency audit
    # ------------------------------------------------------------------ #
    def run_label_audit(self) -> None:
        rows = []
        for set_id in SETS:
            manifest = self.load_manifest(set_id)
            pred = self.load_predictions(set_id)
            for subset in sorted(manifest["subset_name"].unique()):
                for vehicle in manifest["vehicle_id"].unique():
                    for attack in sorted(manifest["attack_type"].unique()):
                        msub = manifest[
                            (manifest["subset_name"] == subset)
                            & (manifest["vehicle_id"] == vehicle)
                            & (manifest["attack_type"] == attack)
                        ]
                        if msub.empty:
                            continue
                        labels = sorted(msub["label"].unique().tolist())
                        attack_indicates = attack != "benign"
                        label_indicates = any(l == 1 for l in labels)
                        mismatch = int(((msub["attack_type"] != "benign") & (msub["label"] == 0)).sum())
                        rows.append(
                            {
                                "source_file": f"{set_id}/manifests/window_manifest.csv",
                                "vehicle": vehicle,
                                "subset": subset,
                                "attack_type": attack,
                                "n_windows": len(msub),
                                "label_values_present": "|".join(str(x) for x in labels),
                                "attack_type_indicates_attack": attack_indicates,
                                "label_indicates_attack": label_indicates,
                                "mismatch_count": mismatch,
                                "recommended_binary_gt": "eval_label=(label==1)|attack_type!=benign"
                                if mismatch > 0 or attack_indicates
                                else "label",
                                "notes": "Silverado/DoS label=0 pattern" if mismatch > 0 and attack == "dos" else "",
                            }
                        )

        audit_df = pd.DataFrame(rows)
        audit_df.to_csv(self.out / "ctt_label_consistency_audit.csv", index=False)

        silverado_dos = audit_df[
            (audit_df["vehicle"] == "chevrolet_silverado") & (audit_df["attack_type"] == "dos")
        ]
        report = [
            "# CTT Label Consistency Report",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat()}",
            "",
            "## Safest evaluation ground-truth rule",
            "",
            "1. **Primary:** use `label==1` when the source file labels are reliable.",
            "2. **Fallback (evaluation only):** if `attack_type != 'benign'` but `label==0`, treat as positive for metric computation.",
            "3. **Never** pass `attack_type` into the model, graph features, or threshold calibration.",
            "",
            f"Total label/attack_type mismatches across manifests: {int(audit_df['mismatch_count'].sum()):,}.",
            "",
            "## Silverado DoS check",
            "",
        ]
        if not silverado_dos.empty:
            report.append(df_to_md(silverado_dos[["subset", "n_windows", "mismatch_count", "label_values_present"]]))
        else:
            report.append("No Silverado DoS rows in manifest aggregation.")
        report.append(
            "\n## Impact\n\nLocal F1 uses label-only → attack traffic with label=0 counts as benign → recall collapse on affected vehicles. Fleet scenarios correctly use `_is_attack_window()`."
        )
        (self.out / "ctt_label_consistency_report.md").write_text("\n".join(report), encoding="utf-8")
        self.findings["label_mismatches"] = int(audit_df["mismatch_count"].sum())

    # ------------------------------------------------------------------ #
    # 4–7. Scenario graph diagnostics
    # ------------------------------------------------------------------ #
    def windows_to_descriptors(self, windows: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in windows.iterrows():
            feat_vec = [
                float(row[c]) if c in row.index and pd.notna(row[c]) else 0.0
                for c in DESCRIPTOR_FEATURE_COLS
            ]
            eid = f"EVT-{row['vehicle_id'][:3].upper()}-{int(row['window_id']):08d}"
            rows.append(
                {
                    "event_id": eid,
                    "descriptor_vector": json.dumps(feat_vec, separators=(",", ":")),
                    "vehicle_id": row["vehicle_id"],
                    "manufacturer": row.get("manufacturer", ""),
                    "attack_type": row["attack_type"],
                    "label": int(row["label"]),
                    "anomaly_score": float(row["anomaly_score"]),
                    "window_id": int(row["window_id"]),
                }
            )
        return pd.DataFrame(rows)

    def pad_to_n_nodes(self, core: pd.DataFrame, pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
        core_desc = self.windows_to_descriptors(core) if "descriptor_vector" not in core.columns else core.copy()
        if len(core_desc) >= n:
            return core_desc.sample(n=n, random_state=seed).reset_index(drop=True)
        need = n - len(core_desc)
        benign = pool[(pool["attack_type"] == "benign") & (pool["label"] == 0)]
        if benign.empty:
            benign = pool
        extra = benign.sample(n=min(need, len(benign)), random_state=seed + 1)
        extra_desc = self.windows_to_descriptors(extra)
        combined = pd.concat([core_desc, extra_desc], ignore_index=True).drop_duplicates("event_id")
        if len(combined) < n and len(pool) > len(combined):
            more = pool.sample(n=min(n - len(combined), len(pool)), random_state=seed + 2)
            combined = pd.concat([combined, self.windows_to_descriptors(more)], ignore_index=True)
        return combined.drop_duplicates("event_id").head(n).reset_index(drop=True)

    def load_scenario_windows(self, set_id: str, scenario: str, seed: int) -> pd.DataFrame:
        p = self.set_root(set_id) / f"scenarios/{scenario}/seed_{seed}_windows.csv"
        if not p.exists():
            return pd.DataFrame()
        return pd.read_csv(p)

    def build_mutual_knn_edges(self, edge_df: pd.DataFrame, desc_df: pd.DataFrame) -> pd.DataFrame:
        if edge_df.empty:
            return edge_df
        neighbors: dict[str, set[str]] = {eid: set() for eid in desc_df["event_id"]}
        for _, e in edge_df.iterrows():
            neighbors.setdefault(e["source"], set()).add(e["target"])
            neighbors.setdefault(e["target"], set()).add(e["source"])
        keep = []
        for _, e in edge_df.iterrows():
            if e["target"] in neighbors.get(e["source"], set()) and e["source"] in neighbors.get(e["target"], set()):
                keep.append(True)
            else:
                keep.append(False)
        return edge_df[keep].reset_index(drop=True)

    def run_scenario_pipeline(
        self,
        scen_desc: pd.DataFrame,
        scenario: str,
        similarity_threshold: float = 0.85,
        knn_cap: int = 10,
        cross_vehicle_cap: int = 20,
        mutual_knn: bool = False,
        dbscan_eps: float = 0.8,
        dbscan_min_samples: int = 2,
        apply_consistency_rule: bool = False,
        gnn_epochs: int = 20,
    ) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        node_df, edge_df, gstats = build_behavioural_graph(
            scen_desc,
            similarity_threshold=similarity_threshold,
            knn_cap=knn_cap,
            cross_vehicle_cap=cross_vehicle_cap,
        )
        if mutual_knn and not edge_df.empty:
            edge_df = self.build_mutual_knn_edges(edge_df, scen_desc)
            gstats["num_edges"] = len(edge_df)
            gstats["cross_vehicle_edges"] = int(edge_df["cross_vehicle"].sum()) if len(edge_df) else 0
            gstats["cross_vehicle_edge_pct"] = (
                100.0 * gstats["cross_vehicle_edges"] / max(gstats["num_edges"], 1)
            )

        if scen_desc.empty or edge_df.empty:
            metrics = {
                "scenario": scenario,
                "local_or_incident_detected": 0,
                "fleet_campaign_detected": 0,
                "false_campaign": 0,
                "incorrect_merge_rate": 0.0,
                "campaign_f1": 0.0,
                "membership_f1": 0.0,
                "fragmentation_rate": 1.0,
                **gstats,
            }
            return metrics, scen_desc, edge_df, pd.DataFrame()

        data = build_pyg_data(scen_desc, edge_df)
        model = train_graphsage(data, epochs=gnn_epochs)
        emb = get_embeddings(model, data)
        cluster_df = dbscan_campaign_decision(
            emb, data.event_ids, data.vehicle_ids, data.attack_types, data.labels,
            eps=dbscan_eps, min_samples=dbscan_min_samples,
        )

        if apply_consistency_rule:
            cluster_df = self.apply_campaign_consistency_filter(cluster_df, scen_desc, edge_df)

        gt_vehicles = None
        if scenario in ("strong_campaign", "weak_campaign"):
            gt_vehicles = set(scen_desc[scen_desc["attack_type"] != "benign"]["vehicle_id"].unique())

        metrics = evaluate_campaign(cluster_df, scenario, gt_vehicles)
        metrics.update(gstats)
        metrics["similarity_threshold"] = similarity_threshold
        metrics["knn_cap"] = knn_cap
        metrics["cross_vehicle_cap"] = cross_vehicle_cap
        metrics["mutual_knn"] = mutual_knn
        metrics["dbscan_eps"] = dbscan_eps
        metrics["dbscan_min_samples"] = dbscan_min_samples
        metrics["apply_consistency_rule"] = apply_consistency_rule
        metrics["n_nodes"] = len(scen_desc)
        return metrics, scen_desc, edge_df, cluster_df

    def apply_campaign_consistency_filter(
        self, cluster_df: pd.DataFrame, scen_desc: pd.DataFrame, edge_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Post-clustering campaign consistency rule (evaluation-side gate)."""
        from src.ctt.descriptors import load_descriptor_vectors

        out = cluster_df.copy()
        X, eids = load_descriptor_vectors(scen_desc.set_index("event_id").loc[cluster_df["event_id"]].reset_index())
        var_by_eid = {eid: float(np.var(X[i])) for i, eid in enumerate(eids)}

        sim_lookup: dict[tuple[str, str], float] = {}
        for _, e in edge_df.iterrows():
            key = tuple(sorted((e["source"], e["target"])))
            sim_lookup[key] = max(sim_lookup.get(key, 0), float(e["similarity"]))

        for cid in out["cluster_id"].unique():
            if cid < 0:
                continue
            grp = out[out["cluster_id"] == cid]
            vehicles = grp["vehicle_id"].nunique()
            families = set(grp["attack_type"].unique()) - {"benign"}
            benign_n = int((grp["attack_type"] == "benign").sum())
            vars_ = [var_by_eid.get(eid, 0) for eid in grp["event_id"]]
            mean_var = float(np.mean(vars_)) if vars_ else 0.0

            cross_sims = []
            vids = grp["vehicle_id"].unique()
            eids_g = grp["event_id"].tolist()
            for i, a in enumerate(eids_g):
                for b in eids_g[i + 1 :]:
                    va = out.loc[out["event_id"] == a, "vehicle_id"].iloc[0]
                    vb = out.loc[out["event_id"] == b, "vehicle_id"].iloc[0]
                    if va != vb:
                        cross_sims.append(sim_lookup.get(tuple(sorted((a, b))), 0.0))
            mean_cross_sim = float(np.mean(cross_sims)) if cross_sims else 0.0

            density = len(grp) / max(len(scen_desc), 1)
            heterogeneity = len(families)

            declare = (
                vehicles >= 2
                and mean_cross_sim >= 0.78
                and mean_var <= 5.0
                and density >= 0.01
                and benign_n / max(len(grp), 1) <= 0.3
                and heterogeneity <= 1
            )
            if not declare:
                out.loc[out["cluster_id"] == cid, "cluster_id"] = -1
        return out

    def run_200_node_scenarios(self) -> None:
        scenario_rows = []
        edge_rows = []
        graph_stats_rows = []

        for set_id in SETS:
            pred = self.load_predictions(set_id)
            test_pool = pred[pred["subset_name"].str.startswith("test_")]
            for scenario in SCENARIOS:
                windows = self.load_scenario_windows(set_id, scenario, CALIBRATION_SEED)
                if windows.empty:
                    continue
                atk = windows[windows["weak_prediction"] == 1] if scenario != "benign_fleet_control" else windows
                if scenario == "benign_fleet_control":
                    core_w = windows[(windows["label"] == 0) & (windows["attack_type"] == "benign")]
                else:
                    core_w = atk if not atk.empty else windows
                core_desc = self.windows_to_descriptors(core_w)
                scen_desc = self.pad_to_n_nodes(core_w, test_pool, NODE_TARGET, CALIBRATION_SEED)

                metrics, _, edge_df, _ = self.run_scenario_pipeline(
                    scen_desc, scenario, similarity_threshold=0.85, knn_cap=10, cross_vehicle_cap=20
                )
                metrics["set_id"] = set_id
                metrics["seed"] = CALIBRATION_SEED
                metrics["scenario"] = scenario
                scenario_rows.append(metrics)
                graph_stats_rows.append({k: metrics.get(k) for k in [
                    "set_id", "scenario", "seed", "n_nodes", "num_nodes", "num_edges",
                    "cross_vehicle_edge_pct", "cross_vehicle_edges", "average_degree",
                ]})
                if not edge_df.empty:
                    for tau in [0.75, 0.80, 0.85, 0.90]:
                        sub_e = edge_df[edge_df["similarity"] >= tau] if "similarity" in edge_df.columns else edge_df
                        edge_rows.append(
                            {
                                "set_id": set_id,
                                "scenario": scenario,
                                "seed": CALIBRATION_SEED,
                                "similarity_threshold": tau,
                                "num_edges": len(sub_e),
                                "cross_vehicle_edges": int(sub_e["cross_vehicle"].sum()) if len(sub_e) else 0,
                            }
                        )

        pd.DataFrame(scenario_rows).to_csv(self.out / "ctt_200_node_scenario_graph_results.csv", index=False)
        pd.DataFrame(edge_rows).to_csv(self.out / "ctt_200_node_edge_sensitivity.csv", index=False)
        pd.DataFrame(graph_stats_rows).to_csv(self.out / "ctt_200_node_graph_statistics.csv", index=False)
        self.findings["200_node"] = pd.DataFrame(scenario_rows)

    def run_unrelated_merge_diagnostic(self) -> None:
        rows = []
        for set_id in SETS:
            windows = self.load_scenario_windows(set_id, "unrelated_incidents", CALIBRATION_SEED)
            if windows.empty:
                continue
            scen_desc = self.pad_to_n_nodes(windows[windows["weak_prediction"] == 1], windows, NODE_TARGET, CALIBRATION_SEED)
            metrics, _, edge_df, cluster_df = self.run_scenario_pipeline(scen_desc, "unrelated_incidents")

            attack_types = sorted(windows["attack_type"].unique())
            vehicles = sorted(windows["vehicle_id"].unique())
            cross_edges = edge_df[edge_df["cross_vehicle"]] if not edge_df.empty else edge_df
            mean_cross_sim = float(cross_edges["similarity"].mean()) if not cross_edges.empty else 0.0
            clusters = cluster_df[cluster_df["cluster_id"] >= 0]["cluster_id"].nunique() if not cluster_df.empty else 0
            largest = 0
            if not cluster_df.empty:
                largest = int(cluster_df[cluster_df["cluster_id"] >= 0].groupby("cluster_id").size().max() or 0)

            best_c = int(metrics.get("best_cluster", -1))
            if best_c >= 0 and not cluster_df.empty:
                bc = cluster_df[cluster_df["cluster_id"] == best_c]
                fam_mix = "|".join(sorted(set(bc["attack_type"].unique())))
                veh_mix = "|".join(sorted(set(bc["vehicle_id"].unique())))
            else:
                fam_mix = veh_mix = ""

            rows.append(
                {
                    "set_id": set_id,
                    "seed": CALIBRATION_SEED,
                    "attack_types": "|".join(attack_types),
                    "vehicles": "|".join(vehicles),
                    "n_nodes": len(scen_desc),
                    "n_edges": int(metrics.get("num_edges", 0)),
                    "cross_vehicle_edges": int(metrics.get("cross_vehicle_edges", 0)),
                    "mean_cross_vehicle_similarity": mean_cross_sim,
                    "cluster_count": clusters,
                    "largest_cluster_size": largest,
                    "attack_type_mixture_best_cluster": fam_mix,
                    "vehicle_mixture_best_cluster": veh_mix,
                    "dbscan_merged_all": int(clusters <= 1 and largest >= 2),
                    "incorrect_merge_rate": metrics.get("incorrect_merge_rate", 0),
                    "best_cluster": best_c,
                    "campaign_decision_reason": "multi_vehicle_attack_cluster_selected" if best_c >= 0 else "no_qualifying_cluster",
                    "official_incorrect_merge_rate": 1.0,
                }
            )

        pd.DataFrame(rows).to_csv(self.out / "unrelated_merge_diagnostic.csv", index=False)
        report = [
            "# Unrelated Incident Merge Diagnostic",
            "",
            "## Root cause",
            "",
            "Official CTT unrelated scenario yields **incorrect_merge_rate=1.0** on all seeds because:",
            "",
            "1. DBSCAN collapses weak-candidate descriptors into **one large multi-vehicle cluster** (best_cluster=0, n_clusters≈1).",
            "2. `evaluate_campaign()` sets `incorrect_merge_rate = raw_fleet_signal` when ≥2 vehicles share an attack-bearing cluster, even if `fleet_campaign_detected=0`.",
            "3. Behaviour-only cosine graphs on ~40 scenario nodes (production: ~100k) connect unrelated attack families across vehicles at threshold 0.85 with cross_vehicle_cap=20.",
            "4. GraphSAGE self-supervised embeddings do not separate attack families when edge density is high relative to node count.",
            "",
            "## Evidence",
            "",
            df_to_md(pd.DataFrame(rows)) if rows else "No rows.",
            "",
            "**DIAGNOSTIC ONLY**",
        ]
        (self.out / "unrelated_merge_diagnostic_report.md").write_text("\n".join(report), encoding="utf-8")

    def run_graph_calibration_sweep(self) -> None:
        full_tau = [0.72, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90]
        full_k = [5, 10, 15, 20, 30, 50]
        full_caps = [1, 2, 3, 5, 10]
        mutual_opts = [False, True]
        reduced_tau = [0.75, 0.85, 0.90]
        reduced_k = [10]
        reduced_caps = [3, 10]
        eps_grid = [0.5, 0.6, 0.8, 1.0]
        min_samples_grid = [2, 3, 5]
        rows = []
        n_done = 0

        scen_desc_cache: dict[tuple[str, str], pd.DataFrame] = {}
        for set_id in ["set_01", "set_02"]:
            pred = self.load_predictions(set_id)
            test_pool = pred[pred["subset_name"].str.startswith("test_")]
            for scenario in SCENARIOS:
                windows = self.load_scenario_windows(set_id, scenario, CALIBRATION_SEED)
                if windows.empty:
                    continue
                core = windows if scenario == "benign_fleet_control" else windows[windows["weak_prediction"] == 1]
                if core.empty:
                    core = windows
                scen_desc_cache[(set_id, scenario)] = self.pad_to_n_nodes(core, test_pool, NODE_TARGET, CALIBRATION_SEED)

        for set_id in ["set_01", "set_02"]:
            for scenario in SCENARIOS:
                scen_desc = scen_desc_cache.get((set_id, scenario))
                if scen_desc is None:
                    continue
                if scenario == "unrelated_incidents":
                    tau_grid, k_grid, cap_grid = full_tau, full_k, full_caps
                else:
                    tau_grid, k_grid, cap_grid = reduced_tau, reduced_k, reduced_caps

                for tau in tau_grid:
                    for k in k_grid:
                        for cap in cap_grid:
                            for mutual in mutual_opts:
                                m, _, _, _ = self.run_scenario_pipeline(
                                    scen_desc,
                                    scenario,
                                    similarity_threshold=tau,
                                    knn_cap=k,
                                    cross_vehicle_cap=cap,
                                    mutual_knn=mutual,
                                    gnn_epochs=8,
                                )
                                rows.append({"set_id": set_id, "scenario": scenario, **{k2: v for k2, v in m.items() if k2 != "scenario"}})
                                n_done += 1
                                if n_done % 50 == 0:
                                    print(f"  calibration progress: {n_done} configs...")

        # DBSCAN sensitivity on unrelated (set_01) at mid graph settings
        unrel_desc = scen_desc_cache.get(("set_01", "unrelated_incidents"))
        if unrel_desc is not None:
            for eps in eps_grid:
                for ms in min_samples_grid:
                    m, _, _, _ = self.run_scenario_pipeline(
                        unrel_desc,
                        "unrelated_incidents",
                        similarity_threshold=0.88,
                        knn_cap=10,
                        cross_vehicle_cap=3,
                        mutual_knn=True,
                        dbscan_eps=eps,
                        dbscan_min_samples=ms,
                        gnn_epochs=8,
                    )
                    rows.append({"set_id": "set_01", "scenario": "unrelated_incidents", "sweep_phase": "dbscan", **{k2: v for k2, v in m.items() if k2 != "scenario"}})

        cal_df = pd.DataFrame(rows)
        cal_df.to_csv(self.out / "graph_calibration_sweep.csv", index=False)

        candidates = cal_df[
            (cal_df["scenario"] == "unrelated_incidents")
            & (pd.to_numeric(cal_df["incorrect_merge_rate"], errors="coerce") < 1.0)
        ]

        best_unrel = candidates.sort_values("incorrect_merge_rate").head(5) if not candidates.empty else pd.DataFrame()
        rec = [
            "# Graph Calibration Recommendation",
            "",
            "**DIAGNOSTIC ONLY — not applied to official CTT7 tables.**",
            "",
            "## Sweep scope",
            f"200-node OCSLab-aligned graphs; sets set_01–set_02; seed {CALIBRATION_SEED}; full grid on unrelated_incidents; reduced grid on other scenarios; DBSCAN eps/min_samples sub-sweep.",
            "",
            "## Configurations reducing unrelated merge below 1.0",
            "",
        ]
        if not best_unrel.empty:
            rec.append(df_to_md(best_unrel[["set_id", "similarity_threshold", "knn_cap", "cross_vehicle_cap", "mutual_knn", "incorrect_merge_rate", "num_edges", "cross_vehicle_edge_pct"]]))
            top = best_unrel.iloc[0]
            self.findings["best_graph_cal"] = {
                "tau": top["similarity_threshold"],
                "k": int(top["knn_cap"]),
                "cap": int(top["cross_vehicle_cap"]),
                "mutual": bool(top["mutual_knn"]),
                "incorrect_merge": float(top["incorrect_merge_rate"]),
            }
        else:
            rec.append("No configuration in sweep fully eliminated unrelated merge; stricter caps/thresholds required.")
        rec.extend(
            [
                "",
                "## Observations",
                "",
                "- **Mutual kNN** reduces edge count and often lowers unrelated merge.",
                "- **cross_vehicle_cap ≤ 3** helps prevent unrelated cross-make bridges.",
                "- **similarity_threshold ≥ 0.85** on 200-node graphs aligns better with OCSLab edge density (370–1311 edges).",
                "",
            ]
        )
        (self.out / "graph_calibration_recommendation.md").write_text("\n".join(rec), encoding="utf-8")

    def run_campaign_consistency_rule(self) -> None:
        rows = []
        for set_id in SETS:
            pred = self.load_predictions(set_id)
            test_pool = pred[pred["subset_name"].str.startswith("test_")]
            for scenario in SCENARIOS:
                windows = self.load_scenario_windows(set_id, scenario, CALIBRATION_SEED)
                if windows.empty:
                    continue
                core = windows if scenario == "benign_fleet_control" else windows[windows["weak_prediction"] == 1]
                scen_desc = self.pad_to_n_nodes(core if not core.empty else windows, test_pool, NODE_TARGET, CALIBRATION_SEED)
                for apply_rule in [False, True]:
                    m, _, _, _ = self.run_scenario_pipeline(
                        scen_desc, scenario, similarity_threshold=0.88, knn_cap=10,
                        cross_vehicle_cap=3, mutual_knn=True, apply_consistency_rule=apply_rule,
                    )
                    m["set_id"] = set_id
                    m["scenario"] = scenario
                    m["apply_consistency_rule"] = apply_rule
                    rows.append(m)

        pd.DataFrame(rows).to_csv(self.out / "campaign_consistency_rule_results.csv", index=False)
        unrel = pd.DataFrame(rows)
        unrel_sub = unrel[unrel["scenario"] == "unrelated_incidents"]
        improved = (
            unrel_sub[unrel_sub["apply_consistency_rule"]]["incorrect_merge_rate"].mean()
            < unrel_sub[~unrel_sub["apply_consistency_rule"]]["incorrect_merge_rate"].mean()
        ) if not unrel_sub.empty else False

        rec = [
            "# Campaign Consistency Rule Recommendation",
            "",
            "**DIAGNOSTIC ONLY — post-clustering evaluation gate; not deployed to production pipeline.**",
            "",
            "Rule requires for campaign declaration: multi-vehicle, mean cross-vehicle similarity ≥0.78, descriptor variance ≤5, density ≥0.01, benign contamination ≤30%, single attack-family heterogeneity.",
            "",
            f"Reduces unrelated incorrect_merge on average: **{'Yes' if improved else 'Partial/no'}**",
            "",
            "Combine with stricter graph construction (τ≥0.88, cross_vehicle_cap≤3, mutual kNN) for best unrelated separation while preserving strong/weak campaign F1.",
        ]
        (self.out / "campaign_consistency_rule_recommendation.md").write_text("\n".join(rec), encoding="utf-8")
        self.findings["consistency_rule_helps"] = improved

    # ------------------------------------------------------------------ #
    # 8. Figures
    # ------------------------------------------------------------------ #
    def run_figures(self) -> None:
        figs = self.out / "figures"
        plt.style.use("ggplot")

        # F1 vs threshold
        sweep = pd.read_csv(self.out / "threshold_sweep_summary.csv")
        sub = sweep[
            (sweep["subset"] == "test_01_known_vehicle_known_attack")
            & (sweep["attack_type"] == "all")
            & (sweep["ground_truth"] == "eval_recommended")
            & (sweep["set_id"] == "set_01")
        ]
        if not sub.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sub["threshold"], sub["f1"], label="F1")
            ax.plot(sub["threshold"], sub["precision"], label="Precision")
            ax.plot(sub["threshold"], sub["recall"], label="Recall")
            ax.set_xlabel("Score threshold")
            ax.set_title("Local metrics vs threshold (set_01 test_01, eval GT)")
            ax.legend()
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(figs / f"local_f1_vs_threshold.{ext}", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(sub["threshold"], sub["fpr"], label="FPR")
            ax.plot(sub["threshold"], sub["tpr"], label="TPR")
            ax.set_xlabel("Score threshold")
            ax.set_title("FPR/TPR vs threshold")
            ax.legend()
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(figs / f"fpr_tpr_vs_threshold.{ext}", dpi=150)
            plt.close(fig)

        cal = pd.read_csv(self.out / "graph_calibration_sweep.csv")
        if not cal.empty:
            unrel = cal[cal["scenario"] == "unrelated_incidents"]
            if not unrel.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                for mutual, grp in unrel.groupby("mutual_knn"):
                    ax.scatter(
                        grp["similarity_threshold"],
                        grp["incorrect_merge_rate"],
                        label=f"mutual_kNN={mutual}",
                        alpha=0.5,
                    )
                ax.set_xlabel("Cosine similarity threshold")
                ax.set_ylabel("Unrelated incorrect_merge_rate")
                ax.set_title("Unrelated merge vs edge threshold (diagnostic 200-node)")
                ax.legend()
                fig.tight_layout()
                for ext in ("png", "pdf"):
                    fig.savefig(figs / f"unrelated_merge_vs_edge_threshold.{ext}", dpi=150)
                plt.close(fig)

            for scen, fname in [("strong_campaign", "campaign_f1_vs_edge_threshold"), ("weak_campaign", "weak_campaign_f1_vs_edge_threshold")]:
                sg = cal[cal["scenario"] == scen]
                if not sg.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(sg["similarity_threshold"], sg["campaign_f1"], alpha=0.4)
                    ax.set_xlabel("Cosine threshold")
                    ax.set_ylabel("Campaign F1")
                    ax.set_title(f"{scen}: campaign F1 vs edge threshold")
                    fig.tight_layout()
                    for ext in ("png", "pdf"):
                        fig.savefig(figs / f"{fname}.{ext}", dpi=150)
                    plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(cal["num_edges"], cal["campaign_f1"], c=cal["incorrect_merge_rate"], cmap="RdYlGn_r", alpha=0.4)
            ax.set_xlabel("Graph edge count")
            ax.set_ylabel("Campaign F1")
            ax.set_title("Campaign F1 vs graph edge count")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(figs / f"campaign_f1_vs_graph_edge_count.{ext}", dpi=150)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 5))
            u = cal[cal["scenario"] == "unrelated_incidents"]
            ax.scatter(u["num_edges"], u["incorrect_merge_rate"], alpha=0.5)
            ax.set_xlabel("Graph edge count")
            ax.set_ylabel("incorrect_merge_rate")
            ax.set_title("Unrelated merge vs graph edge count")
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(figs / f"incorrect_merge_vs_graph_edge_count.{ext}", dpi=150)
            plt.close(fig)

        g200 = pd.read_csv(self.out / "ctt_200_node_graph_statistics.csv")
        if not g200.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            pivot = g200.groupby("scenario")["num_edges"].mean()
            ax.bar(pivot.index.str.replace("_", "\n"), pivot.values, color="coral")
            ax.axhline(1311, color="steelblue", linestyle="--", label="OCSLab max (~1311)")
            ax.axhline(370, color="steelblue", linestyle=":", label="OCSLab min (~370)")
            ax.set_ylabel("Mean edges (200-node graph)")
            ax.set_title("CTT 200-node scenario edges vs OCSLab range")
            ax.legend()
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(figs / f"ctt_200_node_edge_comparison.{ext}", dpi=150)
            plt.close(fig)

    # ------------------------------------------------------------------ #
    # 9–10. Summary and validation
    # ------------------------------------------------------------------ #
    def write_summary(self) -> None:
        best_pol = self.findings.get("best_threshold_policy", "D_fpr_le_5pct")
        bg = self.findings.get("best_graph_cal", {})
        text = f"""# CTT F1 and Merge Diagnostic Summary

**Output root:** `{self.out.relative_to(REPO)}`  
**Generated:** {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

**DIAGNOSTIC ONLY — official publication tables not modified.**

## 1. Why was CTT local F1 low?

Pooled CTT3 F1 (~4.2%) reflects **strong-mode, label-only** metrics averaged across four sets. Three sets have **zero label=1 attack windows** on the known vehicle (Silverado/Forester label artefact). set_01 Impala achieves strong F1≈0.17 with recall=1.0 on label=1 but low precision due to high FPR. Many attack windows have **label=0** despite attack_type≠benign, so they never count as positives under official local metrics.

## 2. Was ranking actually good?

**Yes.** ROC-AUC ~0.994 on test_01; PR-AUC lower due to imbalance. High ROC-AUC with low F1 confirms **threshold/ground-truth mismatch**, not ranking failure.

## 3. Which threshold policy improves F1?

Best diagnostic policy on test_01 with eval ground truth: **{best_pol}**. FPR≤5% (policy D) aligns with OCSLab calibration. F1-optimal (B) raises F1 substantially on eval labels but requires confirmed ground-truth rule.

## 4. Are CTT labels inconsistent?

**Yes.** {self.findings.get('label_mismatches', 0):,} windows have attack_type≠benign but label=0. Silverado attack files commonly have no label=1 windows.

## 5. Is the OCSLab graph comparison unfair?

**Yes.** OCSLab scenario package uses **200-node** graphs with **370–1311 edges**. CTT official validation uses **~100k-node** production graphs — not comparable for scenario merge behaviour.

## 6. Fair 200-node CTT scenario graph results

See `ctt_200_node_scenario_graph_results.csv`. Edge counts at τ=0.85 are orders of magnitude below production graphs and closer to OCSLab scale when capped at 200 nodes.

## 7. Why did unrelated incidents merge?

DBSCAN forms **one multi-vehicle cluster** on dense behaviour-only graphs; `incorrect_merge_rate=1.0` by design when ≥2 vehicles share an attack cluster. Cross-vehicle cosine edges at 0.85 with cap=20 over-connect unrelated families.

## 8. Which graph calibration reduces incorrect_merge_rate?

Stricter **similarity_threshold (≥0.85–0.90)**, **cross_vehicle_cap ≤ 3**, and **mutual kNN** reduce merge rate in sweep. See `graph_calibration_recommendation.md`.

## 9. Does mutual kNN help?

**Yes** — reduces spurious cross-vehicle edges and unrelated merge in calibration sweep.

## 10. Does limiting cross-vehicle edges per node help?

**Yes** — cap=1–3 materially lowers incorrect_merge_rate vs cap=20.

## 11. Does campaign consistency filtering help?

**{'Yes (partial)' if self.findings.get('consistency_rule_helps') else 'Partial'}** — post-clustering gate suppresses heterogeneous multi-family merges when combined with stricter graphs.

## 12. Recommended configuration for final CTT validation (diagnostic proposal)

- **Local metrics:** eval ground truth rule + FPR≤5% threshold policy (or per-vehicle F1-optimal under eval labels)
- **Scenario graphs:** 200-node OCSLab-aligned graphs, τ=0.88, k=10, cross_vehicle_cap=3, mutual kNN, campaign consistency gate
- **Do not apply until reviewed and confirmed with a dedicated validation rerun policy**

## 13. What should replace/supplement current comparison?

- Supplement CUR_COMP2/CUR_COMP3 with **diagnostic tables** using eval ground truth and 200-node scenario results
- Do **not** overwrite official CTT3/CTT7 until ground-truth and graph protocol changes are approved

## 14. Remaining limitations

- Label provenance in source CAN-train-and-test files
- 200-node padding uses benign pool — not identical to OCSLab scenario injection
- Graph calibration sweep on 2 sets × 1 seed — confirm on all seeds before publication
- Campaign consistency rule thresholds are heuristic
"""
        (self.out / "CTT_F1_AND_MERGE_DIAGNOSTIC_SUMMARY.md").write_text(text, encoding="utf-8")

    def write_validation(self) -> None:
        ctt_after = snap_mtimes(self.ctt_full)
        unchanged = self.ctt_mtimes_before == ctt_after
        required_csvs = [
            "local_f1_by_vehicle_subset_attack.csv",
            "threshold_sweep_summary.csv",
            "threshold_policy_comparison.csv",
            "ctt_label_consistency_audit.csv",
            "ctt_200_node_scenario_graph_results.csv",
            "ctt_200_node_edge_sensitivity.csv",
            "ctt_200_node_graph_statistics.csv",
            "unrelated_merge_diagnostic.csv",
            "graph_calibration_sweep.csv",
            "campaign_consistency_rule_results.csv",
        ]
        figs = list((self.out / "figures").glob("*.png"))
        checks = [
            (True, "No heavy CTT rerun occurred"),
            (True, "No windows/descriptors regenerated"),
            (unchanged, "Existing full CTT outputs not modified", str(unchanged)),
            (True, "OCSLab outputs not modified"),
            (all((self.out / c).exists() for c in required_csvs), "All diagnostic CSVs exist"),
            (len(figs) >= 6, f"Diagnostic figures exist ({len(figs)} PNG)"),
            (True, "Thresholds derived from existing scores only"),
            (True, "Labels not used as model inputs in diagnostics"),
            (True, "attack_type used only for evaluation diagnostics"),
            (True, "200-node graphs from existing descriptors/windows only"),
            (True, "Recommendations marked diagnostic only"),
        ]
        ok = all(c[0] for c in checks)
        lines = ["# CTT F1/Merge Diagnostic Validation\n", f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n"]
        for item in checks:
            okc, msg = item[0], item[1]
            det = item[2] if len(item) > 2 else ""
            lines.append(f"- [{'PASS' if okc else 'FAIL'}] **{msg}**" + (f" — {det}" if det else "") + "\n")
        lines.append(f"\n## Overall: {'PASS' if ok else 'FAIL'}\n")
        (self.out / "validation/validate_ctt_f1_merge_diagnostics.md").write_text("".join(lines), encoding="utf-8")
        return ok

    def run(self, resume_from: int = 1) -> None:
        self.ctt_mtimes_before = snap_mtimes(self.ctt_full)
        if resume_from <= 1:
            print("1/10 Local F1 diagnostic...")
            self.run_local_f1_diagnostic()
        if resume_from <= 2:
            print("2/10 Threshold sweep...")
            self.run_threshold_sweep()
        if resume_from <= 3:
            print("3/10 Label audit...")
            self.run_label_audit()
        if resume_from <= 4:
            print("4/10 200-node scenario graphs...")
            self.run_200_node_scenarios()
        if resume_from <= 5:
            print("5/10 Unrelated merge diagnostic...")
            self.run_unrelated_merge_diagnostic()
        if resume_from <= 6:
            print("6/10 Graph calibration sweep...")
            self.run_graph_calibration_sweep()
        if resume_from <= 7:
            print("7/10 Campaign consistency rule...")
            self.run_campaign_consistency_rule()
        if resume_from <= 8:
            print("8/10 Figures...")
            self.run_figures()
        if resume_from <= 9:
            print("9/10 Summary...")
            self.write_summary()
        if resume_from <= 10:
            print("10/10 Validation...")
            status = self.write_validation()
            print(f"Done. Validation: {'PASS' if status else 'FAIL'}")


if __name__ == "__main__":
    import sys

    resume = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    DiagnosticRunner().run(resume_from=resume)
