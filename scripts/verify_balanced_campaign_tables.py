#!/usr/bin/env python3
"""Verify canonical balanced OCSLab strong/weak campaign detection tables."""

from __future__ import annotations

import csv
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "experimental-2026-06-23" / "01_primary_ocslab_balanced"
OUT = BASE / "tables"

SOURCES = {
    "strong_detection": BASE / "tables/table_P7_strong_campaign_results.csv",
    "weak_detection": BASE / "tables/table_P8_weak_campaign_results.csv",
    "strong_membership": BASE / "tables/table_P7_strong_campaign_results.csv",
    "weak_membership": BASE / "tables/table_P8_weak_campaign_results.csv",
    "per_seed_campaign": BASE / "results/campaign_metrics.csv",
}

GROUP = {
    "strong": "Strong Coordinated Campaign",
    "weak": "Weak Coordinated Campaign",
}


def f1_harmonic(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_per_seed_f1(per_seed: list[dict], group: str, campaign_size: int) -> float:
    rows = [
        r
        for r in per_seed
        if r["experiment_group"] == group and int(float(r["campaign_size"])) == campaign_size
    ]
    return sum(float(r["campaign_f1"]) for r in rows) / len(rows)


def extract_detection_rows(table_path: Path, group: str, per_seed: list[dict]) -> list[dict]:
    rows_out: list[dict] = []
    for row in load_csv(table_path):
        cs = int(float(row["campaign_size"]))
        detection = float(row["campaign_detection_rate"])
        precision = float(row["campaign_precision"])
        recall = float(row["campaign_recall"])
        reported_f1 = float(row["campaign_f1"])
        recomputed_f1 = f1_harmonic(precision, recall)
        seed_mean_f1 = mean_per_seed_f1(per_seed, group, cs)
        diff_pr = abs(reported_f1 - recomputed_f1)
        diff_seed = abs(reported_f1 - seed_mean_f1)
        # Authoritative: reported F1 equals mean(per-seed F1)
        seed_pass = diff_seed < 1e-9
        pr_pass = diff_pr < 1e-4
        rows_out.append(
            {
                "campaign_size": cs,
                "detection": detection,
                "precision": precision,
                "recall": recall,
                "f1": reported_f1,
                "reported_f1": reported_f1,
                "recomputed_f1_from_mean_pr": recomputed_f1,
                "mean_per_seed_f1": seed_mean_f1,
                "diff_vs_recomputed_pr": diff_pr,
                "diff_vs_per_seed_mean": diff_seed,
                "f1_consistent_with_mean_pr": "PASS" if pr_pass else "FAIL",
                "f1_consistent_with_per_seed_mean": "PASS" if seed_pass else "FAIL",
                "manuscript_include": "YES" if seed_pass else "NO",
                "fail_reason": (
                    ""
                    if pr_pass
                    else "F1 is mean(per-seed F1); precision/recall are separate per-seed means"
                ),
            }
        )
    return sorted(rows_out, key=lambda r: r["campaign_size"])


def write_detection_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "campaign_size",
        "detection",
        "precision",
        "recall",
        "f1",
        "f1_consistent_with_mean_pr",
        "f1_consistent_with_per_seed_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            if r["manuscript_include"] != "YES":
                continue
            w.writerow({k: r[k] for k in fieldnames})


def write_detection_tex(path: Path, caption: str, label: str, rows: list[dict]) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{@{}rcccc@{}}",
        r"\toprule",
        r"Attacked vehicles & Detection & Precision & Recall & F1 \\",
        r"\midrule",
    ]
    for r in rows:
        if r["manuscript_include"] != "YES":
            continue
        cs = r["campaign_size"]
        lines.append(
            f"{cs} & {r['detection']:.3f} & {r['precision']:.3f} & {r['recall']:.3f} & {r['f1']:.3f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{flushleft}",
            r"\footnotesize F1 is the mean of per-seed campaign F1 across 10 seeds; it may differ slightly from the harmonic mean of mean precision and mean recall.",
            r"\end{flushleft}",
            r"\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_block(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print("campaign_size | detection | precision | recall | f1 | reported_F1 | recomputed_F1 | diff | PR_check | seed_check")
    for r in rows:
        print(
            f"{r['campaign_size']:>13} | {r['detection']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
            f"{r['reported_f1']:.6f} | {r['recomputed_f1_from_mean_pr']:.6f} | {r['diff_vs_recomputed_pr']:.6f} | "
            f"{r['f1_consistent_with_mean_pr']} | {r['f1_consistent_with_per_seed_mean']}"
        )
        if r["fail_reason"]:
            print(f"              -> {r['fail_reason']}")


def main() -> int:
    per_seed = load_csv(SOURCES["per_seed_campaign"])
    strong = extract_detection_rows(SOURCES["strong_detection"], GROUP["strong"], per_seed)
    weak = extract_detection_rows(SOURCES["weak_detection"], GROUP["weak"], per_seed)

    OUT.mkdir(parents=True, exist_ok=True)
    write_detection_csv(OUT / "verified_strong_campaign_detection.csv", strong)
    write_detection_csv(OUT / "verified_weak_campaign_detection.csv", weak)
    write_detection_tex(
        OUT / "verified_strong_campaign_detection.tex",
        "Strong coordinated-campaign detection (balanced OCSLab publication run).",
        "tab:verified_strong_campaign_detection",
        strong,
    )
    write_detection_tex(
        OUT / "verified_weak_campaign_detection.tex",
        "Weak coordinated-campaign detection (balanced OCSLab publication run).",
        "tab:verified_weak_campaign_detection",
        weak,
    )

    print("=== Canonical source files ===")
    for name, path in SOURCES.items():
        print(f"  {name}: {path}")

    print_block("STRONG coordinated-campaign detection", strong)
    print_block("WEAK coordinated-campaign detection", weak)

    # Membership summary from same P7/P8 files
    print("\n=== Membership (columns in P7/P8; per-seed backup: membership_metrics.csv on source branch) ===")
    for label, path in [("STRONG", SOURCES["strong_membership"]), ("WEAK", SOURCES["weak_membership"])]:
        print(f"\n{label} membership")
        for row in load_csv(path):
            cs = int(float(row["campaign_size"]))
            print(
                f" cs={cs} mem_precision={float(row['membership_precision']):.3f} "
                f"mem_recall={float(row['membership_recall']):.3f} "
                f"mem_f1={float(row['membership_f1']):.3f} fragmentation={float(row['fragmentation_rate']):.3f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
