"""Stage 1: statistics and integrity validation for consolidated CAN train data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.validate_dataset import (
    BYTE_COLUMNS,
    REQUIRED_COLUMNS,
    count_raw_rows,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 500_000
CRITICAL_NON_NULL = [
    "timestamp",
    "can_id",
    "dlc",
    "label",
    "attack_type",
    "vehicle_model",
    "source_file",
]


@dataclass
class Stage1Check:
    step: str
    name: str
    status: str
    detail: str = ""


@dataclass
class Stage1Report:
    checks: list[Stage1Check] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)

    def add(self, step: str, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        self.checks.append(Stage1Check(step=step, name=name, status=status, detail=detail))
        print(f"[{status}] Step {step}: {name}")
        if detail:
            for line in detail.split("\n"):
                print(f"       {line}")

    @property
    def all_passed(self) -> bool:
        return all(c.status == "PASS" for c in self.checks)


def _iter_chunks(path: Path, *, chunksize: int = CHUNK_SIZE) -> Any:
    return pd.read_csv(path, chunksize=chunksize)


def _scan_consolidated_chunked(path: Path, *, full_duplicates: bool = False) -> dict[str, Any]:
    """Single pass (chunked) over consolidated CSV for counts and validation."""
    n_rows = 0
    columns: list[str] | None = None
    dtypes: dict[str, str] = {}

    vehicle_ctr: Counter = Counter()
    attack_ctr: Counter = Counter()
    label_ctr: Counter = Counter()
    source_ctr: Counter = Counter()
    can_id_ctr: Counter = Counter()

    critical_missing = 0
    invalid_dlc = 0
    invalid_bytes = 0
    dlc_min, dlc_max = np.inf, -np.inf
    byte_min = {c: np.inf for c in BYTE_COLUMNS}
    byte_max = {c: -np.inf for c in BYTE_COLUMNS}

    numeric_stats: dict[str, dict[str, float]] = {}

    source_ts: dict[str, dict[str, float]] = {}
    veh_attack_ctr: Counter = Counter()

    for chunk in _iter_chunks(path):
        if columns is None:
            columns = list(chunk.columns)
            dtypes = {c: str(chunk[c].dtype) for c in chunk.columns}

        n_rows += len(chunk)
        if "vehicle_model" in chunk.columns:
            vehicle_ctr.update(chunk["vehicle_model"].astype(str).tolist())
        if "attack_type" in chunk.columns:
            attack_ctr.update(chunk["attack_type"].astype(str).tolist())
        if "label" in chunk.columns:
            label_ctr.update(chunk["label"].astype(str).tolist())
        if "source_file" in chunk.columns:
            source_ctr.update(chunk["source_file"].astype(str).tolist())
        if "can_id" in chunk.columns:
            can_id_ctr.update(chunk["can_id"].astype(str).tolist())

        crit_cols = [c for c in CRITICAL_NON_NULL if c in chunk.columns]
        if crit_cols:
            critical_missing += int(chunk[crit_cols].isna().sum().sum())

        if "dlc" in chunk.columns:
            dlc = pd.to_numeric(chunk["dlc"], errors="coerce")
            invalid_dlc += int((dlc.isna() | (dlc < 0) | (dlc > 8)).sum())
            if dlc.notna().any():
                dlc_min = min(dlc_min, float(dlc.min()))
                dlc_max = max(dlc_max, float(dlc.max()))

        for col in BYTE_COLUMNS:
            if col not in chunk.columns:
                continue
            vals = pd.to_numeric(chunk[col], errors="coerce")
            present = vals.dropna()
            if len(present):
                byte_min[col] = min(byte_min[col], float(present.min()))
                byte_max[col] = max(byte_max[col], float(present.max()))
            invalid_bytes += int((vals.notna() & ((vals < 0) | (vals > 255))).sum())

        if {"vehicle_model", "attack_type", "label"}.issubset(chunk.columns):
            veh_attack_ctr.update(
                zip(
                    chunk["vehicle_model"].astype(str),
                    chunk["attack_type"].astype(str),
                    chunk["label"].astype(str),
                )
            )

        if "timestamp" in chunk.columns and "source_file" in chunk.columns:
            ts = pd.to_numeric(chunk["timestamp"], errors="coerce")
            for src, grp in chunk.groupby("source_file"):
                src = str(src)
                tsg = ts.loc[grp.index]
                tmin = float(tsg.min()) if tsg.notna().any() else np.nan
                tmax = float(tsg.max()) if tsg.notna().any() else np.nan
                if src not in source_ts:
                    source_ts[src] = {"min": tmin, "max": tmax, "rows": 0}
                source_ts[src]["rows"] += len(grp)
                if not np.isnan(tmin):
                    source_ts[src]["min"] = (
                        tmin if np.isnan(source_ts[src]["min"]) else min(source_ts[src]["min"], tmin)
                    )
                if not np.isnan(tmax):
                    source_ts[src]["max"] = (
                        tmax if np.isnan(source_ts[src]["max"]) else max(source_ts[src]["max"], tmax)
                    )

        num = chunk.select_dtypes(include=[np.number])
        for col in num.columns:
            vals = num[col].dropna()
            if vals.empty:
                continue
            st = numeric_stats.setdefault(
                col, {"count": 0.0, "sum": 0.0, "sumsq": 0.0, "min": np.inf, "max": -np.inf}
            )
            st["count"] += float(len(vals))
            st["sum"] += float(vals.sum())
            st["sumsq"] += float((vals.astype(float) ** 2).sum())
            st["min"] = min(st["min"], float(vals.min()))
            st["max"] = max(st["max"], float(vals.max()))

    if full_duplicates:
        logger.info("Running full duplicate scan (may take several minutes)...")
        dup_extra = _count_duplicates_full(path)
    else:
        dup_extra = _count_duplicates_sample(path, n_rows)

    return {
        "n_rows": n_rows,
        "columns": columns or [],
        "dtypes": dtypes,
        "vehicle_ctr": vehicle_ctr,
        "attack_ctr": attack_ctr,
        "label_ctr": label_ctr,
        "source_ctr": source_ctr,
        "can_id_ctr": can_id_ctr,
        "critical_missing": critical_missing,
        "invalid_dlc": invalid_dlc,
        "invalid_bytes": invalid_bytes,
        "dlc_min": dlc_min if dlc_min != np.inf else np.nan,
        "dlc_max": dlc_max if dlc_max != -np.inf else np.nan,
        "byte_min": {k: (v if v != np.inf else np.nan) for k, v in byte_min.items()},
        "byte_max": {k: (v if v != -np.inf else np.nan) for k, v in byte_max.items()},
        "numeric_stats": numeric_stats,
        "source_ts": source_ts,
        "veh_attack_ctr": veh_attack_ctr,
        "dup_extra": dup_extra,
    }


def _count_duplicates_full(path: Path, *, chunksize: int = CHUNK_SIZE) -> int:
    """Full-file duplicate count via chunked row hashing (memory-safe)."""
    seen: set[str] = set()
    dup_extra = 0
    usecols = [c for c in REQUIRED_COLUMNS if c in pd.read_csv(path, nrows=0).columns]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        keys = chunk.astype(str).agg("|".join, axis=1)
        for key in keys:
            if key in seen:
                dup_extra += 1
            else:
                seen.add(key)
    return dup_extra


def _count_duplicates_sample(path: Path, total_rows: int, *, sample_rows: int = 300_000) -> int:
    """Duplicate estimate from leading rows (full scan too costly on multi-GB CSVs)."""
    n = min(sample_rows, total_rows)
    if n == 0:
        return 0
    sample = pd.read_csv(path, nrows=n)
    return int(sample.duplicated(keep="first").sum())


def _counter_to_df(counter: Counter, col: str) -> pd.DataFrame:
    total = sum(counter.values()) or 1
    rows = [{"record_count": int(v), col: k, "pct_of_total": round(100 * v / total, 6)} for k, v in counter.items()]
    return pd.DataFrame(rows).sort_values("record_count", ascending=False)


def build_vehicle_attack_from_counter(ctr: Counter) -> pd.DataFrame:
    rows = [
        {
            "vehicle_model": k[0],
            "attack_type": k[1],
            "label": k[2],
            "record_count": int(v),
        }
        for k, v in ctr.items()
    ]
    return pd.DataFrame(rows).sort_values(["vehicle_model", "attack_type", "label"])


def build_numeric_summary_from_stats(numeric_stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    if not numeric_stats:
        return pd.DataFrame()
    rows = []
    for col, st in numeric_stats.items():
        n = st["count"]
        if n <= 0:
            continue
        mean = st["sum"] / n
        var = max(st["sumsq"] / n - mean**2, 0.0)
        rows.append(
            {
                "column": col,
                "count": n,
                "mean": round(mean, 6),
                "std": round(float(np.sqrt(var)), 6),
                "min": st["min"] if st["min"] != np.inf else np.nan,
                "max": st["max"] if st["max"] != -np.inf else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_stage1_validation(
    consolidated_path: Path | str,
    raw_roots: list[Path | str],
    *,
    metrics_dir: Path | str,
    top_can_ids: int = 20,
    verify_raw_rows: bool = False,
    full_duplicates: bool = False,
) -> Stage1Report:
    """Run Stage 1 validations; write metric CSV/TXT files."""
    path = Path(consolidated_path)
    out_dir = Path(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = Stage1Report()
    roots = [Path(r) for r in raw_roots if Path(r).exists()]

    # Step 1
    exists = path.exists()
    report.add(
        "1",
        "Consolidated train CSV exists",
        exists,
        str(path.resolve()) if exists else f"Not found: {path}",
    )
    if not exists:
        (out_dir / "consolidated_dataset_summary.txt").write_text(
            "Stage 1 Validation FAILED\nInput file not found.\n", encoding="utf-8"
        )
        return report

    scan = _scan_consolidated_chunked(path, full_duplicates=full_duplicates)
    n_rows = scan["n_rows"]
    columns = scan["columns"]
    n_cols = len(columns)

    # Step 2
    report.add("2", "Row and column counts", n_rows > 0 and n_cols > 0, f"rows={n_rows:,}, columns={n_cols}")

    # Step 3
    dtype_lines = [f"{c}: {scan['dtypes'].get(c, '?')}" for c in columns]
    print("\n--- Column names and dtypes ---")
    for line in dtype_lines:
        print(line)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in columns]
    report.add(
        "3",
        "Column names and data types",
        not missing_cols,
        f"missing required: {missing_cols}" if missing_cols else f"{n_cols} columns documented",
    )

    # Step 4 — group counts (vehicle, attack, label, source_file)
    group_details: list[str] = []
    group_ok = True
    for col, ctr in [
        ("vehicle_model", scan["vehicle_ctr"]),
        ("attack_type", scan["attack_ctr"]),
        ("label", scan["label_ctr"]),
        ("source_file", scan["source_ctr"]),
    ]:
        df_g = _counter_to_df(ctr, col)
        top = df_g.head(6)
        group_details.append(f"{col} ({len(ctr)} groups):")
        group_details.extend(
            f"  {r[col]}: {int(r['record_count']):,}" for _, r in top.iterrows()
        )
        if len(ctr) == 0:
            group_ok = False
    report.add(
        "4",
        "Records per vehicle_model, attack_type, label, source_file",
        group_ok,
        "\n".join(group_details),
    )

    vehicle_attack = build_vehicle_attack_from_counter(scan["veh_attack_ctr"])
    vehicle_attack.to_csv(out_dir / "vehicle_attack_distribution.csv", index=False)
    report.artifacts["vehicle_attack_distribution"] = out_dir / "vehicle_attack_distribution.csv"

    # Step 5
    can_freq = (
        pd.DataFrame(
            [{"can_id": k, "frame_count": int(v)} for k, v in scan["can_id_ctr"].most_common(top_can_ids)]
        )
        .assign(
            rank=lambda d: np.arange(1, len(d) + 1),
            pct_of_frames=lambda d: (d["frame_count"] / n_rows * 100).round(6),
        )
    )
    can_freq.to_csv(out_dir / "can_id_frequency.csv", index=False)
    report.artifacts["can_id_frequency"] = out_dir / "can_id_frequency.csv"
    n_unique = len(scan["can_id_ctr"])
    top_lines = "\n".join(f"{r['can_id']}: {int(r['frame_count']):,}" for _, r in can_freq.iterrows())
    report.add("5", "Unique CAN IDs and top frequencies", n_unique > 0, f"unique={n_unique:,}\n{top_lines}")

    # Step 6
    report.add(
        "6",
        "DLC validation (0-8)",
        scan["invalid_dlc"] == 0,
        f"min={scan['dlc_min']}, max={scan['dlc_max']}, invalid={scan['invalid_dlc']:,}",
    )

    # Step 7
    byte_detail = "\n".join(
        f"{c}: min={scan['byte_min'].get(c)}, max={scan['byte_max'].get(c)}" for c in BYTE_COLUMNS
    )
    report.add(
        "7",
        "Payload byte validation (0-255)",
        scan["invalid_bytes"] == 0,
        f"invalid_cells={scan['invalid_bytes']:,}\n{byte_detail}",
    )

    # Step 8 — missing values and duplicates
    miss_ok = scan["critical_missing"] == 0
    dup_ok = scan["dup_extra"] == 0
    dup_detail = (
        f"duplicate_extra_rows={scan['dup_extra']:,} (full file scan)"
        if full_duplicates
        else f"duplicate_rows_in_leading_sample={scan['dup_extra']:,} "
        f"(first {min(300_000, n_rows):,} rows; pass --full-duplicates for full scan)"
    )
    report.add(
        "8",
        "Missing values and duplicate rows",
        miss_ok and dup_ok,
        f"critical_missing_cells={scan['critical_missing']:,}\n{dup_detail}",
    )

    # Step 9
    source_rows = [
        {
            "source_file": src,
            "row_count": int(meta["rows"]),
            "timestamp_min": meta["min"],
            "timestamp_max": meta["max"],
            "timestamp_span": meta["max"] - meta["min"] if not np.isnan(meta["max"]) else np.nan,
        }
        for src, meta in scan["source_ts"].items()
    ]
    source_stats = pd.DataFrame(source_rows).sort_values("row_count", ascending=False)
    source_stats.to_csv(out_dir / "source_file_row_counts.csv", index=False)
    report.artifacts["source_file_row_counts"] = out_dir / "source_file_row_counts.csv"
    ts_ok = bool(source_stats.shape[0]) and not source_stats["timestamp_min"].isna().all()
    report.add(
        "9",
        "Timestamp range per source_file",
        ts_ok,
        source_stats.head(5).to_string(index=False) if ts_ok else "No source_file timestamps",
    )

    # Step 10
    if verify_raw_rows and roots:
        logger.info("Scanning raw log files for row-count alignment (may take several minutes)...")
        raw_total, _ = count_raw_rows(roots)
        raw_ok = raw_total == n_rows if raw_total > 0 else True
        raw_detail = f"raw_total={raw_total:,}, consolidated={n_rows:,}"
        if raw_total == 0:
            raw_detail += " (no parseable raw files found)"
    else:
        raw_ok = True
        raw_detail = (
            f"consolidated={n_rows:,}; raw scan skipped "
            "(pass --verify-raw to compare against raw logs)"
        )
    report.add("10", "Consolidated vs raw row totals", raw_ok, raw_detail)

    # Step 11
    num_summary = build_numeric_summary_from_stats(scan["numeric_stats"])
    num_summary.to_csv(out_dir / "consolidated_dataset_stats.csv", index=False)
    report.artifacts["consolidated_dataset_stats"] = out_dir / "consolidated_dataset_stats.csv"
    report.add("11", "Numeric column summary statistics", not num_summary.empty, f"columns={len(num_summary)}")

    summary_path = out_dir / "consolidated_dataset_summary.txt"
    lines = [
        "Stage 1 — Consolidated CAN Dataset Validation",
        "=" * 50,
        f"Input: {path.resolve()}",
        f"Rows: {n_rows:,}  Columns: {n_cols}",
        "",
    ]
    for c in report.checks:
        lines.append(f"[{c.status}] Step {c.step}: {c.name}")
        if c.detail:
            for part in c.detail.split("\n")[:12]:
                lines.append(f"    {part}")
    lines.append("=" * 50)
    lines.append("Stage 1 Validation PASSED" if report.all_passed else "Stage 1 Validation FAILED")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report.artifacts["consolidated_dataset_summary"] = summary_path

    return report


def print_stage1_verdict(report: Stage1Report) -> None:
    print()
    if report.all_passed:
        print("Stage 1 Validation PASSED")
    else:
        print("Stage 1 Validation FAILED")
    print()
