"""Validation checks for consolidated CAN frame datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset_loader import (
    _infer_attack_type,
    _infer_vehicle_from_path,
    _parse_line,
    discover_log_files,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS: list[str] = [
    "timestamp",
    "can_id",
    "dlc",
    *[f"byte{i}" for i in range(8)],
    "label",
    "attack_type",
    "vehicle_model",
    "source_file",
]

BYTE_COLUMNS = [f"byte{i}" for i in range(8)]
CRITICAL_NON_NULL_COLUMNS = [
    "timestamp",
    "can_id",
    "dlc",
    "label",
    "attack_type",
    "vehicle_model",
    "source_file",
]


@dataclass
class ValidationCheck:
    """Single validation check outcome."""

    check_id: str
    check_name: str
    status: str  # PASS | FAIL
    metric: str = ""
    value: str = ""
    details: str = ""


@dataclass
class ValidationReport:
    """Collection of validation results."""

    checks: list[ValidationCheck] = field(default_factory=list)

    def add(self, check: ValidationCheck) -> None:
        self.checks.append(check)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "check_id": c.check_id,
                    "check_name": c.check_name,
                    "status": c.status,
                    "metric": c.metric,
                    "value": c.value,
                    "details": c.details,
                }
                for c in self.checks
            ]
        )

    @property
    def all_passed(self) -> bool:
        return all(c.status == "PASS" for c in self.checks)

    def summary_lines(self) -> list[str]:
        lines = ["CAN Dataset Validation Summary", "=" * 40]
        for c in self.checks:
            lines.append(f"[{c.status}] {c.check_id}: {c.check_name}")
            if c.metric:
                lines.append(f"       {c.metric}: {c.value}")
            if c.details:
                for part in c.details.split("\n"):
                    lines.append(f"       {part}")
        lines.append("=" * 40)
        lines.append(
            f"Overall: {'PASS' if self.all_passed else 'FAIL'} "
            f"({sum(1 for c in self.checks if c.status == 'PASS')}/{len(self.checks)} checks passed)"
        )
        return lines


def count_parseable_rows(path: Path) -> int:
    """Count lines in a raw log that parse successfully (same rules as the loader)."""
    count = 0
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if _parse_line(line) is not None:
                count += 1
    return count


def discover_loadable_raw_files(roots: list[Path]) -> list[Path]:
    """Raw files that the loader would include (known vehicle, not anonymous release)."""
    seen: set[Path] = set()
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in discover_log_files(root):
            if path in seen:
                continue
            seen.add(path)
            if _infer_vehicle_from_path(path) is None:
                continue
            if _infer_attack_type(path) == "release":
                continue
            files.append(path)
    return files


def count_raw_rows(roots: list[Path]) -> tuple[int, dict[str, int]]:
    """Return total parseable rows and per-file counts (absolute path keys)."""
    per_file: dict[str, int] = {}
    total = 0
    for path in discover_loadable_raw_files(roots):
        n = count_parseable_rows(path)
        per_file[str(path.resolve())] = n
        total += n
    return total, per_file


def load_clean_dataset(path: Path | str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {csv_path}")
    logger.info("Loading clean dataset from %s", csv_path)
    return pd.read_csv(csv_path)


def _pass_fail(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def validate_row_counts(
    report: ValidationReport,
    raw_roots: list[Path],
    df: pd.DataFrame,
) -> None:
    """Check 1: raw parseable row count vs clean row count."""
    raw_total, per_raw = count_raw_rows(raw_roots)
    clean_total = len(df)

    # Rows attributable to scanned raw roots (by source_file path prefix)
    root_strs = [str(r.resolve()) for r in raw_roots if r.exists()]
    if root_strs:
        mask = df["source_file"].astype(str).apply(
            lambda s: any(s.startswith(root) for root in root_strs)
        )
        clean_from_roots = int(mask.sum())
    else:
        clean_from_roots = 0

    # Compare loader-eligible raw files to full clean set (pipeline output)
    ok_full = raw_total == clean_total
    ok_roots = clean_from_roots == raw_total if raw_total > 0 else ok_full

    details = (
        f"raw_roots={[str(r) for r in raw_roots]}; "
        f"raw_files={len(per_raw)}; "
        f"clean_rows_from_roots={clean_from_roots}"
    )
    if not ok_full and raw_total == 0 and clean_total > 0:
        details += (
            "\nNote: data/raw/ may be empty; clean data was likely built from "
            "external_dataset_dir. Pass --external-root to include it in the count."
        )

    report.add(
        ValidationCheck(
            check_id="01",
            check_name="Raw vs clean row count",
            status=_pass_fail(ok_full or (raw_total > 0 and ok_roots)),
            metric="row_count",
            value=f"raw={raw_total:,}, clean={clean_total:,}",
            details=details,
        )
    )


def validate_required_columns(report: ValidationReport, df: pd.DataFrame) -> None:
    """Check 2: required columns present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report.add(
        ValidationCheck(
            check_id="02",
            check_name="Required columns exist",
            status=_pass_fail(not missing),
            metric="missing_columns",
            value=str(missing) if missing else "none",
            details=f"required={len(REQUIRED_COLUMNS)} columns",
        )
    )


def validate_missing_values(report: ValidationReport, df: pd.DataFrame) -> None:
    """Check 3: missing values per column."""
    missing = df[REQUIRED_COLUMNS].isna().sum()
    critical_missing = int(missing[CRITICAL_NON_NULL_COLUMNS].sum())
    byte_missing = int(missing[BYTE_COLUMNS].sum())
    lines = [f"{col}: {int(missing[col]):,}" for col in REQUIRED_COLUMNS if missing[col] > 0]
    details = "\n".join(lines) if lines else "No missing values in required columns."

    report.add(
        ValidationCheck(
            check_id="03",
            check_name="Missing values per column",
            status=_pass_fail(critical_missing == 0),
            metric="critical_missing_cells",
            value=str(critical_missing),
            details=f"{details}\nbyte_missing_cells={byte_missing:,} (may be valid beyond DLC)",
        )
    )


def validate_group_counts(
    report: ValidationReport,
    df: pd.DataFrame,
    column: str,
    check_id: str,
    check_name: str,
) -> None:
    """Checks 4–6: record counts per grouping column."""
    if column not in df.columns:
        report.add(
            ValidationCheck(
                check_id=check_id,
                check_name=check_name,
                status="FAIL",
                metric=column,
                value="column missing",
                details="",
            )
        )
        return

    counts = df[column].value_counts(dropna=False)
    lines = [f"{idx}: {int(cnt):,}" for idx, cnt in counts.items()]
    report.add(
        ValidationCheck(
            check_id=check_id,
            check_name=check_name,
            status=_pass_fail(len(counts) > 0 and len(df) > 0),
            metric=f"n_{column}",
            value=str(len(counts)),
            details="\n".join(lines[:30]) + ("\n..." if len(lines) > 30 else ""),
        )
    )


def validate_random_sample(
    report: ValidationReport,
    df: pd.DataFrame,
    n: int = 5,
    seed: int = 42,
) -> None:
    """Check 7: print random sample rows."""
    if df.empty:
        sample_text = "Dataset is empty."
        ok = False
    else:
        sample = df.sample(n=min(n, len(df)), random_state=seed)
        sample_text = sample.to_string(index=False)
        ok = True

    logger.info("Random sample (%d rows):\n%s", min(n, len(df)), sample_text)
    report.add(
        ValidationCheck(
            check_id="07",
            check_name="Random row sample",
            status=_pass_fail(ok),
            metric="sample_size",
            value=str(min(n, len(df))),
            details=sample_text[:2000] + ("..." if len(sample_text) > 2000 else ""),
        )
    )


def validate_dlc_range(report: ValidationReport, df: pd.DataFrame) -> None:
    """Check 8: DLC must be in [0, 8]."""
    dlc = pd.to_numeric(df["dlc"], errors="coerce")
    invalid = dlc.isna() | (dlc < 0) | (dlc > 8)
    n_invalid = int(invalid.sum())
    report.add(
        ValidationCheck(
            check_id="08",
            check_name="DLC in range 0-8",
            status=_pass_fail(n_invalid == 0),
            metric="invalid_dlc_rows",
            value=str(n_invalid),
            details=f"min={dlc.min()}, max={dlc.max()}" if n_invalid else "All DLC valid.",
        )
    )


def validate_byte_range(report: ValidationReport, df: pd.DataFrame) -> None:
    """Check 9: byte values in [0, 255] where present."""
    n_invalid = 0
    per_col: list[str] = []
    for col in BYTE_COLUMNS:
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = vals.notna() & ((vals < 0) | (vals > 255))
        c = int(bad.sum())
        if c:
            per_col.append(f"{col}: {c:,}")
        n_invalid += c

    report.add(
        ValidationCheck(
            check_id="09",
            check_name="Byte values in range 0-255",
            status=_pass_fail(n_invalid == 0),
            metric="invalid_byte_cells",
            value=str(n_invalid),
            details="\n".join(per_col) if per_col else "All present byte values valid.",
        )
    )


def validate_duplicates(report: ValidationReport, df: pd.DataFrame) -> None:
    """Check 10: duplicate rows."""
    n_dup = int(df.duplicated(keep=False).sum())
    n_unique_dup_groups = int(df.duplicated(keep="first").sum())
    report.add(
        ValidationCheck(
            check_id="10",
            check_name="Duplicate rows",
            status=_pass_fail(n_unique_dup_groups == 0),
            metric="duplicate_rows",
            value=f"{n_unique_dup_groups:,} extra ({n_dup:,} including first copies)",
            details="Exact duplicates across all columns.",
        )
    )


def validate_source_file_alignment(
    report: ValidationReport,
    raw_roots: list[Path],
    df: pd.DataFrame,
) -> None:
    """Optional alignment: per-source_file counts raw vs clean."""
    _, per_raw = count_raw_rows(raw_roots)
    if not per_raw:
        return

    clean_counts = df.groupby("source_file").size().to_dict()
    mismatches: list[str] = []
    for src, raw_n in per_raw.items():
        clean_n = clean_counts.get(src, 0)
        if raw_n != clean_n:
            mismatches.append(f"{Path(src).name}: raw={raw_n:,} clean={clean_n:,}")

    if mismatches:
        report.add(
            ValidationCheck(
                check_id="06b",
                check_name="Per-file raw vs clean alignment",
                status="FAIL",
                metric="mismatched_files",
                value=str(len(mismatches)),
                details="\n".join(mismatches[:20]),
            )
        )


def run_validation(
    clean_path: Path | str,
    raw_roots: list[Path | str],
    *,
    sample_size: int = 5,
    seed: int = 42,
) -> ValidationReport:
    """Run all validation checks and return a report."""
    roots = [Path(r) for r in raw_roots]
    df = load_clean_dataset(clean_path)
    report = ValidationReport()

    validate_row_counts(report, roots, df)
    validate_required_columns(report, df)
    validate_missing_values(report, df)
    validate_group_counts(report, df, "vehicle_model", "04", "Records per vehicle_model")
    validate_group_counts(report, df, "attack_type", "05", "Records per attack_type")
    validate_group_counts(report, df, "source_file", "06", "Records per source_file")
    validate_source_file_alignment(report, roots, df)
    validate_random_sample(report, df, n=sample_size, seed=seed)
    validate_dlc_range(report, df)
    validate_byte_range(report, df)
    validate_duplicates(report, df)

    return report


def save_validation_report(
    report: ValidationReport,
    csv_path: Path | str,
    summary_path: Path | str,
) -> tuple[Path, Path]:
    """Write CSV report and text summary."""
    csv_out = Path(csv_path)
    txt_out = Path(summary_path)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    txt_out.parent.mkdir(parents=True, exist_ok=True)

    report.to_dataframe().to_csv(csv_out, index=False)
    txt_out.write_text("\n".join(report.summary_lines()) + "\n", encoding="utf-8")
    logger.info("Wrote validation report to %s and %s", csv_out, txt_out)
    return csv_out, txt_out


def print_validation_results(report: ValidationReport) -> None:
    """Print PASS/FAIL for each check to stdout."""
    print("\n=== Validation Results ===")
    for c in report.checks:
        print(f"[{c.status}] {c.check_id} — {c.check_name}")
        if c.metric:
            print(f"         {c.metric}: {c.value}")
    print(f"\nOverall: {'PASS' if report.all_passed else 'FAIL'}\n")
