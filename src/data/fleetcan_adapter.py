"""Adapt fleetcan-sim-scale-1.0 sessions into the pipeline STANDARD_COLUMNS schema.

Observable frames live under ``data/raw/<session>/can_frames.{parquet,csv}``.
Labels and attack types are evaluation-only and live under
``data/ground_truth/<session>/frame_labels.{parquet,csv}``; they are joined by
``frame_uid`` and must not be treated as detector features.

Detector-safe rule: ``scenario_id`` may appear in raw frames for provenance but
must not be used as a model feature (explicit allow-list elsewhere).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.data.dataset_loader import BYTE_COLUMNS, STANDARD_COLUMNS, _standardize_frame
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Columns required on the raw observation table (schema 1.0).
RAW_REQUIRED_COLUMNS: list[str] = [
    "frame_uid",
    "session_id",
    "scenario_id",
    "timestamp",
    "vehicle_id",
    "can_id",
    "dlc",
    "payload",
]

# Columns required on the ground-truth label table.
LABEL_REQUIRED_COLUMNS: list[str] = [
    "frame_uid",
    "frame_label",
    "attack_type",
]

# Session artefacts required for pipeline load + evaluation join.
SESSION_REQUIRED_ARTEFACTS: dict[str, tuple[str, ...]] = {
    "raw": ("can_frames",),
    "ground_truth": ("frame_labels", "attacks", "campaigns", "campaign_membership"),
    "metadata": ("metadata.json",),
}

ATTACK_TYPE_CANONICAL = {
    "none": "attack_free",
    "None": "attack_free",
    "": "attack_free",
    "attack_free": "attack_free",
    "benign": "attack_free",
    "flooding": "flooding",
    "fuzzy": "fuzzy",
    "malfunction": "malfunction",
    "replay": "replay",
}


def resolve_table(base: Path) -> Path:
    """Resolve ``base`` to ``.parquet`` or ``.csv`` (exactly one must exist)."""
    parquet = Path(str(base) + ".parquet")
    csv = Path(str(base) + ".csv")
    if parquet.exists() and csv.exists():
        raise FileNotFoundError(f"Ambiguous table (both parquet and csv): {base}")
    if parquet.exists():
        return parquet
    if csv.exists():
        return csv
    raise FileNotFoundError(f"Table not found (tried .parquet/.csv): {base}")


def read_table(base: Path) -> pd.DataFrame:
    """Read a fleet-can-dataset table addressed without suffix."""
    path = resolve_table(base)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def dataset_root_from_release(release_root: Path | str) -> Path:
    """
    Map a restored release directory to the data root that contains
    ``raw/``, ``ground_truth/``, ``metadata/``.
    """
    root = Path(release_root)
    candidates = [
        root,
        root / "data",
        root / "fleetcan-sim-scale-1.0" / "data",
        root / "fleetcan-sim-scale-1.0" / "data" / "final" / "fleetcan-sim-scale-1.0",
    ]
    for cand in candidates:
        if (cand / "raw").is_dir() and (cand / "ground_truth").is_dir():
            return cand.resolve()
    raise FileNotFoundError(
        f"Could not locate fleetcan data root under {release_root}. "
        "Expected raw/ and ground_truth/ after restore."
    )


def list_sessions(data_root: Path | str) -> list[str]:
    """Return sorted session IDs present under ``data_root/raw``."""
    raw = Path(data_root) / "raw"
    if not raw.is_dir():
        return []
    return sorted(p.name for p in raw.iterdir() if p.is_dir() and p.name.startswith("FCS"))


def session_paths(data_root: Path | str, session_id: str) -> dict[str, Path]:
    """Return artefact base paths for one session."""
    root = Path(data_root)
    sid = session_id
    return {
        "frames": root / "raw" / sid / "can_frames",
        "labels": root / "ground_truth" / sid / "frame_labels",
        "attacks": root / "ground_truth" / sid / "attacks",
        "campaigns": root / "ground_truth" / sid / "campaigns",
        "membership": root / "ground_truth" / sid / "campaign_membership",
        "metadata": root / "metadata" / sid / "metadata.json",
        "validation": root / "validation" / sid,
    }


def check_session_artefacts(data_root: Path | str, session_id: str) -> list[str]:
    """Return list of missing required artefacts for *session_id* (empty = OK)."""
    paths = session_paths(data_root, session_id)
    missing: list[str] = []
    for key in ("frames", "labels", "attacks", "campaigns", "membership"):
        try:
            resolve_table(paths[key])
        except FileNotFoundError:
            missing.append(str(paths[key]))
    if not paths["metadata"].is_file():
        missing.append(str(paths["metadata"]))
    return missing


def _parse_continuous_hex_payload(payload: Any, dlc: int | None) -> dict[str, float]:
    """Split continuous uppercase hex payload (2×DLC chars) into byte0..byte7."""
    out: dict[str, float] = {col: np.nan for col in BYTE_COLUMNS}
    if payload is None or (isinstance(payload, float) and np.isnan(payload)):
        return out
    text = str(payload).strip().upper().replace(" ", "")
    if not text:
        return out

    # Space-separated hex tokens are also accepted for CSV debugging dumps.
    if " " in str(payload).strip():
        tokens = str(payload).strip().split()
        limit = min(len(tokens), 8)
        if dlc is not None and not (isinstance(dlc, float) and np.isnan(dlc)):
            try:
                limit = min(limit, int(dlc))
            except (TypeError, ValueError):
                pass
        for idx, token in enumerate(tokens[:limit]):
            try:
                out[f"byte{idx}"] = float(int(token, 16))
            except ValueError:
                out[f"byte{idx}"] = np.nan
        return out

    limit = min(len(text) // 2, 8)
    if dlc is not None and not (isinstance(dlc, float) and np.isnan(dlc)):
        try:
            limit = min(limit, int(dlc))
        except (TypeError, ValueError):
            pass
    for idx in range(limit):
        token = text[2 * idx : 2 * idx + 2]
        try:
            out[f"byte{idx}"] = float(int(token, 16))
        except ValueError:
            out[f"byte{idx}"] = np.nan
    return out


def _payload_series_to_bytes(payload: pd.Series, dlc: pd.Series) -> pd.DataFrame:
    """Vectorized continuous-hex payload → byte0..byte7."""
    text = payload.astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    dlc_int = pd.to_numeric(dlc, errors="coerce").fillna(0).astype(int).clip(0, 8)
    cols: dict[str, np.ndarray] = {}
    for i in range(8):
        token = text.str.slice(2 * i, 2 * i + 2)
        valid = (dlc_int > i) & token.str.fullmatch(r"[0-9A-F]{2}", na=False)
        parsed = np.full(len(text), np.nan, dtype=float)
        if bool(valid.any()):
            parsed[valid.to_numpy()] = token[valid].map(lambda s: int(s, 16)).to_numpy(dtype=float)
        cols[f"byte{i}"] = parsed
    return pd.DataFrame(cols)


def _canonicalize_attack_type(value: Any) -> str:
    text = "none" if value is None else str(value).strip()
    if text.lower() in {"none", "nan", "null", ""}:
        text = "none"
    return ATTACK_TYPE_CANONICAL.get(text, ATTACK_TYPE_CANONICAL.get(text.lower(), text.lower()))


def _label_from_frame_label(frame_label: Any, attack_type: str) -> float:
    """Map ground-truth frame_label / attack_type to pipeline label {0,1}."""
    fl = str(frame_label).strip().lower() if frame_label is not None else ""
    if fl in {"benign", "0", "normal"}:
        return 0.0
    if fl in {"malicious", "attack", "1"}:
        return 1.0
    if attack_type == "attack_free":
        return 0.0
    if attack_type in {"flooding", "fuzzy", "malfunction", "replay"}:
        return 1.0
    return np.nan


def validate_raw_schema(frames: pd.DataFrame) -> list[str]:
    """Return schema issues for a raw can_frames table (empty = OK)."""
    issues: list[str] = []
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in frames.columns]
    if missing:
        issues.append(f"missing raw columns: {missing}")
        return issues
    if frames.empty:
        issues.append("raw can_frames is empty")
        return issues
    dlc = pd.to_numeric(frames["dlc"], errors="coerce")
    if int(((dlc < 0) | (dlc > 8) | dlc.isna()).sum()) > 0:
        issues.append("dlc outside [0, 8] or non-numeric")
    payload = frames["payload"].astype(str).str.upper().str.replace(" ", "", regex=False)
    bad_hex = ~payload.str.fullmatch(r"[0-9A-F]*", na=False)
    if int(bad_hex.sum()) > 0:
        issues.append(f"non-hex payload rows: {int(bad_hex.sum()):,}")
    # Classical CAN: payload length == 2 * dlc
    expected_len = (2 * dlc.fillna(0).astype(int)).to_numpy()
    actual_len = payload.str.len().to_numpy()
    mismatch = int((actual_len != expected_len).sum())
    if mismatch:
        issues.append(f"payload length != 2*dlc for {mismatch:,} rows")
    return issues


def validate_label_schema(labels: pd.DataFrame) -> list[str]:
    """Return schema issues for a frame_labels table (empty = OK)."""
    issues: list[str] = []
    missing = [c for c in LABEL_REQUIRED_COLUMNS if c not in labels.columns]
    if missing:
        issues.append(f"missing label columns: {missing}")
        return issues
    if labels.empty:
        issues.append("frame_labels is empty")
    return issues


def session_to_standard(
    data_root: Path | str,
    session_id: str,
    *,
    include_eval_columns: bool = False,
) -> pd.DataFrame:
    """
    Load one session and return a DataFrame with STANDARD_COLUMNS.

    ``vehicle_model`` is set from ``vehicle_id`` (e.g. ``vehicle_01``).
    ``source_file`` records the resolved can_frames path.
    """
    paths = session_paths(data_root, session_id)
    missing = check_session_artefacts(data_root, session_id)
    if missing:
        raise FileNotFoundError(
            f"Session {session_id} missing artefacts:\n  " + "\n  ".join(missing)
        )

    frames = read_table(paths["frames"])
    labels = read_table(paths["labels"])
    raw_issues = validate_raw_schema(frames)
    label_issues = validate_label_schema(labels)
    if raw_issues or label_issues:
        raise ValueError(
            f"Session {session_id} schema validation failed:\n"
            + "\n".join(f"  - {i}" for i in raw_issues + label_issues)
        )

    label_cols = ["frame_uid", "frame_label", "attack_type"]
    optional_eval = [
        c
        for c in ("campaign_id", "attack_id", "frame_source", "message_name")
        if c in labels.columns
    ]
    keep_label = label_cols + (optional_eval if include_eval_columns else [])
    merged = frames.merge(labels[keep_label], on="frame_uid", how="left", validate="one_to_one")

    if merged["frame_label"].isna().any():
        n_miss = int(merged["frame_label"].isna().sum())
        raise ValueError(
            f"Session {session_id}: {n_miss:,} frames lack ground-truth labels "
            "(frame_uid join failed)."
        )

    # Per-(session, vehicle) source_file so the existing window generator
    # (group_col=source_file) matches OCSLab's one-trace-per-file layout.
    frame_table = str(resolve_table(paths["frames"]))
    vehicle_ids = merged["vehicle_id"].astype(str)
    source = vehicle_ids.map(lambda v: f"{frame_table}#{v}")
    dlc = pd.to_numeric(merged["dlc"], errors="coerce")
    attack_type = merged["attack_type"].map(_canonicalize_attack_type)
    frame_label = merged["frame_label"].astype(str).str.strip().str.lower()
    label = np.where(
        frame_label.isin(["benign", "0", "normal"]),
        0.0,
        np.where(
            frame_label.isin(["malicious", "attack", "1"]),
            1.0,
            np.where(
                attack_type.eq("attack_free"),
                0.0,
                np.where(
                    attack_type.isin(["flooding", "fuzzy", "malfunction", "replay"]),
                    1.0,
                    np.nan,
                ),
            ),
        ),
    )
    bytes_df = _payload_series_to_bytes(merged["payload"], dlc)

    out = pd.DataFrame(
        {
            "timestamp": pd.to_numeric(merged["timestamp"], errors="coerce"),
            "can_id": merged["can_id"]
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace("0X", "", regex=False),
            "dlc": dlc,
            "label": label,
            "attack_type": attack_type.astype("string"),
            "vehicle_model": vehicle_ids,
            "source_file": source,
        }
    )
    for col in BYTE_COLUMNS:
        out[col] = bytes_df[col] if col in bytes_df.columns else np.nan

    standardized = _standardize_frame(out)
    if include_eval_columns:
        standardized["session_id"] = session_id
        standardized["frame_uid"] = merged["frame_uid"].astype(str).values
        if "scenario_id" in merged.columns:
            standardized["scenario_id"] = merged["scenario_id"].astype(str).values
        for c in optional_eval:
            standardized[c] = merged[c].astype(str).values
    return standardized


def load_sessions(
    data_root: Path | str,
    session_ids: Sequence[str] | None = None,
    *,
    include_eval_columns: bool = False,
) -> pd.DataFrame:
    """Load and concatenate selected sessions into STANDARD_COLUMNS."""
    root = Path(data_root)
    ids = list(session_ids) if session_ids is not None else list_sessions(root)
    if not ids:
        logger.warning("No sessions selected under %s", root)
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    frames: list[pd.DataFrame] = []
    for sid in ids:
        logger.info("Adapting session %s", sid)
        frames.append(
            session_to_standard(root, sid, include_eval_columns=include_eval_columns)
        )

    merged = pd.concat(frames, ignore_index=True)
    # Temporal order within each vehicle trace (do NOT sort by attack_type —
    # that would break interleaved benign/attack timelines).
    sort_cols = [
        c
        for c in ("session_id", "source_file", "vehicle_model", "timestamp")
        if c in merged.columns
    ]
    if not sort_cols:
        sort_cols = ["vehicle_model", "timestamp"]
    return merged.sort_values(sort_cols).reset_index(drop=True)


def load_manifest_sessions(
    manifest_path: Path | str,
    *,
    n: int | None = None,
    scenarios: Iterable[str] | None = None,
    repetition: int | None = 1,
    attacks: Iterable[str] | None = None,
) -> list[str]:
    """Select session IDs from the frozen manifest.jsonl with simple filters."""
    path = Path(manifest_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    scenario_set = set(scenarios) if scenarios is not None else None
    attack_set = set(attacks) if attacks is not None else None
    selected: list[str] = []
    for row in rows:
        if n is not None and int(row.get("N", -1)) != n:
            continue
        if repetition is not None and int(row.get("repetition", -1)) != repetition:
            continue
        if scenario_set is not None and row.get("scenario") not in scenario_set:
            continue
        if attack_set is not None and row.get("attack") not in attack_set:
            continue
        selected.append(str(row["session_id"]))
    return selected


def default_pilot_sessions() -> list[str]:
    """Fixed N=100 R1 pilot covering S0–S3 (flooding where applicable)."""
    return [
        "FCS10_N100_S0_none_K000_none_R1",
        "FCS10_N100_S1_flooding_K000_isolated_R1",
        "FCS10_N100_S2_flooding_K000_independent_I05near_R1",
        "FCS10_N100_S3_flooding_K005_medium_R1",
    ]


def validate_dataset_coverage(
    data_root: Path | str,
    *,
    session_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    Structural validation of restored fleetcan data against pipeline needs.

    Returns a report dict with overall status and per-session results.
    Does not invent metrics — only checks presence/schema joinability.
    """
    root = Path(data_root)
    ids = list(session_ids) if session_ids is not None else list_sessions(root)
    report: dict[str, Any] = {
        "data_root": str(root),
        "n_sessions_checked": len(ids),
        "sessions": {},
        "required_pipeline_columns": list(STANDARD_COLUMNS),
        "raw_required_columns": list(RAW_REQUIRED_COLUMNS),
        "label_required_columns": list(LABEL_REQUIRED_COLUMNS),
        "overall": "PASS",
    }
    if not ids:
        report["overall"] = "FAIL"
        report["error"] = "No sessions found"
        return report

    for sid in ids:
        entry: dict[str, Any] = {"status": "PASS", "issues": []}
        missing = check_session_artefacts(root, sid)
        if missing:
            entry["status"] = "FAIL"
            entry["issues"].extend(f"missing:{m}" for m in missing)
            report["sessions"][sid] = entry
            report["overall"] = "FAIL"
            continue
        try:
            frames = read_table(session_paths(root, sid)["frames"])
            labels = read_table(session_paths(root, sid)["labels"])
            entry["issues"].extend(validate_raw_schema(frames))
            entry["issues"].extend(validate_label_schema(labels))
            entry["n_frames"] = int(len(frames))
            entry["n_labels"] = int(len(labels))
            entry["n_vehicles"] = int(frames["vehicle_id"].nunique()) if "vehicle_id" in frames else 0
            if len(frames) != len(labels):
                entry["issues"].append(
                    f"frame/label count mismatch: frames={len(frames)} labels={len(labels)}"
                )
            # Join key uniqueness
            if frames["frame_uid"].duplicated().any():
                entry["issues"].append("duplicate frame_uid in raw")
            if labels["frame_uid"].duplicated().any():
                entry["issues"].append("duplicate frame_uid in labels")
            if entry["issues"]:
                entry["status"] = "FAIL"
                report["overall"] = "FAIL"
            else:
                # Mapping contract: raw+labels can supply every STANDARD_COLUMNS field.
                mapping_ok = (
                    {"timestamp", "can_id", "dlc", "payload", "vehicle_id"}
                    <= set(frames.columns)
                    and {"frame_label", "attack_type"} <= set(labels.columns)
                )
                if not mapping_ok:
                    entry["status"] = "FAIL"
                    entry["issues"].append(
                        "cannot map raw+labels onto STANDARD_COLUMNS "
                        f"{STANDARD_COLUMNS}"
                    )
                    report["overall"] = "FAIL"
        except Exception as exc:  # noqa: BLE001 — surface any schema/IO failure
            entry["status"] = "FAIL"
            entry["issues"].append(str(exc))
            report["overall"] = "FAIL"
        report["sessions"][sid] = entry
    return report
