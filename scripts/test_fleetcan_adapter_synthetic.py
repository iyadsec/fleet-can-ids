#!/usr/bin/env python3
"""Synthetic schema fixture: prove adapter maps fleetcan tables → STANDARD_COLUMNS."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset_loader import STANDARD_COLUMNS
from src.data.fleetcan_adapter import (
    session_to_standard,
    validate_dataset_coverage,
)


def _write_session(root: Path, sid: str) -> None:
    raw = root / "raw" / sid
    gt = root / "ground_truth" / sid
    md = root / "metadata" / sid
    raw.mkdir(parents=True)
    gt.mkdir(parents=True)
    md.mkdir(parents=True)

    frames = pd.DataFrame(
        [
            {
                "frame_uid": f"{sid}-F000000001",
                "session_id": sid,
                "scenario_id": "S1",
                "timestamp": 1.0,
                "sim_time_ns": 1_000_000_000,
                "transmission_timestamp": 1.0001,
                "tx_start_ns": 1_000_100_000,
                "tx_end_ns": 1_000_200_000,
                "queue_delay_us": 100.0,
                "frame_bits": 111,
                "capture_time_ns": 1_000_250_000,
                "capture_offset_us": 50.0,
                "vehicle_id": "vehicle_01",
                "channel": "vcan0",
                "can_id": "100",
                "dlc": 8,
                "payload": "0011223344556677",
            },
            {
                "frame_uid": f"{sid}-F000000002",
                "session_id": sid,
                "scenario_id": "S1",
                "timestamp": 1.01,
                "sim_time_ns": 1_010_000_000,
                "transmission_timestamp": 1.0101,
                "tx_start_ns": 1_010_100_000,
                "tx_end_ns": 1_010_200_000,
                "queue_delay_us": 100.0,
                "frame_bits": 111,
                "capture_time_ns": 1_010_250_000,
                "capture_offset_us": 50.0,
                "vehicle_id": "vehicle_02",
                "channel": "vcan1",
                "can_id": "000",
                "dlc": 8,
                "payload": "FFFFFFFFFFFFFFFF",
            },
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "frame_uid": f"{sid}-F000000001",
                "session_id": sid,
                "frame_label": "benign",
                "attack_type": "None",
                "campaign_id": "None",
                "attack_id": "None",
                "frame_source": "benign_generator",
                "message_name": "ENGINE_STATUS",
            },
            {
                "frame_uid": f"{sid}-F000000002",
                "session_id": sid,
                "frame_label": "malicious",
                "attack_type": "flooding",
                "campaign_id": "None",
                "attack_id": "A001",
                "frame_source": "attack_injection",
                "message_name": "",
            },
        ]
    )
    empty = pd.DataFrame()
    frames.to_parquet(raw / "can_frames.parquet", index=False)
    labels.to_parquet(gt / "frame_labels.parquet", index=False)
    empty.to_parquet(gt / "attacks.parquet", index=False)
    empty.to_parquet(gt / "campaigns.parquet", index=False)
    empty.to_parquet(gt / "campaign_membership.parquet", index=False)
    (md / "metadata.json").write_text('{"session_id": "%s", "scenario_id": "S1"}\n' % sid)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sid = "FCS10_N100_S1_flooding_K000_isolated_R1"
        _write_session(root, sid)

        report = validate_dataset_coverage(root, session_ids=[sid])
        assert report["overall"] == "PASS", report

        df = session_to_standard(root, sid, include_eval_columns=True)
        assert list(df.columns)[: len(STANDARD_COLUMNS)] == STANDARD_COLUMNS or set(
            STANDARD_COLUMNS
        ).issubset(df.columns)
        assert len(df) == 2
        assert set(df["vehicle_model"]) == {"vehicle_01", "vehicle_02"}
        assert float(df.loc[df["vehicle_model"] == "vehicle_01", "label"].iloc[0]) == 0.0
        assert float(df.loc[df["vehicle_model"] == "vehicle_02", "label"].iloc[0]) == 1.0
        assert df.loc[df["vehicle_model"] == "vehicle_02", "attack_type"].iloc[0] == "flooding"
        assert int(df.loc[df["vehicle_model"] == "vehicle_01", "byte0"].iloc[0]) == 0x00
        assert int(df.loc[df["vehicle_model"] == "vehicle_01", "byte7"].iloc[0]) == 0x77
        assert int(df.loc[df["vehicle_model"] == "vehicle_02", "byte0"].iloc[0]) == 0xFF
        print("PASS: synthetic fleetcan adapter fixture")
        print(df[STANDARD_COLUMNS].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
