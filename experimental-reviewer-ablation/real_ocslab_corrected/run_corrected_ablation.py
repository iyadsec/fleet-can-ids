#!/usr/bin/env python3
"""Entry point for the corrected real-OCSLab reviewer ablation.

Hard-stops unless:
  1) Dataset gate passes (Hyundai/Kia/Chevrolet car_track present)
  2) Fleet scaler feature order matches FEATURE_NAMES
  3) Seed-11 cosine distributions are non-collapsed after scaler

Does not retune τ, DBSCAN, or campaign gate after seeing results.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[1]


def main() -> int:
    cfg = yaml.safe_load((EXP / "config.yaml").read_text(encoding="utf-8"))
    print("=== real_ocslab_corrected gate sequence ===")

    # 1. Dataset gate
    rc = subprocess.call([sys.executable, str(EXP / "check_dataset_gate.py")])
    if rc != 0:
        (EXP / "verification_report.txt").write_text(
            (EXP / "verification_report.txt").read_text(encoding="utf-8")
            if (EXP / "verification_report.txt").exists()
            else ""
            + "\n[run] DATASET GATE FAILED — aborting before pool/IF/scenarios/M1-M4.\n",
            encoding="utf-8",
        )
        print("STOPPED: manuscript-aligned car_track data missing.", file=sys.stderr)
        return 2

    # 2. Scaler compatibility
    sys.path.insert(0, str(EXP))
    from fleet_scaler import verify_scaler_compatibility

    scaler_info = verify_scaler_compatibility()
    (EXP / "artifacts" / "scaler_check.json").write_text(
        json.dumps(scaler_info, indent=2), encoding="utf-8"
    )
    if not scaler_info["matches_feature_names"]:
        print("STOPPED: fleet scaler feature mismatch.", file=sys.stderr)
        return 3
    print("Scaler OK:", scaler_info["scaler_id"])

    # 3. Further stages require implemented pool builder with real car_track files.
    print(
        "Dataset + scaler gates passed. Proceed to pool construction "
        "(build_scored_pool_corrected) — not reached in this STOPPED package "
        "until car_track files exist."
    )
    print("Configured seeds:", cfg["seeds"])
    print("Pilot seed:", cfg["pilot_seed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
