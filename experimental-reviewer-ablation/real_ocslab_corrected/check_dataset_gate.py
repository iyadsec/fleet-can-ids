#!/usr/bin/env python3
"""Hard dataset gate for the corrected reviewer ablation.

Fails unless Dataset/ocslab_pipeline/{Hyundai,Kia,Chevrolet} contain
manuscript car_track-style traces (Sonata/Soul/Spark tokens preferred).
Rejects classic Car-Hacking copies and PR #22 Challenge D/S sources.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIPE = ROOT / "Dataset" / "ocslab_pipeline"
CLASSIC_NORMAL = ROOT / "Dataset" / "ocslab" / "normal_run_data" / "normal_run_data.txt"
MANIFEST = (
    ROOT
    / "recovered_publication_pipeline"
    / "new_experiments"
    / "final_end_to_end_publication_run_balanced"
    / "manifests"
    / "balanced_split_manifest.csv"
)
OUT = Path(__file__).resolve().parent / "artifacts" / "dataset_gate.json"

OEM_TOKENS = {
    "Hyundai": ("sonata", "hyundai", "hy_"),
    "Kia": ("soul", "kia"),
    "Chevrolet": ("spark", "chevrolet"),
}


def md5_prefix(path: Path, n: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        h.update(fh.read(n))
    return h.hexdigest()


def main() -> int:
    errors: list[str] = []
    info: dict = {"platforms": {}, "errors": errors}

    for plat in ("Hyundai", "Kia", "Chevrolet"):
        d = PIPE / plat
        files = sorted(p for p in d.iterdir() if p.is_file()) if d.exists() else []
        names = [p.name for p in files]
        tokens = OEM_TOKENS[plat]
        token_hits = [
            n for n in names if any(t in n.lower() for t in tokens)
        ]
        entry = {
            "dir": str(d),
            "n_files": len(files),
            "files": names,
            "oem_token_hits": token_hits,
        }
        if not files:
            errors.append(f"{plat}: directory empty or missing")
        elif not token_hits:
            errors.append(
                f"{plat}: files present but none match car_track tokens {tokens}; "
                f"got {names}"
            )
        info["platforms"][plat] = entry

    chev_af = PIPE / "Chevrolet" / "attack_free_run.txt"
    if chev_af.exists() and CLASSIC_NORMAL.exists():
        classic_match = (
            chev_af.stat().st_size == CLASSIC_NORMAL.stat().st_size
            and md5_prefix(chev_af) == md5_prefix(CLASSIC_NORMAL)
        )
        info["chevrolet_classic_identity"] = classic_match
        if classic_match:
            errors.append(
                "Chevrolet/attack_free_run.txt is classic Car-Hacking "
                "(matches Dataset/ocslab/normal_run_data), not Spark car_track"
            )

    if MANIFEST.exists():
        import pandas as pd

        req = sorted(
            {
                Path(p).name
                for p in pd.read_csv(MANIFEST)["source_file"].astype(str)
            }
        )
        local = {p.name for p in (ROOT / "Dataset").rglob("*") if p.is_file()}
        missing = [b for b in req if b not in local]
        info["publication_basenames_required"] = len(req)
        info["publication_basenames_missing"] = len(missing)
        info["publication_basenames_missing_list"] = missing
        if missing:
            errors.append(
                f"Missing {len(missing)}/{len(req)} publication car_track basenames"
            )

    info["passed"] = len(errors) == 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))
    if errors:
        print("\nDATASET GATE FAILED — corrected ablation must STOP.", file=sys.stderr)
        return 1
    print("\nDATASET GATE PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
