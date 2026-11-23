"""Consistent-encoding uplink size measurement on June test windows (separate validation, read-only inputs).

Raw window  : the 100 CAN frames of the window as recorded (OCSLab CSV line without the ground-truth
              R/T flag) and as packed binary (float64 timestamp, uint32 ID, uint8 DLC, DLC data bytes).
Descriptor  : exactly what the fleet layer consumes per window - the 9 GNN input features before
              server-side scaling, a pseudonymous vehicle token and a window sequence number -
              as CSV text (%.6g) and as packed binary (9 x float32, uint32 token, uint32 sequence).
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_campaign_validation as rcv  # noqa: E402,F401  (damaged-file loader + project path)

from src.evaluation.final_gnn_fleet_decision_experiment import GNN_FEATURE_COLUMNS, compute_payload_entropy  # noqa: E402
from src.graph.fleet_similarity_features import build_behavior_view_descriptors  # noqa: E402

RUN = rcv.RUN
PER_FILE = 50
SEED = 2026

wm = pd.read_csv(RUN / "manifests/balanced_window_manifest.csv")
desc = pd.read_csv(RUN / "descriptors/all_descriptors.csv")
test = wm[wm.split == "test"]
sample = test.groupby("source_file", group_keys=False).apply(
    lambda g: g.sample(n=min(PER_FILE, len(g)), random_state=SEED))

d = desc.merge(sample[["window_id", "vehicle_model", "source_file", "start_frame_idx", "end_frame_idx"]],
               on=["window_id", "vehicle_model", "source_file"], how="inner")
d = build_behavior_view_descriptors(d)
d["payload_entropy"] = compute_payload_entropy(d)

rows = []
for src_file, g in d.groupby("source_file"):
    lines = Path(src_file).read_text().splitlines()
    for _, r in g.iterrows():
        frames = lines[int(r.start_frame_idx):int(r.end_frame_idx)]
        raw_text = sum(len((f.rsplit(",", 1)[0] if f.rsplit(",", 1)[-1] in ("R", "T") else f).encode()) + 1
                       for f in frames)
        raw_bin = sum(8 + 4 + 1 + int(f.split(",")[2]) for f in frames)
        vals = [float(r[c]) for c in GNN_FEATURE_COLUMNS]
        desc_text = len((",".join(["V_0161", str(int(r.window_id))] + [f"{v:.6g}" for v in vals]) + "\n").encode())
        desc_bin = len(struct.pack("<9fII", *vals, 161, int(r.window_id)))
        rows.append(dict(source_file=Path(src_file).name, n_frames=len(frames), frame_count=r.frame_count,
                         raw_text_B=raw_text, raw_binary_B=raw_bin, desc_text_B=desc_text, desc_binary_B=desc_bin))

m = pd.DataFrame(rows)
assert (m.n_frames == m.frame_count).all()
summary = pd.DataFrame([{
    "windows_measured": len(m), "source_files": m.source_file.nunique(),
    "raw_text_B": m.raw_text_B.mean(), "desc_text_B": m.desc_text_B.mean(),
    "text_reduction_pct": 100 * (1 - m.desc_text_B.mean() / m.raw_text_B.mean()),
    "raw_binary_B": m.raw_binary_B.mean(), "desc_binary_B": m.desc_binary_B.mean(),
    "binary_reduction_pct": 100 * (1 - m.desc_binary_B.mean() / m.raw_binary_B.mean()),
}]).round(2)
print(summary.T.to_string())
print(m.groupby("source_file")[["raw_text_B", "raw_binary_B", "desc_text_B"]].mean().round(1).to_string())
m.to_csv(HERE / "outputs/descriptor_serialization_per_window.csv", index=False)
summary.to_csv(HERE / "outputs/descriptor_serialization_summary.csv", index=False)
