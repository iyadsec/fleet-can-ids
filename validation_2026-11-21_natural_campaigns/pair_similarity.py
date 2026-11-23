"""9-D scaled cosine similarity by pair type in the natural (B) scenarios (diagnostic only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_campaign_validation as rcv  # noqa: E402  (installs damaged-file loader, imports project code)

CODE = rcv.CODE

from src.evaluation.final_gnn_fleet_decision_experiment import prepare_gnn_fleet_node_matrix  # noqa: E402
from src.experiments.local_descriptor_normalisation import load_scaler_provenance  # noqa: E402

prov = load_scaler_provenance(CODE / "new_experiments/final_end_to_end_publication_run_balanced/scalers/fleet_benign_scaler.json")
acc = {}
for d in sorted((Path(__file__).resolve().parent / "outputs/runs").glob("B_natural__*")):
    s = pd.read_csv(d / "source_windows.csv")
    X, _, _ = prepare_gnn_fleet_node_matrix(s, fleet_scaler_provenance=prov)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    S = Xn @ Xn.T
    iu = np.triu_indices(len(S), 1)
    mal = ((s.ground_truth_malicious == 1) & (s.scenario_role == "coordinated")).to_numpy()
    tok, tr, vm = s.vehicle_token.to_numpy(), s.source_trace.to_numpy(), s.vehicle_model.to_numpy()
    a, b = iu
    sc = "S3" if "strong" in d.name else "S4"
    kinds = {
        "mal-mal same participant": mal[a] & mal[b] & (tok[a] == tok[b]),
        "mal-mal diff participant, same recording": mal[a] & mal[b] & (tok[a] != tok[b]) & (tr[a] == tr[b]),
        "mal-mal diff recording, same platform": mal[a] & mal[b] & (tr[a] != tr[b]) & (vm[a] == vm[b]),
        "mal-mal diff platform": mal[a] & mal[b] & (vm[a] != vm[b]),
        "mal-benign": mal[a] ^ mal[b],
        "benign-benign diff participant": ~mal[a] & ~mal[b] & (tok[a] != tok[b]),
    }
    for k, m in kinds.items():
        if m.any():
            v = S[a, b][m]
            acc.setdefault((sc, k), []).append((v.mean(), (v >= 0.95).mean(), m.sum()))
rows = [dict(scenario=k[0], pair_type=k[1], mean_cosine=np.mean([x[0] for x in v]),
             frac_ge_tau_0_95=np.mean([x[1] for x in v]), runs=len(v)) for k, v in acc.items()]
out = pd.DataFrame(rows).sort_values(["scenario", "pair_type"]).round(3)
print(out.to_string(index=False))
out.to_csv(Path(__file__).resolve().parent / "outputs/natural_pair_similarity.csv", index=False)
