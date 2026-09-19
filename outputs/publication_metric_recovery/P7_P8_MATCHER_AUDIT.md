# P7_P8_MATCHER_AUDIT.md

## Verdict on matcher recovery: **UNRECOVERABLE as executed code**

The function that assigned predicted campaign clusters to GT campaigns for P7/P8 was never committed (`extract_run_metrics` / refinement pipeline). Ablation Jaccard≥0.5 is **not** connected to the balanced publication runner.

---

## 1. What the saved rows imply (artifact identities)

On all 60 strong+weak `campaign_metrics.csv` rows:

| Quantity | Identity that holds 100% |
|----------|---------------------------|
| TP | `min(n_predicted_campaign_clusters, n_true_campaign_clusters)` |
| FP (`false_campaign_cluster_count`) | `max(n_pred − n_true, 0)` |
| FN (`missed_campaign_count`) | `max(n_true − n_pred, 0)` |
| `campaign_precision` | `TP / n_pred` if `n_pred>0` else `0` |
| `campaign_recall` | `TP / n_true` |
| `campaign_detection_rate` | equals `campaign_recall` |
| `completeness` | equals `campaign_recall` |

**Grade:** `VERIFIED_FROM_PUBLICATION_ARTIFACT`.

With `n_true = 1` on every strong/weak row, this reduces to:

- Detected (`recall=1`) iff `n_pred ≥ 1`
- `precision = 1/n_pred` when detected (or `1` if `n_pred=1`)
- Extra predicted campaigns beyond 1 are pure FP count

### Does this require Jaccard / IoU / Hungarian?

**Not for explaining the campaign-level TP/FP/FN columns.**  
Those columns are fully explained by **counts of qualifying predicted campaigns vs counts of GT campaigns**, without an overlap threshold appearing in the aggregates.

**Caveat:** every row with `TP>0` also has `attacked_vehicles_correctly_included > 0` (no counterexample of a “matched” campaign with zero GT-vehicle membership). That is **consistent with** an overlap filter existing *before* a cluster is counted as a predicted campaign, or with the gate never emitting empty-overlap campaigns in this dataset. It does **not** prove Jaccard≥0.5 (or any threshold).

**Grade for “count-based campaign scoring”:** `VERIFIED_FROM_PUBLICATION_ARTIFACT` (identities).  
**Grade for “no overlap matcher was used”:** `INFERRED` (cannot distinguish from always-overlapping outputs).  
**Grade for “Jaccard≥0.5 was the P7/P8 matcher”:** `UNRECOVERABLE` / contradicted as proven link — ablation-only documentation.

---

## 2. Candidate matchers examined

| Candidate | Location | Connected to balanced P7/P8? | Grade |
|-----------|----------|------------------------------|-------|
| Greedy Jaccard≥0.5 `match_campaigns` | reviewer ablation docs / unrecovered ablation tree | **No** — AUTHORITATIVE audit explicitly marks not proven | Not P7/P8 |
| Count-based `min(n_accepted,n_gt)` | `model_diversity_final_tuned/false_campaign_metrics.py` @ `61a8203` | **Not wired** to balanced runner imports | Peer only (`INFERRED` similarity) |
| `campaign_evaluation.compute_campaign_metrics` | same commit | Different field names/semantics; not `extract_run_metrics` | Not P7/P8 |
| Hungarian / IoU / descriptor overlap | — | No source found | `UNRECOVERABLE` |

---

## 3. Campaign-level TP / FP / FN (operational reading)

Given only the saved columns, for scenarios with `n_true_campaign_clusters = 1`:

| Term | Operational meaning on saved rows |
|------|-----------------------------------|
| TP campaign | A run is a campaign TP if `n_pred ≥ 1` (then TP count = 1) |
| FP campaign clusters | `max(n_pred − 1, 0)` |
| FN campaign | `1` if `n_pred = 0`, else `0` |
| One-to-one at campaign *count* level | Yes: `TP + FP = n_pred`, `TP + FN = n_true` |
| DBSCAN noise (`-1`) | Not counted as a predicted campaign cluster (noise excluded from `n_predicted_campaign_clusters` by definition of that field); **how** noise was dropped is in missing code |
| Multiple pred clusters vs one GT | Extra clusters counted as FP; `fragmentation_rate=1` when `n_pred>1` |
| One pred cluster vs multiple GT | Not exercised in strong/weak (`n_true=1`); unrelated uses separate incorrect-merging flag |

**Grade:** `VERIFIED_FROM_PUBLICATION_ARTIFACT` for the accounting; **`UNRECOVERABLE`** for the predicate that marks a DBSCAN cluster as a “predicted campaign cluster”.

---

## 4. Attack family in matching?

No evidence in P7/P8 artifacts that attack family enters campaign matching. Membership columns are vehicle counts (`attacked_vehicles_*`, `benign_vehicles_included`), not attack-type purity for scoring.

**Grade:** `INFERRED` (absence in columns); scoring code still missing.

---

## 5. Bottom line

| Question | Answer |
|----------|--------|
| Exact matcher function recovered? | **No** |
| Exact threshold recovered? | **No** |
| Ablation Jaccard≥0.5 proven for P7/P8? | **No** |
| Artifact-consistent campaign score? | Count-based `min(n_pred,n_true)` accounting |
| Safe to implement as authoritative? | **No** — would invent a replacement |
