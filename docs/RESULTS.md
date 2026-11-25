# Results — MICrONS Connectome-GNN Interpretability

This document pre-registers the results package in paper style: every
table and figure below is captioned and mapped to the issue that
authorizes it, so reporting stays auditable before any empirical
analysis is run. Tables 1–5 are **placeholders** whose cells fill in as
the corresponding issues close; Table 6 records completed validation
work. The attribution-analysis chain is gated on the trained-model
artifacts of the companion `microns-tuning-from-wiring-benchmark`
repository.

## Pre-registered tables

**Table 1. Attribution run summary per tuning property.** One row per
tuning property analyzed on the companion benchmark models: number of graphs
attributed, attribution backend(s) applied, null-baseline calibration
(shuffled-label / randomized-model controls), and the count of motif
classes surviving null calibration. *Authorized by issue #14 (run
attribution with null baselines), which depends on #11 (attribution run
pipeline), in turn on #6 (pre-registered aggregation rule); parent
umbrella #3. Status: pending — gated on the companion benchmark's release-harness
artifacts.*

| Tuning property | Graphs attributed | Backend(s) | Null control | Motif classes passing null calibration |
|---|---|---|---|---|
| *(pending)* | — | — | — | — |

**Table 2. Ranked connectivity-motif tables per tuning property.** The
versioned headline ranking: for each tuning property, motif classes
ranked by the pre-registered aggregation statistic, with per-graph
pooling, ranking statistic, and multiplicity handling exactly as fixed
in the aggregation pre-registration (#6). Each table ships with
config/seed metadata. *Authorized by issue #19 (versioned ranked motif
tables and summary), depends on #14; parent umbrella #3. Status:
pending.*

| Tuning property | Rank | Motif class | Aggregated attribution (pre-registered statistic) | Null-calibrated significance |
|---|---|---|---|---|
| *(pending)* | — | — | — | — |

**Table 3. Wiring-rule agreement report.** One row per operationalized
published wiring rule (like-to-like connectivity, E/I balance, MICrONS
2025 descriptive findings), with full citations, the formal comparator
output per tuning property, and an explicit novel-features section
listing attributed features that match no published rule. *Authorized
by issue #16 (agreement report), depends on #10 (agreement
computation), in turn on #8 (operationalize wiring rules with
citations); parent umbrella #2. Status: pending.*

| Wiring rule (citation) | Formal definition | Agreement record per tuning property | Novel attributed features |
|---|---|---|---|
| *(pending)* | — | — | — |

**Table 4. Motif ablation and minimal-feature sufficiency.** Ablation
rows: prediction-accuracy change after removing/masking each
top-attributed motif class in the pre-registered order (#7).
Sufficiency rows: accuracy recovered as motif classes are added back in
that same order, with bootstrap or seed-replicate uncertainty bands per
tuning property. Within-model causality only. *Authorized by issues #13
(implement ablation/masking) and #15 (sufficiency curves with
uncertainty); parent umbrella #5. Status: pending.*

| Tuning property | Motif class ablated/added (pre-registered order) | Accuracy change vs full model | Uncertainty (bootstrap / seeds) |
|---|---|---|---|
| *(pending)* | — | — | — |

**Table 5. Results-package assembly record.** Regeneration record for
the versioned artifacts: one command per table/figure family, the clean
checkout re-run outcome, environment and seeds, and any deviations.
*Authorized by issues #12 (results tables), #9 (figure scripts), #17
(reproducibility re-run), and #18 (manuscript outline); parent umbrella
#4. Status: pending.*

| Artifact family | Regeneration command | Deterministic given seed? | Clean-checkout re-run | Deviations |
|---|---|---|---|---|
| *(pending)* | — | — | — | — |

## Planned figures

**Figure 1 (planned). Attribution-ranked motif schematic.** Per tuning
property, the top-ranked connectivity motifs from Table 2 rendered on
the graph, with null-calibration significance encoded. *Authorized by
issue #9 (figure scripts), consuming Tables 1–2. Status: pending.*

**Figure 2 (planned). Agreement and sufficiency curves.** Left panel:
per-rule agreement between attributions and operationalized wiring
rules (Table 3). Right panel: minimal-feature sufficiency curves with
uncertainty bands (Table 4). *Authorized by issue #9, consuming Tables
3–4. Status: pending.*

## Completed validation

**Table 6. Synthetic validation record (2026-10-27, `main` @
`a16ebe2`).** Data-free validation of the attribution backends
(`src/attribution.py`, issue #1) on the planted-motif synthetic graph
of `tests/test_attribution.py` (14 nodes, 43 edges; planted directed
3-edge chain with gate exactly 4.0 vs background gates ≤ 0.09).
Environment: Python 3.12.12, CPU, numpy 2.2.5, pytest 9.1.1. Executed
files verified bit-identical to the committed blobs by git blob-SHA
parity (`conftest.py` 9d8929d, `src/__init__.py` 62a5cda,
`src/attribution.py` b5a0c5a, `tests/test_attribution.py` 1afaab1 @
`main` a16ebe2). Real-data attribution runs are deferred (needs
environment/data — the companion benchmark's release-harness artifacts, issue #3).

| Check | Expected (planted ground truth) | Observed | Outcome |
|---|---|---|---|
| Test suite (`python -m pytest -q`) | all pass | 38/38 pass in 0.65 s | PASS |
| Integrated gradients, top-3 edges | planted chain edges | top-3 = planted set | PASS |
| IG planted-edge attribution | exactly gate × weight = 4.0 × 1.5 = 6.0 | 6.0, 6.0, 6.0 (rtol/atol 1e-12) | PASS |
| IG max \|background\| attribution | ≪ planted | 0.043 | PASS |
| IG edge completeness | sums to predict(input) − predict(baseline) | holds at rtol 1e-10 | PASS |
| IG bitwise determinism | identical arrays on repeat | identical (edges + nodes) | PASS |
| Explainer top-3 mask entries | planted chain edges | top-3 = planted set | PASS |
| Explainer mask separation | planted → 1, background < 0.5 | planted min 0.9989; background max 0.1344 | PASS |
| Explainer loss trajectory | finite, decreasing | −6.51 → −17.26 over 300 steps | PASS |
| Explainer seed robustness | different inits, same top edges | seeds 1 vs 2: same planted top-3 | PASS |
| Input/contract validation | malformed graphs, gradients, outputs rejected | all rejection tests pass | PASS |
| Empty-edge graph | no crash, correct shapes | shapes (0,) / (steps,) preserved | PASS |
