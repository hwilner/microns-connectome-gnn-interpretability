# Introduction — Paper 2: What Do the Connectome GNNs Learn? (Interpretability)

**Series note:** This is **Paper 2 of 4** in the MICrONS function-from-wiring series. **It builds directly on Paper 1** (`microns-tuning-from-wiring-benchmark`): Paper 1's trained models, fixed splits, and baseline features are the objects of analysis here. Papers 3–4 reuse this paper's motif attribution results.

## Background

Paper 1 establishes *whether* wiring predicts tuning; this paper asks *what graph structure* the predictors use. Comparing learned attributions against known wiring rules (like-to-like connectivity, excitatory/inhibitory balance, motif frequencies from the MICrONS 2025 descriptive papers) converts a black-box benchmark into biological insight — with ablation providing within-model causal evidence.

## Research questions

1. Which connectivity motifs drive the GNN's tuning predictions?
2. Do learned features recover known wiring rules, or find new ones?
3. What is the minimal set of connectivity statistics sufficient for prediction?

## Data

| Resource | Scale | Access |
|---|---|---|
| Paper 1 models, splits, features | internal | This series |
| MICrONS EM + functional | >200k cells | Open |

## Methods

Integrated gradients / GNNExplainer attributions, motif ablation, feature-sufficiency curves, comparison to published wiring rules.

## Expected contributions

- An attribution map linking specific connectivity patterns to tuning predictability.
- A principled minimal feature set used by Paper 3 for transfer.

## Scope and boundary

Planning, software, and synthetic tests live here. Within-model causal ablation does not imply biological causation.
