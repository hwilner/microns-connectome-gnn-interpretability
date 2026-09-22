# Concept figure (Mermaid fallback) — Paper 2: Opening the Black Box of Connectome GNNs

> The canonical vector figure is `concept_figure.svg` in this directory. This file is a
> faithful Mermaid rendition kept as a text-based fallback (the PNG raster could not be
> committed through the available API tooling).

```mermaid
flowchart TD
    W[Connectome graph] --> G[Trained GNN<br/>black box]
    G --> P[Tuning prediction]
    G --> IG[Integrated gradients:<br/>wiggle / fade each synapse,<br/>sensitivity per edge]
    G --> GE[GNNExplainer:<br/>dimmer switch per synapse,<br/>smallest sufficient subcircuit]
    IG --> M[Ranked wiring motifs<br/>reciprocal pairs, feedforward triplets]
    GE --> M
    M --> AB[Ablation test:<br/>remove motif, does accuracy drop?]
    M -. compare .-> L[Known wiring rules:<br/>like-to-like · inhibitory census]
```
