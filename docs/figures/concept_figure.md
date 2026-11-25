# Concept figure (Mermaid fallback) — Opening the Black Box of Connectome GNNs

> **Note.** The rendered concept figure is [`01-concept-schematic.png`](01-concept-schematic.png) in this folder — a generated scientific illustration. This file keeps the Mermaid source of the same diagram so it remains editable and re-renderable.

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
