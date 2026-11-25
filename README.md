# MICrONS Connectome-GNN Interpretability


![Concept figure: features learned by a GNN from the connectome are compared against known wiring rules to test whether the model rediscovers biological principles](docs/figures/01-concept-schematic.png)

*Figure 1: Features learned by a GNN from the connectome are compared against known wiring rules to test whether the model rediscovers biological principles. (Editable Mermaid source: [docs/figures/concept_figure.md](docs/figures/concept_figure.md).)*

## Introduction

This independent research repository plans and tracks an interpretability analysis of the companion benchmark connectome GNNs (`microns-tuning-from-wiring-benchmark`) — which graph features carry functional information, and how they relate to known wiring rules. It provides interpretability utilities for transparent review and extension; every computation ships with a synthetic-graph test so results stay auditable.

## Extended introduction

The companion benchmark showed how far connectome wiring alone goes in predicting neuronal tuning, benchmarked against degree and feature baselines with a rewired-graph topology null. This repository asks the follow-up question: *which* connectivity structures does the model rely on, and do they coincide with wiring rules reported in the MICrONS 2025 descriptive analyses (like-to-like connectivity, E/I balance)? The analysis has three stages. First, attribution (integrated gradients and a GNNExplainer-style edge mask, both implemented in `src/attribution.py` behind a small differentiable-model interface) ranks the edges and motifs each trained model depends on, calibrated against null baselines so rankings are not driven by architecture noise. Second, the ranked motifs are compared against published wiring rules that have been operationalized as quantitative, citable statements — agreements and genuinely novel attributed features are reported separately. Third, motif ablation and feature-sufficiency curves test causal-within-model necessity and sufficiency: removing top-attributed motif classes should degrade prediction, and a minimal motif set should recover most of the accuracy. All claims are model-explanation claims, not biological causal claims.

## Materials and methods

- **Attribution backends** (`src/attribution.py`): integrated gradients with fixed midpoint quadrature (bitwise deterministic, no randomness) and a GNNExplainer-style soft edge mask trained by gradient descent on a sparsity- and entropy-regularized objective (deterministic given seed). Both wrap a minimal `DifferentiableGraphModel` interface (`predict` scalar forward + `gradients`), so the companion benchmark artifacts are adapted without framework lock-in.
- **Pre-registration before inspection**: the attribution aggregation rule (issue #6) and the motif ablation order (issue #7) are committed to `docs/` before any empirical results are inspected; ranked motif tables (#19) and the agreement report (#16) reference them.
- **Wiring-rule comparison**: published rules are operationalized with full citations as encoded comparators (#8) before agreement is computed (#10) and reported (#16).
- **Ablation/sufficiency**: top-attributed motif classes are masked in the pre-registered order and accuracy re-scored (#13); sufficiency curves add classes back with bootstrap/seed-replicate uncertainty (#15). Within-model causality only.
- **Testing**: the full suite is data-free — a synthetic 14-node graph with a planted 3-edge predictive motif (gate 4.0 vs background ≤ 0.09) must be recovered by both backends, plus determinism, completeness, input-validation, contract-enforcement, and degenerate-graph tests. Run with `python -m pytest -q` (38 tests, < 1 s, NumPy only).

## Results

The pre-registered results package lives in [docs/RESULTS.md](docs/RESULTS.md): captioned paper-style Tables 1–5 and planned Figures 1–2, each mapped to its authorizing issue. Empirical cells are pending — the attribution chain is gated on the companion benchmark's release-harness artifacts. Completed validation (RESULTS.md Table 6, 2026-10-27, `main` @ `a16ebe2`): **38/38 synthetic tests pass** (Python 3.12.12, CPU, numpy 2.2.5, pytest 9.1.1, 0.65 s); integrated gradients recover the planted motif as top-3 with exact attribution 6.0 per edge (background max 0.043, completeness at rtol 1e-10, bitwise deterministic); the explainer recovers the planted motif with mask ≥ 0.9989 vs background ≤ 0.1344, robust across seeds. Real-data runs: deferred (needs environment/data — companion-benchmark artifacts, issue #3).

## Research plan

| Planned work | Issue | Depends on | Expected outcome |
|---|---|---|---|
| Attribution utilities with planted-motif tests | #1 (done) | — | Deterministic IG + explainer backends (this repo's `src/attribution.py`). |
| Attribution run on the companion benchmark models | #3 (umbrella: #6, #11, #14, #19) | #1; companion-benchmark release-harness artifacts | Ranked connectivity motifs per tuning property, null-calibrated. |
| Wiring-rule comparison | #2 (umbrella: #8, #10, #16) | #3 | Agreement report with MICrONS 2025 descriptive findings; novel features listed. |
| Ablation and feature sufficiency | #5 (umbrella: #7, #13, #15) | #3 | Causal-within-model accuracy drops; minimal sufficient motif sets. |
| Results package and manuscript outline | #4 (umbrella: #9, #12, #17, #18) | #2, #5 | Reproducible figures/tables; manuscript outline stating the dependence on the companion benchmark. |

**Current status:** attribution utilities implemented and validated on synthetic graphs (issue #1 closed); empirical work blocked on the companion benchmark's model artifacts.

## What is included

| Path | Contents |
|---|---|
| `src/` | Attribution backends (`attribution.py`); ablation, sufficiency, wiring-rule, table, and figure modules added by their own planned subtasks. |
| `tests/` | Synthetic planted-motif tests (data-free, deterministic). |
| `docs/` | [Pre-registered results package](docs/RESULTS.md), [concept figure source](docs/figures/concept_figure.md), and figures. |

## Use and validation

```bash
python -m pytest -q
```

## Keywords

MICrONS, connectomics, interpretability, graph neural networks, wiring rules, computational neuroscience, reproducible research.
