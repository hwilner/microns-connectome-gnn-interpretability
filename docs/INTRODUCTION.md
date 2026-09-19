# Introduction — Paper 2: What Do the Connectome GNNs Learn? (Interpretability)

**Series note:** This is **Paper 2 of 4** in the MICrONS function-from-wiring series. **It builds directly on Paper 1** (`microns-tuning-from-wiring-benchmark`): Paper 1's trained models, fixed splits, and baseline features are the objects of analysis here. Papers 3–4 reuse this paper's motif attribution results.

## Background

Understanding how a nervous system computes requires more than knowing its wiring diagram and more than knowing what each neuron does — it requires knowing *which features of the wiring are responsible for which aspects of function*. Connectomics began with the complete EM reconstruction of the *C. elegans* nervous system [1], and has scaled to whole adult *Drosophila* brains [2] and, most recently, to a cubic millimeter of mouse visual cortex. The MICrONS (Machine Intelligence from Cortical Networks) project, funded by IARPA, reconstructed >200,000 cells and ~523 million synapses in mouse visual cortex and co-registered them with in vivo two-photon calcium imaging of ~75,000 of the same neurons during visual stimulation [3,4]. This pairing of synapse-resolution anatomy with per-neuron visual function turned the old structure–function question into a quantitative, learnable problem.

Paper 1 of this series exploited that pairing: it trained graph neural networks — message-passing models in the lineage of graph convolutional networks [5] — on the MICrONS wiring graph to predict orientation tuning, direction selectivity, and receptive-field properties from connectivity alone, and established (with baselines and topology nulls) *how much* functional information wiring contains. The fly field has shown the complementary result that wiring-constrained network models can reproduce measured activity across the *Drosophila* visual system [6]. And in the mouse, Ding et al. demonstrated a general wiring rule — excitatory neurons preferentially connect to neurons with similar visual responses ("like-to-like") — directly linking specific connectivity patterns to specific functional properties [7].

But a predictive model is not yet an explanation. Deep networks, including the deep image-synthesis models that best describe visual cortical responses [8], are black boxes. A GNN that predicts orientation tuning from wiring might use biologically meaningful structure — e.g., the like-to-like rule, cell-type-specific inhibitory motifs, or feedforward convergence — or it might exploit statistical quirks of the reconstruction that happen to correlate with function in this one animal. Distinguishing these possibilities is the goal of interpretability: attributing predictions to specific input structures, ablating those structures, and comparing the learned attributions against independently discovered wiring rules.

Interpretability matters doubly in connectomics. First, EM reconstructions contain segmentation errors, truncated axons at volume boundaries, and proofreading gaps; a model whose predictions hinge on such artifacts must be detected. Second, the scientific value of function-from-wiring prediction is precisely the set of *wiring motifs* it identifies as functionally relevant: these become hypotheses for perturbation experiments and constraints for circuit theory. The MICrONS 2025 papers provide a rich set of published wiring rules to test against — like-to-like connectivity among excitatory neurons [7] and a census of cell-type-specific inhibitory circuitry [9] — giving this paper's attributions independent ground truth.

## Prior work and gap

Attribution methods for GNNs (gradient-based saliency, integrated gradients, perturbation/explainer approaches) are well developed in general machine learning, but have seen essentially no application to large-scale EM connectomes. Existing biological analyses of MICrONS wiring rules [7,9] are correlational and hypothesis-driven: they test pre-specified patterns, not the full space of graph features a learned model might exploit. Conversely, Paper 1's models are predictive but uninterpreted. The gap: **no analysis links the learned predictive signal of connectome GNNs to specific, named synaptic motifs and cell-type connectivity patterns**, verified by ablation and by comparison with published wiring rules.

## Research questions

1. Which connectivity motifs (e.g., like-to-like excitatory convergence, specific interneuron targeting patterns, feedforward vs. recurrent structure) drive the GNN's tuning predictions?
2. Do the learned attributions recover known wiring rules [7,9], and do they identify candidate new rules not described in the literature?
3. What is the minimal set of connectivity statistics sufficient to reproduce the full GNN's predictive accuracy — i.e., can the black box be distilled into an interpretable feature set?
4. Are attributions stable across model seeds, architectures, and held-out neurons, or are they idiosyncratic to single fits?

## Data

| Resource | Contents | Scale | Access |
|---|---|---|---|
| Paper 1 trained GNNs, splits, baselines | Models and evaluation harness from the benchmark | Ensemble of trained models | This series |
| MICrONS EM connectome | Directed synaptic graph with cell-type labels | >200k cells, ~0.5B synapses | Open (microns-explorer.org) |
| MICrONS functional imaging | Tuning labels for attribution targets | ~75k co-registered neurons | Open (MICrONS Explorer) |
| Published wiring rules | Like-to-like rule; inhibitory census | Reference patterns | Literature [7,9] |

## Methods

We apply complementary attribution methods to Paper 1's trained models: gradient-based edge/node attributions and integrated gradients over the input connectivity, plus perturbation-based explanation that identifies the minimal edge subsets sufficient for each prediction. Attributions are aggregated across neurons sharing a tuning profile to recover population-level motifs, then quantified as enriched graphlets and cell-type-typed connection patterns. Within-model causal evidence comes from ablation: removing or shuffling the top-attributed motifs (while preserving degree sequences) and measuring the drop in predictive accuracy. Distillation experiments fit interpretable models (regularized linear models, gradient-boosted trees on motif-count features) to the GNN's predictions to estimate the sufficient feature set. All motifs are cross-checked against the like-to-like rule [7], the inhibitory connectivity census [9], and the fly wiring literature [2,6] for convergent principles.

## Expected contributions

1. An **attribution map** from specific connectivity patterns to specific predicted tuning properties, with ablation-validated importance scores.
2. A test of whether learned function-from-wiring models rediscover known wiring rules or find novel ones.
3. A distilled, interpretable minimal feature set — the transferable currency that Paper 3 uses for cross-area and cross-species generalization and Paper 4 uses as the fidelity criterion for graph sparsification.

## Scope and boundary

This repository contains planning, software, and synthetic-data tests. Ablation provides causal evidence *within the model* only: showing that a motif drives a GNN's prediction does not establish that the motif causally shapes tuning in the brain; such claims require experimental perturbation. We do not interpret dynamical simulations, and we defer cross-species validation of motifs to Paper 3.

## References

1. White, J. G., Southgate, E., Thomson, J. N. & Brenner, S. The structure of the nervous system of the nematode *Caenorhabditis elegans*. *Philosophical Transactions of the Royal Society of London. Series B, Biological Sciences* 314, 1–340 (1986). DOI: 10.1098/rstb.1986.0056
2. Dorkenwald, S., Matsliah, A., Sterling, A. R. et al. Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138 (2024). DOI: 10.1038/s41586-024-07558-y
3. MICrONS Consortium et al. Functional connectomics spanning multiple areas of mouse visual cortex. *Nature* 640, 435–447 (2025). DOI: 10.1038/s41586-025-08790-w
4. Turner, N. L., Macrina, T., Bae, J. A. et al. Reconstruction of neocortex: Organelles, compartments, cells, circuits, and activity. *Cell* 185, 1082–1100 (2022). DOI: 10.1016/j.cell.2022.01.023
5. Kipf, T. N. & Welling, M. Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations (ICLR)* (2017). arXiv:1609.02907
6. Lappalainen, J. K. et al. Connectome-constrained networks predict neural activity across the fly visual system. *Nature* 634, 1132–1140 (2024). DOI: 10.1038/s41586-024-07939-3
7. Ding, Z., Fahey, P. G., Papadopoulos, S. et al. Functional connectomics reveals general wiring rule in mouse visual cortex. *Nature* 640, 459–469 (2025). DOI: 10.1038/s41586-025-08840-3
8. Bashivan, P., Kar, K. & DiCarlo, J. J. Neural population control via deep image synthesis. *Science* 364, eaav9436 (2019). DOI: 10.1126/science.aav9436
9. Schneider-Mizell, C. M. et al. Inhibitory specificity from a connectomic census of mouse visual cortex. *Nature* 640, 448–458 (2025). DOI: 10.1038/s41586-024-07780-8
