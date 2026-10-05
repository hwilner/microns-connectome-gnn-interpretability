"""Interpretability utilities for the connectome GNN project.

The package currently exposes the attribution backends
(:mod:`src.attribution`); ablation and sufficiency utilities are added
by their own planned subtasks.
"""

from .attribution import (
    AttributionResult,
    DifferentiableGraphModel,
    ExplainerResult,
    gnn_explainer_edge_mask,
    integrated_gradients,
)

__all__ = [
    "AttributionResult",
    "DifferentiableGraphModel",
    "ExplainerResult",
    "gnn_explainer_edge_mask",
    "integrated_gradients",
]
