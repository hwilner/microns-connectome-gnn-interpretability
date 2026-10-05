"""Attribution utilities for connectome GNN interpretability.

This module implements the two attribution backends planned for the
interpretability analysis of trained connectome GNN artifacts
(integrated gradients and a GNNExplainer-style edge mask), plus the
minimal differentiable-model interface both backends wrap.

Design rationale
----------------
The trained model artifacts from the companion
``microns-tuning-from-wiring-benchmark`` repository are not yet
available inside this repository, so the backends are written against a
small abstract interface, :class:`DifferentiableGraphModel`, instead of
any concrete framework.  Any upstream artifact can later be adapted by
implementing that interface (a forward ``predict`` and an analytical or
automatic ``gradients`` method), while the synthetic tests in
``tests/test_attribution.py`` exercise the backends through exactly the
same interface with a planted ground-truth motif.

Everything here is deterministic given the caller's seed: integrated
gradients uses a fixed midpoint quadrature with no randomness at all,
and the explainer's only randomness is an optional seeded perturbation
of the initial mask logits.  Only NumPy is required, keeping the
repository dependency-light at the planning stage.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DifferentiableGraphModel",
    "AttributionResult",
    "ExplainerResult",
    "integrated_gradients",
    "gnn_explainer_edge_mask",
]

# Valid values for the ``attribute`` argument of ``integrated_gradients``.
_VALID_TARGETS = frozenset({"edges", "nodes"})


class DifferentiableGraphModel(abc.ABC):
    """Minimal interface a graph model must implement to be attributable.

    The interface deliberately mirrors what the upstream benchmark
    artifacts expose after adaptation: a scalar forward pass over a
    weighted directed graph, and gradients of that scalar with respect
    to both the node features and the edge weights.  Restricting to a
    scalar output keeps the attribution semantics unambiguous (one
    attribution value per input element); multi-output artifacts are
    wrapped one target logit at a time.

    Conventions
    -----------
    * ``node_features`` has shape ``(n_nodes, n_features)`` and dtype
      float.
    * ``edge_index`` has shape ``(2, n_edges)`` with integer node
      indices; row 0 holds sources and row 1 holds targets.
    * ``edge_weight`` has shape ``(n_edges,)`` and dtype float.
    """

    @abc.abstractmethod
    def predict(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
    ) -> float:
        """Evaluate the model on one graph.

        Args:
            node_features: Array of shape ``(n_nodes, n_features)``.
            edge_index: Integer array of shape ``(2, n_edges)`` with
                source indices in row 0 and target indices in row 1.
            edge_weight: Array of shape ``(n_edges,)`` with the weight
                of each directed edge.

        Returns:
            The scalar model output for the graph (e.g. a regression
            value or a single target logit).
        """

    @abc.abstractmethod
    def gradients(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_weight: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Differentiate :meth:`predict` with respect to the inputs.

        Args:
            node_features: Array of shape ``(n_nodes, n_features)``.
            edge_index: Integer array of shape ``(2, n_edges)``.
            edge_weight: Array of shape ``(n_edges,)``.

        Returns:
            A pair ``(d_node_features, d_edge_weight)`` where
            ``d_node_features`` has shape ``(n_nodes, n_features)`` and
            ``d_edge_weight`` has shape ``(n_edges,)``, holding the
            partial derivatives of :meth:`predict` with respect to each
            input element at the given point.
        """


@dataclass(frozen=True)
class AttributionResult:
    """Attributions produced by :func:`integrated_gradients`.

    Attributes:
        edge_attribution: Array of shape ``(n_edges,)`` with one
            attribution score per edge, or ``None`` when edges were not
            among the attributed modalities.
        node_attribution: Array of shape ``(n_nodes, n_features)`` with
            one attribution score per node feature, or ``None`` when
            nodes were not among the attributed modalities.
    """

    edge_attribution: Optional[np.ndarray]
    node_attribution: Optional[np.ndarray]

    def top_edges(self, k: int) -> np.ndarray:
        """Return the indices of the ``k`` edges with highest attribution.

        Ranking uses the signed attribution value (largest first), which
        matches the planted-motif use case where predictive edges have
        large positive scores.  Ties are broken by edge index so the
        result is deterministic even for tied scores.

        Args:
            k: Number of top edges to return; must satisfy
                ``1 <= k <= n_edges``.

        Returns:
            Integer array of shape ``(k,)`` with edge indices ordered by
            decreasing attribution.

        Raises:
            ValueError: If no edge attribution is available, or if ``k``
                is outside ``[1, n_edges]``.
        """
        if self.edge_attribution is None:
            raise ValueError("no edge attribution available in this result")
        n_edges = self.edge_attribution.shape[0]
        if not 1 <= k <= n_edges:
            raise ValueError(
                f"k must be in [1, {n_edges}], got {k}"
            )
        # np.argsort is ascending; sort the negated attributions with a
        # stable algorithm so exact ties resolve to the lower edge index.
        order = np.argsort(-self.edge_attribution, kind="stable")
        return order[:k]


@dataclass(frozen=True)
class ExplainerResult:
    """Result of :func:`gnn_explainer_edge_mask`.

    Attributes:
        edge_mask: Array of shape ``(n_edges,)`` with the learned soft
            edge mask; each entry lies in the open interval ``(0, 1)``,
            and larger values mark edges the explainer kept as
            important for the model output.
        loss_history: Array of shape ``(steps,)`` with the total loss
            after every optimization step, recorded so the optimization
            trajectory stays auditable.
    """

    edge_mask: np.ndarray
    loss_history: np.ndarray


def _validate_graph(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce and validate the shared graph inputs.

    Centralizing the checks keeps both public backends consistent and
    gives callers one informative error per failure mode.

    Args:
        node_features: Candidate ``(n_nodes, n_features)`` float array.
        edge_index: Candidate ``(2, n_edges)`` integer index array.
        edge_weight: Candidate ``(n_edges,)`` float array.

    Returns:
        The three inputs as ``np.ndarray`` objects with dtypes
        ``float64``, ``int64``, and ``float64`` respectively.

    Raises:
        ValueError: If ``node_features`` is not a finite 2-D array with
            at least one node, if ``edge_index`` is not a ``(2, n_edges)``
            integer array with all indices inside ``[0, n_nodes)``, or if
            ``edge_weight`` is not a finite 1-D array whose length matches
            the number of edges.
    """
    node_features = np.asarray(node_features, dtype=np.float64)
    edge_index = np.asarray(edge_index)
    edge_weight = np.asarray(edge_weight, dtype=np.float64)

    if node_features.ndim != 2 or node_features.shape[0] == 0:
        raise ValueError(
            "node_features must be a 2-D array with at least one node, "
            f"got shape {node_features.shape}"
        )
    if not np.all(np.isfinite(node_features)):
        raise ValueError("node_features must contain only finite values")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(
            "edge_index must have shape (2, n_edges), "
            f"got shape {edge_index.shape}"
        )
    if not np.issubdtype(edge_index.dtype, np.integer):
        # Reject float indices even when integral-valued: silent rounding
        # of a fractional index would silently rewire the graph.
        raise ValueError("edge_index must contain integer node indices")
    edge_index = edge_index.astype(np.int64)
    n_nodes = node_features.shape[0]
    if edge_index.size:
        if int(edge_index.min()) < 0 or int(edge_index.max()) >= n_nodes:
            raise ValueError(
                f"edge_index entries must lie in [0, {n_nodes}), got "
                f"min {int(edge_index.min())}, max {int(edge_index.max())}"
            )

    n_edges = edge_index.shape[1]
    if edge_weight.ndim != 1 or edge_weight.shape[0] != n_edges:
        raise ValueError(
            f"edge_weight must be 1-D with length {n_edges}, "
            f"got shape {edge_weight.shape}"
        )
    if not np.all(np.isfinite(edge_weight)):
        raise ValueError("edge_weight must contain only finite values")

    return node_features, edge_index, edge_weight


def _check_scalar_output(value: object, *, context: str) -> float:
    """Validate that a model forward pass returned a finite scalar.

    Args:
        value: The raw return value of ``model.predict``.
        context: Name of the calling backend, used in error messages.

    Returns:
        The value as a plain Python ``float``.

    Raises:
        ValueError: If ``value`` is not a finite scalar.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 0 or not np.isfinite(arr):
        raise ValueError(
            f"{context}: model.predict must return a finite scalar, "
            f"got {value!r}"
        )
    return float(arr)


def _check_gradients(
    d_node: np.ndarray,
    d_edge: np.ndarray,
    node_shape: Tuple[int, int],
    n_edges: int,
    *,
    context: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate gradient arrays returned by ``model.gradients``.

    Catching a malformed gradient here prevents silently wrong
    attributions when a new artifact adapter is wired up incorrectly.

    Args:
        d_node: Gradient with respect to node features.
        d_edge: Gradient with respect to edge weights.
        node_shape: Expected shape of ``d_node``.
        n_edges: Expected length of ``d_edge``.
        context: Name of the calling backend, used in error messages.

    Returns:
        The gradients as finite ``float64`` arrays.

    Raises:
        ValueError: If either gradient has the wrong shape or contains
            non-finite values.
    """
    d_node = np.asarray(d_node, dtype=np.float64)
    d_edge = np.asarray(d_edge, dtype=np.float64)
    if d_node.shape != node_shape:
        raise ValueError(
            f"{context}: node-feature gradient must have shape "
            f"{node_shape}, got {d_node.shape}"
        )
    if d_edge.shape != (n_edges,):
        raise ValueError(
            f"{context}: edge-weight gradient must have shape "
            f"({n_edges},), got {d_edge.shape}"
        )
    if not np.all(np.isfinite(d_node)) or not np.all(np.isfinite(d_edge)):
        raise ValueError(f"{context}: gradients must be finite")
    return d_node, d_edge


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid.

    Args:
        z: Input array of arbitrary shape.

    Returns:
        ``1 / (1 + exp(-z))`` evaluated element-wise without overflow
        for large-magnitude inputs (the positive and negative branches
        are computed with separate stable expressions).
    """
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    # For z < 0, exp(z) is small, so this form avoids exp(-z) overflow.
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _require_positive_int(value: int, *, name: str) -> int:
    """Validate a positive integer hyperparameter.

    Args:
        value: Candidate value.
        name: Parameter name for error messages.

    Returns:
        The value as a plain Python ``int``.

    Raises:
        ValueError: If ``value`` is not an integer or is below 1.
    """
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def integrated_gradients(
    model: DifferentiableGraphModel,
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    *,
    node_baseline: Optional[np.ndarray] = None,
    edge_baseline: Optional[np.ndarray] = None,
    steps: int = 64,
    attribute: Sequence[str] = ("edges",),
) -> AttributionResult:
    """Compute integrated-gradients attributions for a graph model.

    Integrated gradients (Sundararajan-style) attributes the difference
    between the model output at the input and at a baseline by
    integrating the input-gradient along the straight-line path between
    them.  The integral is approximated with a fixed midpoint rule, so
    the computation involves no randomness and is bitwise deterministic.

    Each entry of ``attribute`` selects one modality to explain; the
    path interpolation is restricted to the selected modalities while
    the others stay fixed at their input values.  Per-modality paths
    keep attributions directly comparable across edges (or nodes), which
    is what the downstream ranked-motif tables consume.

    Args:
        model: A :class:`DifferentiableGraphModel` instance.
        node_features: ``(n_nodes, n_features)`` input features.
        edge_index: ``(2, n_edges)`` integer edge indices.
        edge_weight: ``(n_edges,)`` input edge weights.
        node_baseline: Baseline for node features; defaults to an
            all-zeros array (the standard "absent signal" baseline).
        edge_baseline: Baseline for edge weights; defaults to zeros,
            corresponding to the graph with every edge removed.
        steps: Number of midpoint quadrature points along the path.
        attribute: Iterable containing ``"edges"`` and/or ``"nodes"``.

    Returns:
        An :class:`AttributionResult`; each modality's attribution array
        is populated only if that modality was requested.  With a single
        requested modality the attributions satisfy the completeness
        property: they sum to ``predict(input) - predict(baseline)`` up
        to quadrature error.

    Raises:
        ValueError: If the graph inputs fail validation, if ``steps`` is
            not a positive integer, if ``attribute`` is empty or holds
            unknown modalities, if a baseline has the wrong shape, or if
            the model returns malformed outputs or gradients.
    """
    node_features, edge_index, edge_weight = _validate_graph(
        node_features, edge_index, edge_weight
    )
    steps = _require_positive_int(steps, name="steps")

    targets = tuple(attribute)
    if not targets or not set(targets) <= _VALID_TARGETS:
        raise ValueError(
            f"attribute must be a non-empty subset of "
            f"{sorted(_VALID_TARGETS)}, got {targets!r}"
        )

    n_edges = edge_index.shape[1]
    if node_baseline is None:
        node_baseline = np.zeros_like(node_features)
    else:
        node_baseline = np.asarray(node_baseline, dtype=np.float64)
        if node_baseline.shape != node_features.shape:
            raise ValueError(
                f"node_baseline must have shape {node_features.shape}, "
                f"got {node_baseline.shape}"
            )
    if edge_baseline is None:
        edge_baseline = np.zeros_like(edge_weight)
    else:
        edge_baseline = np.asarray(edge_baseline, dtype=np.float64)
        if edge_baseline.shape != edge_weight.shape:
            raise ValueError(
                f"edge_baseline must have shape {edge_weight.shape}, "
                f"got {edge_baseline.shape}"
            )

    want_edges = "edges" in targets
    want_nodes = "nodes" in targets

    # Midpoint quadrature nodes along the path parameter alpha in (0, 1).
    alphas = (np.arange(steps, dtype=np.float64) + 0.5) / steps

    node_grad_sum = np.zeros_like(node_features)
    edge_grad_sum = np.zeros_like(edge_weight)
    for alpha in alphas:
        # Interpolate only the attributed modalities; the fixed
        # modalities remain at their input values.
        x = node_baseline + alpha * (node_features - node_baseline) if want_nodes else node_features
        w = edge_baseline + alpha * (edge_weight - edge_baseline) if want_edges else edge_weight
        d_node, d_edge = model.gradients(x, edge_index, w)
        d_node, d_edge = _check_gradients(
            d_node, d_edge, node_features.shape, n_edges,
            context="integrated_gradients",
        )
        node_grad_sum += d_node
        edge_grad_sum += d_edge

    node_attr = None
    edge_attr = None
    if want_nodes:
        node_attr = (node_features - node_baseline) * (node_grad_sum / steps)
    if want_edges:
        edge_attr = (edge_weight - edge_baseline) * (edge_grad_sum / steps)

    return AttributionResult(
        edge_attribution=edge_attr,
        node_attribution=node_attr,
    )


def gnn_explainer_edge_mask(
    model: DifferentiableGraphModel,
    node_features: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    *,
    steps: int = 200,
    learning_rate: float = 0.5,
    sparsity_weight: float = 0.01,
    entropy_weight: float = 0.01,
    init_logit: float = 0.0,
    init_scale: float = 0.0,
    seed: int = 0,
) -> ExplainerResult:
    """Learn a soft edge mask with a GNNExplainer-style objective.

    A learnable logit ``m_e`` per edge gates the edge weight as
    ``w_e * sigmoid(m_e)``.  The loss maximizes the model output under
    the masked graph while penalizing mask mass and mask entropy::

        L(m) = -f(x, w * sigmoid(m))
               + sparsity_weight * sum(sigmoid(m))
               + entropy_weight * H(sigmoid(m))

    so that after optimization the largest mask entries mark the edges
    that most increase the model output, while the penalties keep the
    mask sparse and push entries toward 0 or 1.  This is the
    "maximize target score" variant of the explainer; for artifacts
    with per-class logits the same form is applied to one target logit.

    The gradient with respect to the logits follows from the chain rule
    through the multiplicative gate and the sigmoid::

        dL/dm_e = (-g_e * w_e + sparsity_weight
                   - entropy_weight * m_e) * s_e * (1 - s_e)

    where ``g_e`` is the model's gradient with respect to the masked
    weight of edge ``e`` and ``s_e = sigmoid(m_e)``; the term linear in
    ``m_e`` is the derivative of the entropy regularizer.

    Optimization is plain gradient descent with a fixed step count and
    step size, and the only randomness is an optional seeded uniform
    perturbation of the initial logits, so results are bitwise
    deterministic given ``seed``.

    Args:
        model: A :class:`DifferentiableGraphModel` instance.
        node_features: ``(n_nodes, n_features)`` input features.
        edge_index: ``(2, n_edges)`` integer edge indices.
        edge_weight: ``(n_edges,)`` input edge weights.
        steps: Number of gradient-descent steps.
        learning_rate: Gradient-descent step size.
        sparsity_weight: Non-negative coefficient of the L1-style mask
            penalty; larger values produce sparser masks.
        entropy_weight: Non-negative coefficient of the element-wise
            entropy penalty; larger values push mask entries toward the
            extremes 0 and 1.
        init_logit: Constant initial value for every mask logit
            (``0.0`` starts every edge at mask value 0.5).
        init_scale: Half-width of an optional uniform perturbation of
            the initial logits; ``0.0`` disables it.
        seed: Seed for the perturbation generator; only used when
            ``init_scale > 0``.

    Returns:
        An :class:`ExplainerResult` with the final soft mask in
        ``(0, 1)`` and the per-step loss history.

    Raises:
        ValueError: If the graph inputs fail validation, if ``steps`` is
            not a positive integer, if any continuous hyperparameter is
            non-finite, negative where non-negativity is required, or
            non-positive for ``learning_rate``, or if the model returns
            malformed outputs or gradients.
    """
    node_features, edge_index, edge_weight = _validate_graph(
        node_features, edge_index, edge_weight
    )
    steps = _require_positive_int(steps, name="steps")

    learning_rate = float(learning_rate)
    sparsity_weight = float(sparsity_weight)
    entropy_weight = float(entropy_weight)
    init_logit = float(init_logit)
    init_scale = float(init_scale)
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError(
            f"learning_rate must be positive and finite, got {learning_rate}"
        )
    for value, name in (
        (sparsity_weight, "sparsity_weight"),
        (entropy_weight, "entropy_weight"),
        (init_scale, "init_scale"),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be non-negative and finite, got {value}")
    if not np.isfinite(init_logit):
        raise ValueError(f"init_logit must be finite, got {init_logit}")

    n_edges = edge_index.shape[1]
    logits = np.full(n_edges, init_logit, dtype=np.float64)
    if init_scale > 0.0:
        rng = np.random.default_rng(seed)
        logits = logits + rng.uniform(-init_scale, init_scale, size=n_edges)

    loss_history = np.empty(steps, dtype=np.float64)
    for step in range(steps):
        mask = _sigmoid(logits)
        masked_weight = edge_weight * mask
        output = _check_scalar_output(
            model.predict(node_features, edge_index, masked_weight),
            context="gnn_explainer_edge_mask",
        )
        # The node-feature gradient is validated even though the mask
        # objective only needs the edge factor: a broken adapter should
        # surface here rather than at first node-side use.
        d_node, d_edge = model.gradients(node_features, edge_index, masked_weight)
        _, d_edge = _check_gradients(
            d_node, d_edge,
            node_features.shape, n_edges,
            context="gnn_explainer_edge_mask",
        )

        # Total loss, recorded for auditability of the trajectory.
        entropy = -np.sum(
            mask * np.log(mask) + (1.0 - mask) * np.log1p(-mask)
        )
        loss_history[step] = (
            -output
            + sparsity_weight * float(np.sum(mask))
            + entropy_weight * float(entropy)
        )

        # Chain rule through w_e * sigmoid(m_e): the model-side factor
        # is d_edge * w_e, and dsigmoid/dlogit = mask * (1 - mask).
        # The -entropy_weight * logits term is dH/dlogit because
        # dH/dsigmoid = -logit(sigmoid) = -logits.
        grad_logits = (
            -d_edge * edge_weight
            + sparsity_weight
            - entropy_weight * logits
        ) * mask * (1.0 - mask)
        logits = logits - learning_rate * grad_logits
        # Keep logits in a range where sigmoid stays strictly inside
        # (0, 1) in float64, so the entropy term can never produce NaN.
        logits = np.clip(logits, -30.0, 30.0)

    return ExplainerResult(
        edge_mask=_sigmoid(logits),
        loss_history=loss_history,
    )
