"""Synthetic planted-motif tests for :mod:`src.attribution`.

The real model artifacts from the companion
``microns-tuning-from-wiring-benchmark`` repository are not yet
available here, so these tests build a small synthetic graph whose
ground-truth predictive motif is fully controlled: a directed chain of
three edges whose endpoint features produce an outsized, exactly known
contribution to the model output.  Both attribution backends must
recover those three planted edges as their top-ranked edges, and both
must be bitwise deterministic under fixed seeds.
"""

import numpy as np
import pytest

from src.attribution import (
    AttributionResult,
    DifferentiableGraphModel,
    ExplainerResult,
    gnn_explainer_edge_mask,
    integrated_gradients,
)

# Graph dimensions used across the suite.  Ten background nodes carry
# only weak random structure; four motif nodes carry the planted chain.
N_BACKGROUND_NODES = 10
N_MOTIF_NODES = 4
N_FEATURES = 6
N_BACKGROUND_EDGES = 40
BACKGROUND_WEIGHT_LOW = 0.2
BACKGROUND_WEIGHT_HIGH = 0.8
PLANTED_WEIGHT = 1.5
# Feature value planted at feature index 0 of every motif node; the
# model's edge gate reads exactly this feature pair (see below).
MOTIF_FEATURE_VALUE = 2.0
BACKGROUND_FEATURE_LIMIT = 0.3


class PlantedMotifModel(DifferentiableGraphModel):
    """Synthetic edge-gated message-passing readout with exact gradients.

    The scalar output is

    .. code-block:: text

        y = sum_e w_e * (v . (x_src(e) * x_dst(e)))  +  b * mean(tanh(x @ c))

    The first term is a one-hop edge-gated aggregation: each directed
    edge contributes its weight times a gate computed from the element-
    wise product of its endpoint features, which is the quantity the
    attribution backends should assign credit to.  The second term is a
    smooth, edge-independent readout that keeps node-feature gradients
    non-trivial without affecting edge attributions.

    Because the output is linear in the edge weights with gates that do
    not themselves depend on the weights, the edge gradient is constant
    along the integrated-gradients path, so edge attributions are exact
    (not just converged) for any step count.
    """

    def __init__(self, gate_vector, readout_weights, readout_gain):
        """Store fixed model parameters.

        Args:
            gate_vector: Array of shape ``(n_features,)``; weights the
                endpoint-feature product in each edge gate.
            readout_weights: Array of shape ``(n_features,)``; mixes
                node features inside the tanh readout.
            readout_gain: Scalar multiplier of the readout term.
        """
        self._gate_vector = np.asarray(gate_vector, dtype=np.float64)
        self._readout_weights = np.asarray(readout_weights, dtype=np.float64)
        self._readout_gain = float(readout_gain)

    def _edge_gates(self, node_features, edge_index):
        """Compute the per-edge gate values ``v . (x_src * x_dst)``.

        Args:
            node_features: ``(n_nodes, n_features)`` feature array.
            edge_index: ``(2, n_edges)`` edge indices.

        Returns:
            Array of shape ``(n_edges,)`` with one gate per edge.
        """
        src, dst = edge_index[0], edge_index[1]
        endpoint_product = node_features[src] * node_features[dst]
        return endpoint_product @ self._gate_vector

    def predict(self, node_features, edge_index, edge_weight):
        """Evaluate the synthetic readout on one graph.

        Args:
            node_features: ``(n_nodes, n_features)`` feature array.
            edge_index: ``(2, n_edges)`` edge indices.
            edge_weight: ``(n_edges,)`` edge weights.

        Returns:
            Scalar output as a Python ``float``.
        """
        gates = self._edge_gates(node_features, edge_index)
        edge_term = float(edge_weight @ gates)
        readout = float(np.mean(np.tanh(node_features @ self._readout_weights)))
        return edge_term + self._readout_gain * readout

    def gradients(self, node_features, edge_index, edge_weight):
        """Return exact analytical gradients of :meth:`predict`.

        Args:
            node_features: ``(n_nodes, n_features)`` feature array.
            edge_index: ``(2, n_edges)`` edge indices.
            edge_weight: ``(n_edges,)`` edge weights (unused by the
                analytical expressions, which are weight-independent for
                edges; kept for interface fidelity).

        Returns:
            Pair ``(d_node_features, d_edge_weight)`` with shapes
            ``(n_nodes, n_features)`` and ``(n_edges,)``.
        """
        n_nodes = node_features.shape[0]
        src, dst = edge_index[0], edge_index[1]

        # d y / d w_e is simply the edge gate.
        d_edge_weight = self._edge_gates(node_features, edge_index)

        # d y / d x_i accumulates, for every edge touching i, the gate
        # derivative: v * x_other endpoint, weighted by the edge weight.
        d_node_features = np.zeros_like(node_features)
        if edge_index.shape[1]:
            np.add.at(
                d_node_features, src,
                edge_weight[:, None] * self._gate_vector[None, :]
                * node_features[dst],
            )
            np.add.at(
                d_node_features, dst,
                edge_weight[:, None] * self._gate_vector[None, :]
                * node_features[src],
            )

        # Readout term derivative: b/N * (1 - tanh^2(x c)) * c per node.
        mixed = node_features @ self._readout_weights
        tanh_derivative = 1.0 - np.tanh(mixed) ** 2
        d_node_features += (
            self._readout_gain / n_nodes
        ) * tanh_derivative[:, None] * self._readout_weights[None, :]

        return d_node_features, d_edge_weight


def build_planted_graph(seed=20260919):
    """Construct the synthetic graph, model, and planted-motif indices.

    The gate vector is the indicator of feature 0, so an edge's gate is
    exactly ``x_src[0] * x_dst[0]``.  Background nodes draw feature 0
    from ``U(-0.3, 0.3)`` while motif nodes hold ``+2.0``, giving planted
    gates of ``4.0`` against background gates bounded by ``0.09`` — a
    margin of over 40x before weights are even considered, so recovery
    of the planted motif cannot hinge on quadrature or optimizer noise.

    Args:
        seed: Seed for all random construction; fixed by default so the
            whole suite shares one graph.

    Returns:
        A tuple ``(model, node_features, edge_index, edge_weight,
        planted_edges)`` where ``planted_edges`` lists the row indices
        of the three planted chain edges within ``edge_index``.
    """
    rng = np.random.default_rng(seed)
    n_nodes = N_BACKGROUND_NODES + N_MOTIF_NODES

    node_features = np.empty((n_nodes, N_FEATURES), dtype=np.float64)
    node_features[:, 0] = rng.uniform(
        -BACKGROUND_FEATURE_LIMIT, BACKGROUND_FEATURE_LIMIT, size=n_nodes
    )
    node_features[:, 1:] = rng.normal(
        0.0, 0.4, size=(n_nodes, N_FEATURES - 1)
    )
    motif_nodes = np.arange(N_BACKGROUND_NODES, n_nodes)
    node_features[motif_nodes, 0] = MOTIF_FEATURE_VALUE

    # Background edges connect background nodes only, so no background
    # edge can inherit a large gate from a motif endpoint.
    pairs = rng.integers(
        0, N_BACKGROUND_NODES, size=(N_BACKGROUND_EDGES, 2)
    )
    self_loops = pairs[:, 0] == pairs[:, 1]
    pairs[self_loops, 1] = (pairs[self_loops, 1] + 1) % N_BACKGROUND_NODES
    background_weight = rng.uniform(
        BACKGROUND_WEIGHT_LOW, BACKGROUND_WEIGHT_HIGH,
        size=N_BACKGROUND_EDGES,
    )

    # Planted motif: a directed chain across the four motif nodes.
    planted_pairs = np.column_stack([motif_nodes[:-1], motif_nodes[1:]])
    planted_weight = np.full(N_MOTIF_NODES - 1, PLANTED_WEIGHT)

    edge_index = np.concatenate(
        [pairs.T, planted_pairs.T], axis=1
    ).astype(np.int64)
    edge_weight = np.concatenate([background_weight, planted_weight])
    planted_edges = list(
        range(N_BACKGROUND_EDGES, N_BACKGROUND_EDGES + N_MOTIF_NODES - 1)
    )

    gate_vector = np.zeros(N_FEATURES)
    gate_vector[0] = 1.0
    readout_weights = rng.normal(0.0, 0.3, size=N_FEATURES)
    model = PlantedMotifModel(gate_vector, readout_weights, readout_gain=0.7)

    return model, node_features, edge_index, edge_weight, planted_edges


@pytest.fixture(scope="module")
def planted_graph():
    """Provide the shared synthetic planted-motif graph."""
    return build_planted_graph()


class TestIntegratedGradients:
    """Integrated-gradients behavior on the planted-motif graph."""

    def test_recovers_planted_motif_as_top_attribution(self, planted_graph):
        """The three planted chain edges must be the top-3 attributions."""
        model, x, edge_index, w, planted = planted_graph
        result = integrated_gradients(
            model, x, edge_index, w, attribute=("edges",), steps=32
        )
        assert set(result.top_edges(3).tolist()) == set(planted)

    def test_edge_attribution_exact_for_linear_edge_term(self, planted_graph):
        """Planted attributions equal gate * weight to float precision.

        The synthetic model is linear in the edge weights with gates
        independent of the weights, so the midpoint quadrature of a
        constant gradient is exact at any step count; this test guards
        that structural property (planted gate is exactly 2.0 * 2.0).
        """
        model, x, edge_index, w, planted = planted_graph
        result = integrated_gradients(
            model, x, edge_index, w, attribute=("edges",), steps=16
        )
        expected = MOTIF_FEATURE_VALUE**2 * PLANTED_WEIGHT
        np.testing.assert_allclose(
            result.edge_attribution[planted],
            np.full(len(planted), expected),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_edge_completeness_holds(self, planted_graph):
        """Edge attributions sum to predict(input) - predict(baseline)."""
        model, x, edge_index, w, planted = planted_graph
        result = integrated_gradients(
            model, x, edge_index, w, attribute=("edges",), steps=48
        )
        output_full = model.predict(x, edge_index, w)
        output_baseline = model.predict(x, edge_index, np.zeros_like(w))
        np.testing.assert_allclose(
            float(np.sum(result.edge_attribution)),
            output_full - output_baseline,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_deterministic_across_repeated_calls(self, planted_graph):
        """Two identical calls must return bitwise identical arrays."""
        model, x, edge_index, w, planted = planted_graph
        first = integrated_gradients(
            model, x, edge_index, w, attribute=("edges", "nodes"), steps=32
        )
        second = integrated_gradients(
            model, x, edge_index, w, attribute=("edges", "nodes"), steps=32
        )
        np.testing.assert_array_equal(
            first.edge_attribution, second.edge_attribution
        )
        np.testing.assert_array_equal(
            first.node_attribution, second.node_attribution
        )

    def test_joint_attribution_shapes_and_completeness(self, planted_graph):
        """Joint edge+node attribution is finite, shaped, and complete.

        Joint completeness is only approximate here because the tanh
        readout makes the node-side integrand non-constant; the
        tolerance is loose enough for the midpoint quadrature at 256
        steps yet tight enough to catch wiring mistakes.
        """
        model, x, edge_index, w, planted = planted_graph
        result = integrated_gradients(
            model, x, edge_index, w, attribute=("edges", "nodes"), steps=256
        )
        assert result.node_attribution.shape == x.shape
        assert result.edge_attribution.shape == w.shape
        assert np.all(np.isfinite(result.node_attribution))
        assert np.all(np.isfinite(result.edge_attribution))

        output_full = model.predict(x, edge_index, w)
        output_baseline = model.predict(
            np.zeros_like(x), edge_index, np.zeros_like(w)
        )
        total = float(
            np.sum(result.edge_attribution) + np.sum(result.node_attribution)
        )
        np.testing.assert_allclose(
            total, output_full - output_baseline, rtol=2e-3, atol=1e-6
        )

    def test_unrequested_modalities_are_none(self, planted_graph):
        """Only the requested modality arrays should be populated."""
        model, x, edge_index, w, planted = planted_graph
        edges_only = integrated_gradients(
            model, x, edge_index, w, attribute=("edges",)
        )
        assert edges_only.edge_attribution is not None
        assert edges_only.node_attribution is None
        nodes_only = integrated_gradients(
            model, x, edge_index, w, attribute=("nodes",)
        )
        assert nodes_only.node_attribution is not None
        assert nodes_only.edge_attribution is None


class TestGnnExplainerEdgeMask:
    """GNNExplainer-style edge-mask behavior on the planted-motif graph."""

    # Hyperparameters shared by the recovery tests: the sparsity weight
    # exceeds the largest possible background gate * weight product
    # (0.09 * 0.8 = 0.072), so every background logit is pushed down
    # while the planted logits (contribution 6.0 each) saturate upward.
    EXPLAINER_KWARGS = dict(
        steps=300,
        learning_rate=0.5,
        sparsity_weight=0.1,
        entropy_weight=0.01,
    )

    def test_recovers_planted_motif_as_top_mask(self, planted_graph):
        """The three planted chain edges must be the top-3 mask entries."""
        model, x, edge_index, w, planted = planted_graph
        result = gnn_explainer_edge_mask(
            model, x, edge_index, w, **self.EXPLAINER_KWARGS
        )
        order = np.argsort(-result.edge_mask, kind="stable")
        assert set(order[:3].tolist()) == set(planted)

    def test_planted_mask_saturates_background_suppressed(self, planted_graph):
        """Planted entries approach 1 while background stays below 1/2."""
        model, x, edge_index, w, planted = planted_graph
        result = gnn_explainer_edge_mask(
            model, x, edge_index, w, **self.EXPLAINER_KWARGS
        )
        background = np.ones_like(w, dtype=bool)
        background[planted] = False
        assert np.all(result.edge_mask[planted] > 0.9)
        assert np.all(result.edge_mask[background] < 0.5)

    def test_deterministic_given_seed(self, planted_graph):
        """Same seed must give bitwise identical mask and loss history."""
        model, x, edge_index, w, planted = planted_graph
        kwargs = dict(self.EXPLAINER_KWARGS, init_scale=0.05, seed=123)
        first = gnn_explainer_edge_mask(model, x, edge_index, w, **kwargs)
        second = gnn_explainer_edge_mask(model, x, edge_index, w, **kwargs)
        np.testing.assert_array_equal(first.edge_mask, second.edge_mask)
        np.testing.assert_array_equal(first.loss_history, second.loss_history)

    def test_different_seeds_keep_same_top_edges(self, planted_graph):
        """Init noise changes raw values but not the recovered motif."""
        model, x, edge_index, w, planted = planted_graph
        kwargs = dict(self.EXPLAINER_KWARGS, init_scale=0.05)
        first = gnn_explainer_edge_mask(model, x, edge_index, w, seed=1, **kwargs)
        second = gnn_explainer_edge_mask(model, x, edge_index, w, seed=2, **kwargs)
        assert not np.array_equal(first.edge_mask, second.edge_mask)
        top_first = np.argsort(-first.edge_mask, kind="stable")[:3]
        top_second = np.argsort(-second.edge_mask, kind="stable")[:3]
        assert set(top_first.tolist()) == set(planted)
        assert set(top_second.tolist()) == set(planted)

    def test_loss_history_recorded_and_decreasing(self, planted_graph):
        """Loss history has one finite entry per step and ends lower."""
        model, x, edge_index, w, planted = planted_graph
        result = gnn_explainer_edge_mask(
            model, x, edge_index, w, **self.EXPLAINER_KWARGS
        )
        assert result.loss_history.shape == (self.EXPLAINER_KWARGS["steps"],)
        assert np.all(np.isfinite(result.loss_history))
        assert result.loss_history[-1] < result.loss_history[0]


class TestAttributionResult:
    """Behavior of the AttributionResult helper itself."""

    def test_top_edges_breaks_ties_by_index(self):
        """Equal attributions must resolve to the lower edge index."""
        result = AttributionResult(
            edge_attribution=np.array([1.0, 2.0, 2.0, 0.5]),
            node_attribution=None,
        )
        np.testing.assert_array_equal(result.top_edges(2), np.array([1, 2]))
        np.testing.assert_array_equal(result.top_edges(1), np.array([1]))

    def test_top_edges_requires_edge_attribution(self):
        """Requesting top edges from a nodes-only result must fail."""
        result = AttributionResult(
            edge_attribution=None,
            node_attribution=np.zeros((3, 2)),
        )
        with pytest.raises(ValueError, match="no edge attribution"):
            result.top_edges(1)

    @pytest.mark.parametrize("k", [0, -1, 4])
    def test_top_edges_k_out_of_bounds(self, k):
        """k outside [1, n_edges] must be rejected."""
        result = AttributionResult(
            edge_attribution=np.array([0.1, 0.2, 0.3]),
            node_attribution=None,
        )
        with pytest.raises(ValueError, match="k must be in"):
            result.top_edges(k)


class TestInputValidation:
    """Shared input validation across both backends."""

    def test_abstract_interface_cannot_be_instantiated(self):
        """The model interface must stay abstract."""
        with pytest.raises(TypeError):
            DifferentiableGraphModel()

    def test_non_finite_node_features_rejected(self, planted_graph):
        model, x, edge_index, w, _ = planted_graph
        bad = x.copy()
        bad[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            integrated_gradients(model, bad, edge_index, w)

    def test_non_finite_edge_weight_rejected(self, planted_graph):
        model, x, edge_index, w, _ = planted_graph
        bad = w.copy()
        bad[0] = np.inf
        with pytest.raises(ValueError, match="finite"):
            gnn_explainer_edge_mask(model, x, edge_index, bad)

    def test_misshapen_edge_index_rejected(self, planted_graph):
        model, x, edge_index, w, _ = planted_graph
        bad = np.vstack([edge_index, edge_index[0]])
        with pytest.raises(ValueError, match="shape"):
            integrated_gradients(model, x, bad, w)

    def test_float_edge_index_rejected(self, planted_graph):
        """Float indices must fail even when integral-valued."""
        model, x, edge_index, w, _ = planted_graph
        with pytest.raises(ValueError, match="integer"):
            integrated_gradients(
                model, x, edge_index.astype(np.float64), w
            )

    def test_out_of_range_edge_index_rejected(self, planted_graph):
        model, x, edge_index, w, _ = planted_graph
        bad = edge_index.copy()
        bad[0, 0] = x.shape[0]  # one past the last valid node index
        with pytest.raises(ValueError, match="lie in"):
            integrated_gradients(model, x, bad, w)

    def test_edge_weight_length_mismatch_rejected(self, planted_graph):
        model, x, edge_index, w, _ = planted_graph
        with pytest.raises(ValueError, match="length"):
            integrated_gradients(model, x, edge_index, w[:-1])

    @pytest.mark.parametrize("steps", [0, -3])
    def test_non_positive_steps_rejected(self, planted_graph, steps):
        model, x, edge_index, w, _ = planted_graph
        with pytest.raises(ValueError, match="steps"):
            integrated_gradients(model, x, edge_index, w, steps=steps)
        with pytest.raises(ValueError, match="steps"):
            gnn_explainer_edge_mask(model, x, edge_index, w, steps=steps)

    def test_unknown_attribution_target_rejected(self, planted_graph):
        model, x, edge_index, w, _ = planted_graph
        with pytest.raises(ValueError, match="attribute"):
            integrated_gradients(
                model, x, edge_index, w, attribute=("edges", "weights")
            )

    def test_empty_attribution_target_rejected(self, planted_graph):
        model, x, edge_index, w, _ = planted_graph
        with pytest.raises(ValueError, match="attribute"):
            integrated_gradients(model, x, edge_index, w, attribute=())

    def test_misshapen_baseline_rejected(self, planted_graph):
        model, x, edge_index, w, _ = planted_graph
        with pytest.raises(ValueError, match="node_baseline"):
            integrated_gradients(
                model, x, edge_index, w,
                node_baseline=np.zeros((x.shape[0] + 1, x.shape[1])),
            )
        with pytest.raises(ValueError, match="edge_baseline"):
            integrated_gradients(
                model, x, edge_index, w,
                edge_baseline=np.zeros(w.shape[0] + 1),
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"learning_rate": 0.0},
            {"learning_rate": -0.5},
            {"sparsity_weight": -0.1},
            {"entropy_weight": -0.1},
            {"init_scale": -0.1},
            {"init_logit": np.nan},
        ],
    )
    def test_bad_explainer_hyperparameters_rejected(self, planted_graph, kwargs):
        model, x, edge_index, w, _ = planted_graph
        with pytest.raises(ValueError):
            gnn_explainer_edge_mask(model, x, edge_index, w, **kwargs)


class _VectorOutputModel(DifferentiableGraphModel):
    """Broken adapter stub: predict returns a vector, not a scalar."""

    def predict(self, node_features, edge_index, edge_weight):
        return np.zeros(3)  # contract violation on purpose

    def gradients(self, node_features, edge_index, edge_weight):
        return (
            np.zeros_like(node_features, dtype=np.float64),
            np.zeros_like(edge_weight, dtype=np.float64),
        )


class _MisshapenGradientModel(DifferentiableGraphModel):
    """Broken adapter stub: gradients have the wrong edge length."""

    def predict(self, node_features, edge_index, edge_weight):
        return 0.0

    def gradients(self, node_features, edge_index, edge_weight):
        return (
            np.zeros_like(node_features, dtype=np.float64),
            np.zeros(edge_weight.shape[0] + 1),  # contract violation
        )


class TestModelContractEnforcement:
    """Backends must reject model adapters that violate the interface."""

    def test_vector_output_rejected(self, planted_graph):
        _, x, edge_index, w, _ = planted_graph
        with pytest.raises(ValueError, match="scalar"):
            gnn_explainer_edge_mask(
                _VectorOutputModel(), x, edge_index, w, steps=2
            )

    def test_misshapen_gradient_rejected(self, planted_graph):
        _, x, edge_index, w, _ = planted_graph
        with pytest.raises(ValueError, match="gradient"):
            integrated_gradients(
                _MisshapenGradientModel(), x, edge_index, w, steps=2
            )


class TestEmptyEdgeGraph:
    """Degenerate graphs with zero edges must not crash the backends."""

    def test_integrated_gradients_with_no_edges(self, planted_graph):
        model, x, _, _, _ = planted_graph
        empty_index = np.zeros((2, 0), dtype=np.int64)
        empty_weight = np.zeros(0)
        result = integrated_gradients(
            model, x, empty_index, empty_weight, attribute=("edges",)
        )
        assert result.edge_attribution.shape == (0,)
        with pytest.raises(ValueError, match="k must be in"):
            result.top_edges(1)

    def test_explainer_with_no_edges(self, planted_graph):
        model, x, _, _, _ = planted_graph
        empty_index = np.zeros((2, 0), dtype=np.int64)
        empty_weight = np.zeros(0)
        result = gnn_explainer_edge_mask(
            model, x, empty_index, empty_weight, steps=3
        )
        assert isinstance(result, ExplainerResult)
        assert result.edge_mask.shape == (0,)
        assert result.loss_history.shape == (3,)
        assert np.all(np.isfinite(result.loss_history))
