import math

import numpy as np

from finchbert.infrastructure.dense_ops import gelu, layer_norm, softmax

_ORACLE_ABSOLUTE_TOLERANCE = 1e-6


def test_gelu_matches_erf_formula() -> None:
    values = np.array([-2.0, -0.5, 0.0, 0.3, 1.7], dtype=np.float32)
    expected = np.array(
        [0.5 * value * (1.0 + math.erf(value / math.sqrt(2.0))) for value in values],
        dtype=np.float32,
    )
    np.testing.assert_allclose(gelu(values), expected, atol=_ORACLE_ABSOLUTE_TOLERANCE)


def test_layer_norm_matches_feature_wise_formula() -> None:
    activations = np.array([[1.0, 3.0, 5.0], [2.0, 2.0, 8.0]], dtype=np.float32)
    weight = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.0], dtype=np.float32)
    mean = activations.mean(axis=-1, keepdims=True)
    variance = np.square(activations - mean).mean(axis=-1, keepdims=True)
    expected = weight * (activations - mean) / np.sqrt(variance + 1e-12) + bias
    np.testing.assert_allclose(
        layer_norm(activations, weight, bias),
        expected,
        atol=_ORACLE_ABSOLUTE_TOLERANCE,
    )


def test_softmax_rows_sum_to_one() -> None:
    scores = np.array([[1.0, 2.0, 3.0], [0.0, -1.0, 4.0]], dtype=np.float32)
    weights = softmax(scores)
    np.testing.assert_allclose(weights.sum(axis=-1), np.ones(2), atol=1e-6)
    assert weights[0, 2] > weights[0, 1] > weights[0, 0]
