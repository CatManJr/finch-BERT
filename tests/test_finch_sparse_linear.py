import numpy as np
import pytest

from finchbert.application import SparseLinearPort
from finchbert.infrastructure import FinchSparseLinear

_ORACLE_ABSOLUTE_TOLERANCE = 1e-4


def test_transform_matches_dense_numpy_oracle_with_bias() -> None:
    weight = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    bias = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    activations = (
        (1.0, 2.0, 3.0),
        (0.5, 0.0, 1.0),
    )
    linear: SparseLinearPort = FinchSparseLinear.from_dense_weight(weight, bias)
    result = linear.transform(activations)
    expected = np.asarray(activations, dtype=np.float32) @ weight.T + bias
    np.testing.assert_allclose(
        np.asarray(result, dtype=np.float32),
        expected,
        atol=_ORACLE_ABSOLUTE_TOLERANCE,
    )


def test_transform_matches_dense_numpy_oracle_without_bias() -> None:
    weight = np.array(
        [
            [0.0, 0.0],
            [1.5, 0.0],
            [0.0, -2.0],
        ],
        dtype=np.float32,
    )
    activations = ((2.0, 4.0),)
    linear = FinchSparseLinear.from_dense_weight(weight)
    result = linear.transform(activations)
    expected = np.asarray(activations, dtype=np.float32) @ weight.T
    np.testing.assert_allclose(
        np.asarray(result, dtype=np.float32),
        expected,
        atol=_ORACLE_ABSOLUTE_TOLERANCE,
    )


def test_transform_matches_oracle_on_unstructured_sparsity() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((16, 8)).astype(np.float32)
    weight[rng.random(weight.shape) < 0.9] = 0.0
    bias = rng.standard_normal(16).astype(np.float32)
    activations = rng.standard_normal((5, 8)).astype(np.float32)
    linear = FinchSparseLinear.from_dense_weight(weight, bias)
    result = linear.transform(_as_rows(activations))
    expected = activations @ weight.T + bias
    np.testing.assert_allclose(
        np.asarray(result, dtype=np.float32),
        expected,
        atol=_ORACLE_ABSOLUTE_TOLERANCE,
    )


def test_all_zero_weight_returns_bias_or_zeros() -> None:
    weight = np.zeros((3, 2), dtype=np.float32)
    activations = ((1.0, 1.0), (2.0, 3.0), (0.0, 4.0), (-1.0, 0.5))
    linear = FinchSparseLinear.from_dense_weight(weight)
    assert linear.transform(activations) == (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )


def test_transform_matrix_matches_tuple_transform() -> None:
    weight = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    bias = np.array([0.25, -0.5], dtype=np.float32)
    activations = np.array([[1.0, 2.0, 3.0], [0.5, 0.0, 1.0]], dtype=np.float32)
    linear = FinchSparseLinear.from_dense_weight(weight, bias)
    from_tuples = np.asarray(linear.transform(_as_rows(activations)), dtype=np.float32)
    from_matrix = linear.transform_matrix(activations)
    np.testing.assert_allclose(
        from_matrix, from_tuples, atol=_ORACLE_ABSOLUTE_TOLERANCE
    )


def test_empty_token_batch_returns_no_rows() -> None:
    weight = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    linear = FinchSparseLinear.from_dense_weight(weight)
    assert linear.transform(()) == ()


def test_rejects_activation_width_mismatch() -> None:
    weight = np.array([[1.0, 0.0, 2.0]], dtype=np.float32)
    linear = FinchSparseLinear.from_dense_weight(weight)
    with pytest.raises(ValueError, match="features"):
        linear.transform(((1.0, 2.0),))


def test_rejects_bias_length_mismatch() -> None:
    weight = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="bias length"):
        FinchSparseLinear.from_dense_weight(weight, bias=[0.1])


def test_rejects_one_dimensional_weight() -> None:
    with pytest.raises(ValueError, match="2-D"):
        FinchSparseLinear.from_dense_weight([1.0, 2.0, 3.0])


def _as_rows(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)
