"""Finch CSR SpMM adapter for pruned linear maps."""

from __future__ import annotations

# Finch's public functions are untyped; this adapter is the typed boundary.
# mypy: disable-error-code="no-untyped-call"
import numpy as np
from finch import (
    COMPILE_NUMBA,
    DenseLevel,
    ElementLevel,
    FiberTensor,
    SparseListLevel,
    add,
    asarray,
    einsum,
    element,
    to_numpy,
    with_default_scheduler,
)
from finch.codegen import NumpyBuffer
from numpy.typing import ArrayLike

# \(Y = (W X^{\top})^{\top}\)
_SPMM_EINSUM_SUBSCRIPTS = "ik,jk->ij"
_WEIGHT_DTYPE = np.float32
_CSR_FILL_VALUE = _WEIGHT_DTYPE(0.0)


class FinchSparseLinear:
    def __init__(
        self,
        weight: FiberTensor,
        in_features: int,
        out_features: int,
        bias: np.ndarray | None = None,
    ) -> None:
        self._weight = weight
        self._in_features = in_features
        self._out_features = out_features
        self._bias = (
            None if bias is None else np.array(bias, dtype=_WEIGHT_DTYPE, copy=True)
        )

    @classmethod
    def from_dense_weight(
        cls,
        weight: ArrayLike,
        bias: ArrayLike | None = None,
    ) -> FinchSparseLinear:
        weight_array = np.asarray(weight, dtype=_WEIGHT_DTYPE)
        _require_weight_matrix(weight_array)
        bias_array = None if bias is None else np.asarray(bias, dtype=_WEIGHT_DTYPE)
        if bias_array is not None:
            _require_bias_length(bias_array, int(weight_array.shape[0]))
        return cls(
            weight=_fiber_tensor_from_csr_matrix(weight_array),
            in_features=int(weight_array.shape[1]),
            out_features=int(weight_array.shape[0]),
            bias=bias_array,
        )

    def transform(
        self, activations: tuple[tuple[float, ...], ...]
    ) -> tuple[tuple[float, ...], ...]:
        activation_matrix = _activation_matrix(activations, self._in_features)
        try:
            result = _finch_spmm(self._weight, activation_matrix, self._bias)
        except Exception as exc:
            raise RuntimeError("Finch CSR SpMM failed.") from exc
        if result.shape != (activation_matrix.shape[0], self._out_features):
            raise RuntimeError("Finch CSR SpMM failed.")
        return _as_rows(result)


def _require_weight_matrix(weight: np.ndarray) -> None:
    if weight.ndim != 2:
        raise ValueError("weight must be a 2-D matrix.")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise ValueError("weight dimensions must be positive.")


def _require_bias_length(bias: np.ndarray, out_features: int) -> None:
    if bias.ndim != 1:
        raise ValueError("bias must be a 1-D vector.")
    if bias.shape[0] != out_features:
        raise ValueError("bias length must match the number of output features.")


def _activation_matrix(
    activations: tuple[tuple[float, ...], ...],
    in_features: int,
) -> np.ndarray:
    if not activations:
        return np.zeros((0, in_features), dtype=_WEIGHT_DTYPE)
    matrix = np.asarray(activations, dtype=_WEIGHT_DTYPE)
    if matrix.ndim != 2:
        raise ValueError("activations must be a 2-D matrix.")
    if matrix.shape[1] != in_features:
        raise ValueError(f"each activation row must have {in_features} features.")
    return np.ascontiguousarray(matrix)


def _fiber_tensor_from_csr_matrix(weight: np.ndarray) -> FiberTensor:
    row_count, column_count = weight.shape
    nonzero_rows, nonzero_cols = np.nonzero(weight)
    values = np.ascontiguousarray(
        weight[nonzero_rows, nonzero_cols],
        dtype=_WEIGHT_DTYPE,
    )
    columns = np.ascontiguousarray(nonzero_cols, dtype=np.intp)
    row_counts = np.bincount(nonzero_rows, minlength=row_count)
    row_pointers = np.zeros(row_count + 1, dtype=np.intp)
    np.cumsum(row_counts, out=row_pointers[1:])
    values_level = ElementLevel(element(_CSR_FILL_VALUE), NumpyBuffer(values))
    column_level = SparseListLevel(
        values_level,
        np.intp(column_count),
        NumpyBuffer(row_pointers),
        NumpyBuffer(columns),
    )
    return FiberTensor(DenseLevel(column_level, np.intp(row_count)))


def _finch_spmm(
    weight: FiberTensor,
    activations: np.ndarray,
    bias: np.ndarray | None,
) -> np.ndarray:
    # FiberTensor is not executable on the notation interpreter.
    with with_default_scheduler(COMPILE_NUMBA):
        product = einsum(_SPMM_EINSUM_SUBSCRIPTS, asarray(activations), weight)
        output = product if bias is None else add(product, asarray(bias))
        return np.asarray(to_numpy(output), dtype=_WEIGHT_DTYPE)


def _as_rows(matrix: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)
