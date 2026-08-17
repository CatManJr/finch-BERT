"""Pack a dense matrix with zeros into a Finch CSR FiberTensor."""

from __future__ import annotations

import numpy as np
from finch import DenseLevel, ElementLevel, FiberTensor, SparseListLevel, element
from finch.codegen import NumpyBuffer

_WEIGHT_DTYPE = np.float32
_CSR_FILL_VALUE = _WEIGHT_DTYPE(0.0)


def fiber_tensor_from_dense_matrix(weight: np.ndarray) -> FiberTensor:
    matrix = np.ascontiguousarray(weight, dtype=_WEIGHT_DTYPE)
    if matrix.ndim != 2:
        raise ValueError("weight must be a 2-D matrix.")
    row_count, column_count = matrix.shape
    nonzero_rows, nonzero_cols = np.nonzero(matrix)
    values = np.ascontiguousarray(
        matrix[nonzero_rows, nonzero_cols],
        dtype=_WEIGHT_DTYPE,
    )
    columns = np.ascontiguousarray(nonzero_cols, dtype=np.intp)
    row_counts = np.bincount(nonzero_rows, minlength=row_count)
    row_pointers = np.zeros(row_count + 1, dtype=np.intp)
    np.cumsum(row_counts, out=row_pointers[1:])
    values_level = ElementLevel(
        element(_CSR_FILL_VALUE),  # type: ignore[no-untyped-call]
        NumpyBuffer(values),
    )
    column_level = SparseListLevel(
        values_level,
        np.intp(column_count),
        NumpyBuffer(row_pointers),
        NumpyBuffer(columns),
    )
    return FiberTensor(DenseLevel(column_level, np.intp(row_count)))
