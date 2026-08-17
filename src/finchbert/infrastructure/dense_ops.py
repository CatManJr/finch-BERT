"""Dense NumPy ops used beside Finch SpMM: LayerNorm, GELU, and attention softmax."""

from __future__ import annotations

import math

import numpy as np

_ACTIVATION_DTYPE = np.float32
_LAYER_NORM_EPSILON = 1e-12
# Original BERT additive mask; a true -inf pad bias can NaN softmax on empty rows.
_PADDED_ATTENTION_BIAS = _ACTIVATION_DTYPE(-10000.0)
_MATH_ERROR_FUNCTION = np.vectorize(math.erf, otypes=[np.float64])


def gelu(activations: np.ndarray) -> np.ndarray:
    values = np.asarray(activations, dtype=_ACTIVATION_DTYPE)
    scale = np.sqrt(_ACTIVATION_DTYPE(2.0))
    return np.asarray(
        _ACTIVATION_DTYPE(0.5)
        * values
        * (_ACTIVATION_DTYPE(1.0) + _error_function(values / scale)),
        dtype=_ACTIVATION_DTYPE,
    )


def layer_norm(
    activations: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    values = np.asarray(activations, dtype=_ACTIVATION_DTYPE)
    mean = values.mean(axis=-1, keepdims=True)
    variance = np.square(values - mean).mean(axis=-1, keepdims=True)
    normalized = (values - mean) / np.sqrt(variance + _LAYER_NORM_EPSILON)
    return np.asarray(weight * normalized + bias, dtype=_ACTIVATION_DTYPE)


def softmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=_ACTIVATION_DTYPE)
    shifted = values - values.max(axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return np.asarray(
        exponential / exponential.sum(axis=-1, keepdims=True),
        dtype=_ACTIVATION_DTYPE,
    )


def additive_attention_bias(attention_mask: np.ndarray) -> np.ndarray:
    keep = np.asarray(attention_mask, dtype=_ACTIVATION_DTYPE)
    return np.asarray(
        (_ACTIVATION_DTYPE(1.0) - keep) * _PADDED_ATTENTION_BIAS,
        dtype=_ACTIVATION_DTYPE,
    )


def _error_function(values: np.ndarray) -> np.ndarray:
    return np.asarray(_MATH_ERROR_FUNCTION(values), dtype=_ACTIVATION_DTYPE)
