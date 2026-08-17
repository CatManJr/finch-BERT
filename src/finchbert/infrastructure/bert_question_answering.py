"""BERT encoder and SQuAD span head. Pruned Linear maps run as Finch CSR SpMM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from finchbert.domain import BertConfig, TokenBatch
from finchbert.infrastructure.dense_ops import (
    additive_attention_bias,
    gelu,
    layer_norm,
    softmax,
)
from finchbert.infrastructure.finch_sparse_linear import FinchSparseLinear
from finchbert.infrastructure.huggingface_weight_catalog import (
    HuggingFaceWeightCatalog,
)

_ACTIVATION_DTYPE = np.float32
_SPAN_LABEL_COUNT = 2
_WORD_EMBEDDINGS = "bert.embeddings.word_embeddings.weight"
_POSITION_EMBEDDINGS = "bert.embeddings.position_embeddings.weight"
_TOKEN_TYPE_EMBEDDINGS = "bert.embeddings.token_type_embeddings.weight"
_EMBEDDING_NORM = "bert.embeddings.LayerNorm"
_QA_OUTPUTS = "qa_outputs"


@dataclass(frozen=True, slots=True)
class _LayerNorm:
    weight: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True, slots=True)
class _Embeddings:
    word: np.ndarray
    position: np.ndarray
    token_type: np.ndarray
    layer_norm: _LayerNorm


@dataclass(frozen=True, slots=True)
class _EncoderBlock:
    query: FinchSparseLinear
    key: FinchSparseLinear
    value: FinchSparseLinear
    attention_output: FinchSparseLinear
    attention_norm: _LayerNorm
    intermediate: FinchSparseLinear
    feed_forward_output: FinchSparseLinear
    output_norm: _LayerNorm


class BertQuestionAnsweringModel:
    def __init__(
        self,
        config: BertConfig,
        embeddings: _Embeddings,
        blocks: tuple[_EncoderBlock, ...],
        qa_outputs: FinchSparseLinear,
    ) -> None:
        if len(blocks) != config.hidden_layer_count:
            raise ValueError("encoder block count must match hidden_layer_count.")
        self._config = config
        self._embeddings = embeddings
        self._blocks = blocks
        self._qa_outputs = qa_outputs
        _validate_embeddings(config, embeddings)
        for block in blocks:
            _validate_block(config, block)
        _validate_span_head(config, qa_outputs)

    @classmethod
    def from_catalog(
        cls,
        config: BertConfig,
        catalog: HuggingFaceWeightCatalog,
    ) -> BertQuestionAnsweringModel:
        embeddings = _embeddings_from_catalog(config, catalog)
        blocks = tuple(
            _block_from_catalog(config, catalog, layer_index)
            for layer_index in range(config.hidden_layer_count)
        )
        qa_outputs = catalog.sparse_linear(_QA_OUTPUTS)
        return cls(config, embeddings, blocks, qa_outputs)

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path | None = None,
        config: BertConfig | None = None,
    ) -> BertQuestionAnsweringModel:
        catalog = HuggingFaceWeightCatalog.from_pretrained(source)
        return cls.from_catalog(config or BertConfig.bert_base(), catalog)

    def span_logits(
        self, tokens: TokenBatch
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        hidden = self._encode(tokens)
        logits = self._qa_outputs.transform_matrix(hidden)
        if logits.shape != (tokens.length, _SPAN_LABEL_COUNT):
            raise RuntimeError("SQuAD head produced an unexpected shape.")
        start = tuple(float(value) for value in logits[:, 0])
        end = tuple(float(value) for value in logits[:, 1])
        return start, end

    def _encode(self, tokens: TokenBatch) -> np.ndarray:
        if tokens.length == 0:
            raise ValueError("token batch must be non-empty.")
        if tokens.length > self._config.max_sequence_length:
            raise ValueError("token batch exceeds max_sequence_length.")
        input_ids = np.asarray(tokens.input_ids, dtype=np.intp)
        token_type_ids = np.asarray(tokens.token_type_ids, dtype=np.intp)
        attention_mask = np.asarray(tokens.attention_mask, dtype=_ACTIVATION_DTYPE)
        _require_id_range("input_ids", input_ids, self._config.vocab_size)
        _require_id_range(
            "token_type_ids", token_type_ids, self._config.type_vocab_size
        )
        hidden = _embed(self._embeddings, input_ids, token_type_ids)
        for block in self._blocks:
            hidden = _apply_block(block, hidden, attention_mask, self._config)
        return hidden


def _embeddings_from_catalog(
    config: BertConfig,
    catalog: HuggingFaceWeightCatalog,
) -> _Embeddings:
    return _Embeddings(
        word=_require_matrix(
            catalog.numpy_tensor(_WORD_EMBEDDINGS),
            config.vocab_size,
            config.hidden_size,
            _WORD_EMBEDDINGS,
        ),
        position=_require_matrix(
            catalog.numpy_tensor(_POSITION_EMBEDDINGS),
            config.max_position_count,
            config.hidden_size,
            _POSITION_EMBEDDINGS,
        ),
        token_type=_require_matrix(
            catalog.numpy_tensor(_TOKEN_TYPE_EMBEDDINGS),
            config.type_vocab_size,
            config.hidden_size,
            _TOKEN_TYPE_EMBEDDINGS,
        ),
        layer_norm=_layer_norm_from_catalog(
            catalog, _EMBEDDING_NORM, config.hidden_size
        ),
    )


def _block_from_catalog(
    config: BertConfig,
    catalog: HuggingFaceWeightCatalog,
    layer_index: int,
) -> _EncoderBlock:
    prefix = f"bert.encoder.layer.{layer_index}"
    return _EncoderBlock(
        query=catalog.sparse_linear(f"{prefix}.attention.self.query"),
        key=catalog.sparse_linear(f"{prefix}.attention.self.key"),
        value=catalog.sparse_linear(f"{prefix}.attention.self.value"),
        attention_output=catalog.sparse_linear(f"{prefix}.attention.output.dense"),
        attention_norm=_layer_norm_from_catalog(
            catalog,
            f"{prefix}.attention.output.LayerNorm",
            config.hidden_size,
        ),
        intermediate=catalog.sparse_linear(f"{prefix}.intermediate.dense"),
        feed_forward_output=catalog.sparse_linear(f"{prefix}.output.dense"),
        output_norm=_layer_norm_from_catalog(
            catalog,
            f"{prefix}.output.LayerNorm",
            config.hidden_size,
        ),
    )


def _layer_norm_from_catalog(
    catalog: HuggingFaceWeightCatalog,
    stem: str,
    hidden_size: int,
) -> _LayerNorm:
    return _LayerNorm(
        weight=_require_vector(
            catalog.numpy_tensor(f"{stem}.weight"), hidden_size, stem
        ),
        bias=_require_vector(catalog.numpy_tensor(f"{stem}.bias"), hidden_size, stem),
    )


def _embed(
    embeddings: _Embeddings,
    input_ids: np.ndarray,
    token_type_ids: np.ndarray,
) -> np.ndarray:
    sequence_length = int(input_ids.shape[0])
    positions = np.arange(sequence_length, dtype=np.intp)
    hidden = (
        embeddings.word[input_ids]
        + embeddings.position[positions]
        + embeddings.token_type[token_type_ids]
    )
    return layer_norm(hidden, embeddings.layer_norm.weight, embeddings.layer_norm.bias)


def _apply_block(
    block: _EncoderBlock,
    hidden: np.ndarray,
    attention_mask: np.ndarray,
    config: BertConfig,
) -> np.ndarray:
    attention = _self_attention(block, hidden, attention_mask, config)
    hidden = layer_norm(
        attention + hidden,
        block.attention_norm.weight,
        block.attention_norm.bias,
    )
    intermediate = gelu(block.intermediate.transform_matrix(hidden))
    feed_forward = block.feed_forward_output.transform_matrix(intermediate)
    return layer_norm(
        feed_forward + hidden,
        block.output_norm.weight,
        block.output_norm.bias,
    )


def _self_attention(
    block: _EncoderBlock,
    hidden: np.ndarray,
    attention_mask: np.ndarray,
    config: BertConfig,
) -> np.ndarray:
    sequence_length = hidden.shape[0]
    head_count = config.attention_head_count
    head_size = config.attention_head_size
    query = _split_heads(
        block.query.transform_matrix(hidden),
        sequence_length,
        head_count,
        head_size,
    )
    key = _split_heads(
        block.key.transform_matrix(hidden),
        sequence_length,
        head_count,
        head_size,
    )
    value = _split_heads(
        block.value.transform_matrix(hidden),
        sequence_length,
        head_count,
        head_size,
    )
    scale = _ACTIVATION_DTYPE(head_size**-0.5)
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
    scores = scores + additive_attention_bias(attention_mask)
    weights = softmax(scores)
    context = np.matmul(weights, value)
    merged = np.swapaxes(context, 0, 1).reshape(sequence_length, head_count * head_size)
    return block.attention_output.transform_matrix(
        np.ascontiguousarray(merged, dtype=_ACTIVATION_DTYPE)
    )


def _split_heads(
    projected: np.ndarray,
    sequence_length: int,
    head_count: int,
    head_size: int,
) -> np.ndarray:
    expected = (sequence_length, head_count * head_size)
    if projected.shape != expected:
        raise RuntimeError("attention projection has an unexpected shape.")
    return projected.reshape(sequence_length, head_count, head_size).transpose(1, 0, 2)


def _validate_embeddings(config: BertConfig, embeddings: _Embeddings) -> None:
    _require_matrix(
        embeddings.word,
        config.vocab_size,
        config.hidden_size,
        _WORD_EMBEDDINGS,
    )
    _require_matrix(
        embeddings.position,
        config.max_position_count,
        config.hidden_size,
        _POSITION_EMBEDDINGS,
    )
    _require_matrix(
        embeddings.token_type,
        config.type_vocab_size,
        config.hidden_size,
        _TOKEN_TYPE_EMBEDDINGS,
    )
    _require_vector(
        embeddings.layer_norm.weight,
        config.hidden_size,
        _EMBEDDING_NORM,
    )
    _require_vector(embeddings.layer_norm.bias, config.hidden_size, _EMBEDDING_NORM)


def _validate_block(config: BertConfig, block: _EncoderBlock) -> None:
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    _require_linear(block.query, hidden, hidden, "query")
    _require_linear(block.key, hidden, hidden, "key")
    _require_linear(block.value, hidden, hidden, "value")
    _require_linear(block.attention_output, hidden, hidden, "attention output")
    _require_linear(block.intermediate, hidden, intermediate, "intermediate")
    _require_linear(
        block.feed_forward_output, intermediate, hidden, "feed-forward output"
    )
    _require_vector(block.attention_norm.weight, hidden, "attention LayerNorm")
    _require_vector(block.attention_norm.bias, hidden, "attention LayerNorm")
    _require_vector(block.output_norm.weight, hidden, "output LayerNorm")
    _require_vector(block.output_norm.bias, hidden, "output LayerNorm")


def _validate_span_head(config: BertConfig, qa_outputs: FinchSparseLinear) -> None:
    _require_linear(qa_outputs, config.hidden_size, _SPAN_LABEL_COUNT, _QA_OUTPUTS)


def _require_linear(
    linear: FinchSparseLinear,
    in_features: int,
    out_features: int,
    name: str,
) -> None:
    if linear.in_features != in_features or linear.out_features != out_features:
        raise ValueError(f"{name} weight shape must be {(out_features, in_features)}.")


def _require_matrix(
    tensor: np.ndarray,
    rows: int,
    columns: int,
    name: str,
) -> np.ndarray:
    if tensor.shape != (rows, columns):
        raise ValueError(f"{name} must have shape {(rows, columns)}.")
    return tensor


def _require_vector(tensor: np.ndarray, length: int, name: str) -> np.ndarray:
    if tensor.shape != (length,):
        raise ValueError(f"{name} must have shape {(length,)}.")
    return tensor


def _require_id_range(name: str, ids: np.ndarray, limit: int) -> None:
    if np.any(ids < 0) or np.any(ids >= limit):
        raise ValueError(f"{name} values must lie in [0, {limit}).")
