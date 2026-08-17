"""BERT encoder and SQuAD head: Finch Linear with a dense NumPy oracle."""

from __future__ import annotations

import numpy as np
import pytest

from finchbert.application import AnswerQuestion, QuestionAnsweringModelPort
from finchbert.domain import BertConfig, TokenBatch, highest_scoring_span
from finchbert.infrastructure import (
    BertQuestionAnsweringModel,
    HuggingFaceWeightCatalog,
)
from finchbert.infrastructure.dense_ops import (
    additive_attention_bias,
    gelu,
    layer_norm,
    softmax,
)

_ORACLE_ABSOLUTE_TOLERANCE = 1e-4


def test_span_logits_match_dense_numpy_oracle() -> None:
    config = _tiny_config()
    tensors = _tiny_tensors(config)
    tokens = _tokens()
    model: QuestionAnsweringModelPort = BertQuestionAnsweringModel.from_catalog(
        config,
        HuggingFaceWeightCatalog.from_tensors(tensors),
    )
    start, end = model.span_logits(tokens)
    expected_start, expected_end = _numpy_span_logits(config, tensors, tokens)
    np.testing.assert_allclose(start, expected_start, atol=_ORACLE_ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(end, expected_end, atol=_ORACLE_ABSOLUTE_TOLERANCE)


def test_padding_mask_changes_span_logits() -> None:
    config = _tiny_config()
    tensors = _tiny_tensors(config)
    model = BertQuestionAnsweringModel.from_catalog(
        config,
        HuggingFaceWeightCatalog.from_tensors(tensors),
    )
    full = model.span_logits(_tokens(attention_mask=(1, 1, 1, 1)))
    padded = model.span_logits(_tokens(attention_mask=(1, 1, 1, 0)))
    assert full != padded


def test_answer_question_runs_encoder_for_span_text() -> None:
    config = _tiny_config()
    tensors = _tiny_tensors(config)
    tokens = _tokens()
    model = BertQuestionAnsweringModel.from_catalog(
        config,
        HuggingFaceWeightCatalog.from_tensors(tensors),
    )
    start, end = model.span_logits(tokens)
    use_case = AnswerQuestion(_RecordingTokenizer(tokens), model)
    span = use_case.answer("How many?", "There are two cats.")
    expected_start, expected_end = _argmax_span(start, end, tokens)
    assert span.start_token == expected_start
    assert span.end_token == expected_end
    assert span.text == "two"


def test_rejects_empty_token_batch() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="non-empty"):
        model.span_logits(TokenBatch((), (), ()))


def test_rejects_sequence_longer_than_config() -> None:
    model = _tiny_model()
    too_long = TokenBatch(
        input_ids=tuple(range(1, 10)),
        token_type_ids=(0,) * 9,
        attention_mask=(1,) * 9,
    )
    with pytest.raises(ValueError, match="max_sequence_length"):
        model.span_logits(too_long)


def test_rejects_input_id_outside_vocab() -> None:
    model = _tiny_model()
    with pytest.raises(ValueError, match="input_ids"):
        model.span_logits(
            TokenBatch((1, 99, 2, 3), (0, 0, 1, 1), (1, 1, 1, 1)),
        )


class _RecordingTokenizer:
    def __init__(self, tokens: TokenBatch) -> None:
        self._tokens = tokens

    def encode(self, question: str, context: str) -> TokenBatch:
        del question, context
        return self._tokens

    def decode_span(self, tokens: TokenBatch, start_token: int, end_token: int) -> str:
        del tokens, start_token, end_token
        return "two"


def _tiny_model() -> BertQuestionAnsweringModel:
    config = _tiny_config()
    return BertQuestionAnsweringModel.from_catalog(
        config,
        HuggingFaceWeightCatalog.from_tensors(_tiny_tensors(config)),
    )


def _tiny_config() -> BertConfig:
    return BertConfig(
        hidden_size=8,
        hidden_layer_count=1,
        attention_head_count=2,
        intermediate_size=16,
        vocab_size=12,
        max_position_count=8,
        type_vocab_size=2,
        max_sequence_length=8,
    )


def _tokens(*, attention_mask: tuple[int, ...] = (1, 1, 1, 1)) -> TokenBatch:
    return TokenBatch((1, 2, 3, 4), (0, 0, 1, 1), attention_mask)


def _tiny_tensors(config: BertConfig, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden = config.hidden_size
    tensors: dict[str, np.ndarray] = {
        "bert.embeddings.word_embeddings.weight": _table(
            rng, config.vocab_size, hidden
        ),
        "bert.embeddings.position_embeddings.weight": _table(
            rng, config.max_position_count, hidden
        ),
        "bert.embeddings.token_type_embeddings.weight": _table(
            rng, config.type_vocab_size, hidden
        ),
        "bert.embeddings.LayerNorm.weight": rng.standard_normal(hidden).astype(
            np.float32
        ),
        "bert.embeddings.LayerNorm.bias": rng.standard_normal(hidden).astype(np.float32)
        * np.float32(0.1),
    }
    for layer_index in range(config.hidden_layer_count):
        prefix = f"bert.encoder.layer.{layer_index}"
        tensors.update(_linear(rng, f"{prefix}.attention.self.query", hidden, hidden))
        tensors.update(_linear(rng, f"{prefix}.attention.self.key", hidden, hidden))
        tensors.update(_linear(rng, f"{prefix}.attention.self.value", hidden, hidden))
        tensors.update(_linear(rng, f"{prefix}.attention.output.dense", hidden, hidden))
        tensors.update(
            _linear(
                rng, f"{prefix}.intermediate.dense", config.intermediate_size, hidden
            )
        )
        tensors.update(
            _linear(rng, f"{prefix}.output.dense", hidden, config.intermediate_size)
        )
        tensors[f"{prefix}.attention.output.LayerNorm.weight"] = rng.standard_normal(
            hidden
        ).astype(np.float32)
        tensors[f"{prefix}.attention.output.LayerNorm.bias"] = rng.standard_normal(
            hidden
        ).astype(np.float32) * np.float32(0.1)
        tensors[f"{prefix}.output.LayerNorm.weight"] = rng.standard_normal(
            hidden
        ).astype(np.float32)
        tensors[f"{prefix}.output.LayerNorm.bias"] = rng.standard_normal(hidden).astype(
            np.float32
        ) * np.float32(0.1)
    tensors.update(_linear(rng, "qa_outputs", 2, hidden, sparsity=0.25))
    return tensors


def _table(rng: np.random.Generator, rows: int, columns: int) -> np.ndarray:
    return rng.standard_normal((rows, columns)).astype(np.float32)


def _linear(
    rng: np.random.Generator,
    name: str,
    out_features: int,
    in_features: int,
    sparsity: float = 0.5,
) -> dict[str, np.ndarray]:
    weight = rng.standard_normal((out_features, in_features)).astype(np.float32)
    weight[rng.random(weight.shape) < sparsity] = 0.0
    bias = rng.standard_normal(out_features).astype(np.float32) * np.float32(0.1)
    return {f"{name}.weight": weight, f"{name}.bias": bias}


def _numpy_span_logits(
    config: BertConfig,
    tensors: dict[str, np.ndarray],
    tokens: TokenBatch,
) -> tuple[np.ndarray, np.ndarray]:
    hidden = _numpy_encode(config, tensors, tokens)
    logits = _dense_linear(hidden, tensors, "qa_outputs")
    return logits[:, 0], logits[:, 1]


def _numpy_encode(
    config: BertConfig,
    tensors: dict[str, np.ndarray],
    tokens: TokenBatch,
) -> np.ndarray:
    input_ids = np.asarray(tokens.input_ids, dtype=np.intp)
    token_type_ids = np.asarray(tokens.token_type_ids, dtype=np.intp)
    attention_mask = np.asarray(tokens.attention_mask, dtype=np.float32)
    hidden = (
        tensors["bert.embeddings.word_embeddings.weight"][input_ids]
        + tensors["bert.embeddings.position_embeddings.weight"][
            np.arange(tokens.length)
        ]
        + tensors["bert.embeddings.token_type_embeddings.weight"][token_type_ids]
    )
    hidden = layer_norm(
        hidden,
        tensors["bert.embeddings.LayerNorm.weight"],
        tensors["bert.embeddings.LayerNorm.bias"],
    )
    for layer_index in range(config.hidden_layer_count):
        prefix = f"bert.encoder.layer.{layer_index}"
        attention = _numpy_self_attention(
            config, tensors, prefix, hidden, attention_mask
        )
        hidden = layer_norm(
            attention + hidden,
            tensors[f"{prefix}.attention.output.LayerNorm.weight"],
            tensors[f"{prefix}.attention.output.LayerNorm.bias"],
        )
        intermediate = gelu(
            _dense_linear(hidden, tensors, f"{prefix}.intermediate.dense")
        )
        feed_forward = _dense_linear(intermediate, tensors, f"{prefix}.output.dense")
        hidden = layer_norm(
            feed_forward + hidden,
            tensors[f"{prefix}.output.LayerNorm.weight"],
            tensors[f"{prefix}.output.LayerNorm.bias"],
        )
    return hidden


def _numpy_self_attention(
    config: BertConfig,
    tensors: dict[str, np.ndarray],
    prefix: str,
    hidden: np.ndarray,
    attention_mask: np.ndarray,
) -> np.ndarray:
    sequence_length = hidden.shape[0]
    head_count = config.attention_head_count
    head_size = config.attention_head_size
    query = _split_heads(
        _dense_linear(hidden, tensors, f"{prefix}.attention.self.query"),
        sequence_length,
        head_count,
        head_size,
    )
    key = _split_heads(
        _dense_linear(hidden, tensors, f"{prefix}.attention.self.key"),
        sequence_length,
        head_count,
        head_size,
    )
    value = _split_heads(
        _dense_linear(hidden, tensors, f"{prefix}.attention.self.value"),
        sequence_length,
        head_count,
        head_size,
    )
    scale = np.float32(head_size**-0.5)
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
    scores = scores + additive_attention_bias(attention_mask)
    weights = softmax(scores)
    context = np.matmul(weights, value)
    merged = np.swapaxes(context, 0, 1).reshape(sequence_length, head_count * head_size)
    return _dense_linear(merged, tensors, f"{prefix}.attention.output.dense")


def _split_heads(
    projected: np.ndarray,
    sequence_length: int,
    head_count: int,
    head_size: int,
) -> np.ndarray:
    return projected.reshape(sequence_length, head_count, head_size).transpose(1, 0, 2)


def _dense_linear(
    activations: np.ndarray,
    tensors: dict[str, np.ndarray],
    name: str,
) -> np.ndarray:
    return np.asarray(
        activations @ tensors[f"{name}.weight"].T + tensors[f"{name}.bias"],
        dtype=np.float32,
    )


def _argmax_span(
    start_logits: tuple[float, ...],
    end_logits: tuple[float, ...],
    tokens: TokenBatch,
) -> tuple[int, int]:
    return highest_scoring_span(start_logits, end_logits, tokens)
