"""BERT encoder configuration for extractive question answering."""

from __future__ import annotations

from dataclasses import dataclass

_BERT_BASE_HIDDEN_SIZE = 768
_BERT_BASE_HIDDEN_LAYERS = 12
_BERT_BASE_ATTENTION_HEADS = 12
_BERT_BASE_INTERMEDIATE_SIZE = 3072
_BERT_BASE_VOCAB_SIZE = 30522
_BERT_BASE_MAX_POSITIONS = 512
_BERT_BASE_TYPE_VOCAB_SIZE = 2
_DEFAULT_MAX_SEQUENCE_LENGTH = 128


@dataclass(frozen=True, slots=True)
class BertConfig:
    hidden_size: int
    hidden_layer_count: int
    attention_head_count: int
    intermediate_size: int
    vocab_size: int
    max_position_count: int
    type_vocab_size: int
    max_sequence_length: int

    def __post_init__(self) -> None:
        _require_positive("hidden_size", self.hidden_size)
        _require_positive("hidden_layer_count", self.hidden_layer_count)
        _require_positive("attention_head_count", self.attention_head_count)
        _require_positive("intermediate_size", self.intermediate_size)
        _require_positive("vocab_size", self.vocab_size)
        _require_positive("max_position_count", self.max_position_count)
        _require_positive("type_vocab_size", self.type_vocab_size)
        _require_positive("max_sequence_length", self.max_sequence_length)
        if self.hidden_size % self.attention_head_count != 0:
            raise ValueError("hidden_size must divide evenly by attention_head_count.")
        if self.max_sequence_length > self.max_position_count:
            raise ValueError("max_sequence_length cannot exceed max_position_count.")

    @property
    def attention_head_size(self) -> int:
        return self.hidden_size // self.attention_head_count

    @classmethod
    def bert_base(cls) -> BertConfig:
        return cls(
            hidden_size=_BERT_BASE_HIDDEN_SIZE,
            hidden_layer_count=_BERT_BASE_HIDDEN_LAYERS,
            attention_head_count=_BERT_BASE_ATTENTION_HEADS,
            intermediate_size=_BERT_BASE_INTERMEDIATE_SIZE,
            vocab_size=_BERT_BASE_VOCAB_SIZE,
            max_position_count=_BERT_BASE_MAX_POSITIONS,
            type_vocab_size=_BERT_BASE_TYPE_VOCAB_SIZE,
            max_sequence_length=_DEFAULT_MAX_SEQUENCE_LENGTH,
        )


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
