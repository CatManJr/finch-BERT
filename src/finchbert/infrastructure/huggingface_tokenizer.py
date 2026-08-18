"""HuggingFace WordPiece tokenizer for extractive question answering."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from finchbert.domain import BertConfig, TokenBatch
from finchbert.infrastructure.huggingface_weight_catalog import (
    DEFAULT_OBERT_SQUAD_MODEL,
)

_CONTEXT_TOKEN_TYPE = 1


class _EncodedPair(Protocol):
    def __getitem__(self, key: str) -> Any: ...


class HuggingFaceTokenizer:
    def __init__(self, tokenizer: object, max_sequence_length: int) -> None:
        _require_positive("max_sequence_length", max_sequence_length)
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("tokenizer must be a fast Hugging Face tokenizer.")
        self._tokenizer = tokenizer
        self._max_sequence_length = max_sequence_length

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path | None = None,
        config: BertConfig | None = None,
    ) -> HuggingFaceTokenizer:
        settings = config or BertConfig.bert_base()
        tokenizer = _load_pretrained(_resolve_source(source))
        return cls(tokenizer, max_sequence_length=settings.max_sequence_length)

    def encode(self, question: str, context: str) -> TokenBatch:
        encoded = _encode_pair(
            self._tokenizer, question, context, self._max_sequence_length
        )
        input_ids = _int_tuple(encoded["input_ids"])
        token_type_ids = _int_tuple(encoded["token_type_ids"])
        attention_mask = _int_tuple(encoded["attention_mask"])
        offset_mapping = _offset_tuple(encoded["offset_mapping"])
        if not (
            len(input_ids)
            == len(token_type_ids)
            == len(attention_mask)
            == len(offset_mapping)
        ):
            raise ValueError("tokenizer produced misaligned encodings.")
        if len(input_ids) > self._max_sequence_length:
            raise ValueError("encoded sequence exceeds max_sequence_length.")
        return TokenBatch(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            offset_mapping=offset_mapping,
            context=context,
        )

    def decode_span(self, tokens: TokenBatch, start_token: int, end_token: int) -> str:
        _require_span_bounds(tokens, start_token, end_token)
        if len(tokens.offset_mapping) != tokens.length:
            raise ValueError("token batch is missing offset mapping.")
        char_start: int | None = None
        char_end: int | None = None
        for index in range(start_token, end_token + 1):
            if tokens.attention_mask[index] == 0:
                continue
            if tokens.token_type_ids[index] != _CONTEXT_TOKEN_TYPE:
                continue
            start, end = tokens.offset_mapping[index]
            if start == end:
                continue
            if char_start is None:
                char_start = start
            char_end = end
        if char_start is None or char_end is None:
            return ""
        return tokens.context[char_start:char_end]


def _load_pretrained(source: str) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source)


def _resolve_source(source: str | Path | None) -> str:
    if source is None:
        return DEFAULT_OBERT_SQUAD_MODEL
    path = Path(source)
    if path.exists():
        return str(path)
    if _looks_like_local_path(path):
        raise FileNotFoundError(f"tokenizer not found: {path}.")
    return str(source)


def _looks_like_local_path(path: Path) -> bool:
    text = str(path)
    return path.is_absolute() or path.suffix != "" or text.startswith(".")


def _encode_pair(
    tokenizer: object,
    question: str,
    context: str,
    max_sequence_length: int,
) -> _EncodedPair:
    if not callable(tokenizer):
        raise ValueError("tokenizer must be a fast Hugging Face tokenizer.")
    encode = cast(Any, tokenizer)
    try:
        encoded = encode(
            question,
            context,
            add_special_tokens=True,
            truncation="only_second",
            max_length=max_sequence_length,
            padding=False,
            return_offsets_mapping=True,
            return_token_type_ids=True,
            return_attention_mask=True,
        )
    except Exception as exc:
        raise ValueError("failed to encode question and context.") from exc
    if not isinstance(encoded, Mapping):
        raise ValueError("failed to encode question and context.")
    return cast(_EncodedPair, encoded)


def _int_tuple(values: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def _offset_tuple(values: Any) -> tuple[tuple[int, int], ...]:
    offsets: list[tuple[int, int]] = []
    for raw in values:
        if raw is None:
            offsets.append((0, 0))
            continue
        start, end = raw
        offsets.append((int(start), int(end)))
    return tuple(offsets)


def _require_span_bounds(tokens: TokenBatch, start_token: int, end_token: int) -> None:
    if tokens.length == 0:
        raise ValueError("token batch must be non-empty.")
    if start_token < 0 or end_token < 0:
        raise ValueError("Answer token offsets must be non-negative.")
    if end_token < start_token:
        raise ValueError("Answer end must not precede start.")
    if start_token >= tokens.length or end_token >= tokens.length:
        raise ValueError("Answer token offsets must lie in the token batch.")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
