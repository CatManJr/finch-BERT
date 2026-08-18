from pathlib import Path
from typing import Any

import pytest
import transformers

from finchbert.application import AnswerQuestion, TokenizerPort
from finchbert.domain import TokenBatch
from finchbert.infrastructure import HuggingFaceTokenizer

_QUESTION = "How many?"
_CONTEXT = "There are two cats."
_TINY_VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "how": 5,
    "many": 6,
    "there": 7,
    "are": 8,
    "two": 9,
    "cats": 10,
    "?": 11,
    ".": 12,
}


def test_adapter_satisfies_tokenizer_port() -> None:
    tokenizer: TokenizerPort = _tiny_tokenizer()
    tokens = tokenizer.encode(_QUESTION, _CONTEXT)
    assert tokens.context == _CONTEXT
    assert tokens.length == len(tokens.input_ids)
    assert tokens.input_ids[0] == _TINY_VOCAB["[CLS]"]
    assert tokens.input_ids[-1] == _TINY_VOCAB["[SEP]"]
    assert 1 in tokens.token_type_ids


def test_decode_span_recovers_original_passage_casing() -> None:
    tokenizer = _tiny_tokenizer()
    tokens = tokenizer.encode(_QUESTION, _CONTEXT)
    there = _context_token_index(tokens, "There")
    assert tokenizer.decode_span(tokens, there, there) == "There"
    two = _context_token_index(tokens, "two")
    cats = _context_token_index(tokens, "cats")
    assert tokenizer.decode_span(tokens, two, cats) == "two cats"


def test_decode_span_ignores_question_tokens() -> None:
    tokenizer = _tiny_tokenizer()
    tokens = tokenizer.encode(_QUESTION, _CONTEXT)
    how = tokens.input_ids.index(_TINY_VOCAB["how"])
    assert tokens.token_type_ids[how] == 0
    assert tokenizer.decode_span(tokens, how, how) == ""


def test_answer_question_uses_huggingface_offsets() -> None:
    tokenizer = _tiny_tokenizer()
    tokens = tokenizer.encode(_QUESTION, _CONTEXT)
    two = _context_token_index(tokens, "two")
    span = AnswerQuestion(tokenizer, _FixedSpanModel(two)).answer(_QUESTION, _CONTEXT)
    assert span.start_token == two
    assert span.end_token == two
    assert span.text == "two"


def test_truncation_keeps_question_and_limits_length() -> None:
    tokenizer = HuggingFaceTokenizer(_backend(), max_sequence_length=8)
    tokens = tokenizer.encode(_QUESTION, _CONTEXT)
    assert tokens.length == 8
    assert tokens.input_ids[0] == _TINY_VOCAB["[CLS]"]
    assert _TINY_VOCAB["how"] in tokens.input_ids
    assert _TINY_VOCAB["cats"] not in tokens.input_ids


def test_from_pretrained_reads_local_tokenizer(tmp_path: Path) -> None:
    backend = _backend()
    backend.save_pretrained(tmp_path)
    tokenizer = HuggingFaceTokenizer.from_pretrained(tmp_path)
    tokens = tokenizer.encode(_QUESTION, _CONTEXT)
    two = _context_token_index(tokens, "two")
    assert tokenizer.decode_span(tokens, two, two) == "two"


def test_missing_local_tokenizer_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing-tokenizer"
    with pytest.raises(FileNotFoundError, match="tokenizer not found"):
        HuggingFaceTokenizer.from_pretrained(missing)


def test_rejects_slow_tokenizer() -> None:
    class _SlowTokenizer:
        is_fast = False

    with pytest.raises(ValueError, match="fast"):
        HuggingFaceTokenizer(_SlowTokenizer(), max_sequence_length=8)


def test_rejects_non_positive_max_sequence_length() -> None:
    with pytest.raises(ValueError, match="max_sequence_length"):
        HuggingFaceTokenizer(_backend(), max_sequence_length=0)


def test_decode_span_rejects_offsets_outside_batch() -> None:
    tokenizer = _tiny_tokenizer()
    tokens = tokenizer.encode(_QUESTION, _CONTEXT)
    with pytest.raises(ValueError, match="lie in the token batch"):
        tokenizer.decode_span(tokens, 0, tokens.length)


def test_decode_span_rejects_missing_offset_mapping() -> None:
    tokenizer = _tiny_tokenizer()
    tokens = TokenBatch((2, 5, 3), (0, 0, 0), (1, 1, 1))
    with pytest.raises(ValueError, match="offset mapping"):
        tokenizer.decode_span(tokens, 1, 1)


class _FixedSpanModel:
    def __init__(self, index: int) -> None:
        self._index = index

    def span_logits(
        self, tokens: TokenBatch
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        start = [0.0] * tokens.length
        end = [0.0] * tokens.length
        start[self._index] = 4.0
        end[self._index] = 5.0
        return tuple(start), tuple(end)


def _tiny_tokenizer() -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer(_backend(), max_sequence_length=32)


def _backend() -> Any:
    return transformers.BertTokenizer(vocab=_TINY_VOCAB, do_lower_case=True)


def _context_token_index(tokens: TokenBatch, text: str) -> int:
    for index, (start, end) in enumerate(tokens.offset_mapping):
        if tokens.token_type_ids[index] != 1:
            continue
        if tokens.context[start:end] == text:
            return index
    raise AssertionError(f"context token {text!r} not found")
