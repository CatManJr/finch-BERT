from pathlib import Path

import pytest
import transformers
from safetensors.numpy import save_file

from finchbert import QuestionAnswering
from finchbert.domain import BertConfig, TokenBatch
from test_bert_question_answering import _tiny_tensors

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


def test_from_pretrained_answers_from_local_checkpoint(tmp_path: Path) -> None:
    config = _checkpoint_config()
    _write_checkpoint(tmp_path, config)
    qa = QuestionAnswering.from_pretrained(tmp_path, config=config)
    span = qa.answer(_QUESTION, _CONTEXT)
    assert span.start_token <= span.end_token
    assert span.text == "" or span.text in _CONTEXT


def test_constructor_runs_answer_question_use_case() -> None:
    qa = QuestionAnswering(_PassThroughTokenizer(), _FixedModel())
    span = qa.answer(_QUESTION, _CONTEXT)
    assert span.start_token == 1
    assert span.end_token == 1
    assert span.text == "two"


def test_missing_local_checkpoint_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "missing-qa"
    with pytest.raises(FileNotFoundError):
        QuestionAnswering.from_pretrained(missing, config=_checkpoint_config())


class _PassThroughTokenizer:
    def encode(self, question: str, context: str) -> TokenBatch:
        del question, context
        return TokenBatch((101, 7, 8, 102), (0, 0, 1, 1), (1, 1, 1, 1))

    def decode_span(self, tokens: TokenBatch, start_token: int, end_token: int) -> str:
        del tokens, start_token, end_token
        return "two"


class _FixedModel:
    def span_logits(
        self, tokens: TokenBatch
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        start = [0.0] * tokens.length
        end = [0.0] * tokens.length
        start[1] = 4.0
        end[1] = 5.0
        return tuple(start), tuple(end)


def _checkpoint_config() -> BertConfig:
    return BertConfig(
        hidden_size=8,
        hidden_layer_count=1,
        attention_head_count=2,
        intermediate_size=16,
        vocab_size=len(_TINY_VOCAB),
        max_position_count=16,
        type_vocab_size=2,
        max_sequence_length=16,
    )


def _write_checkpoint(directory: Path, config: BertConfig) -> None:
    tokenizer_cls = transformers.BertTokenizer
    tokenizer_cls(vocab=_TINY_VOCAB, do_lower_case=True).save_pretrained(directory)
    save_file(_tiny_tensors(config), directory / "model.safetensors")
