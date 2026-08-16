import pytest

from sparsebert.application import AnswerQuestion
from sparsebert.domain import TokenBatch


class _TestTokenizer:
    def encode(self, question: str, context: str) -> TokenBatch:
        del question, context
        return TokenBatch((101, 7, 8, 102), (0, 0, 1, 1), (1, 1, 1, 1))

    def decode_span(self, tokens: TokenBatch, start_token: int, end_token: int) -> str:
        del tokens, start_token, end_token
        return "two"


class _TestModel:
    def span_logits(
        self, tokens: TokenBatch
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        start = [0.0] * tokens.length
        end = [0.0] * tokens.length
        start[1] = 4.0
        end[1] = 5.0
        return tuple(start), tuple(end)


def test_answer_question_selects_span_from_logits() -> None:
    use_case = AnswerQuestion(_TestTokenizer(), _TestModel())
    span = use_case.answer("How many?", "There are two cats.")
    assert span.start_token == 1
    assert span.end_token == 1
    assert span.text == "two"


def test_answer_question_rejects_blank_question() -> None:
    use_case = AnswerQuestion(_TestTokenizer(), _TestModel())
    with pytest.raises(ValueError, match="question"):
        use_case.answer("   ", "A passage.")
