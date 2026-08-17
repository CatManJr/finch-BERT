"""Answer a question from a passage."""

from __future__ import annotations

from finchbert.application.ports import QuestionAnsweringModelPort, TokenizerPort
from finchbert.domain import AnswerSpan, highest_scoring_span


class AnswerQuestion:
    def __init__(
        self,
        tokenizer: TokenizerPort,
        model: QuestionAnsweringModelPort,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model

    def answer(self, question: str, context: str) -> AnswerSpan:
        _require_text("question", question)
        _require_text("context", context)
        tokens = self._tokenizer.encode(question, context)
        start_logits, end_logits = self._model.span_logits(tokens)
        start_token, end_token = highest_scoring_span(
            start_logits,
            end_logits,
            tokens,
        )
        text = self._tokenizer.decode_span(tokens, start_token, end_token)
        return AnswerSpan(start_token, end_token, text)


def _require_text(name: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{name} must be non-empty.")
