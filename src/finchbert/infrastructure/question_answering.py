"""Composition root: tokenizer, Finch BERT, and the answer-question use case."""

from __future__ import annotations

from pathlib import Path

from finchbert.application import (
    AnswerQuestion,
    QuestionAnsweringModelPort,
    TokenizerPort,
)
from finchbert.domain import AnswerSpan, BertConfig
from finchbert.infrastructure.bert_question_answering import BertQuestionAnsweringModel
from finchbert.infrastructure.huggingface_tokenizer import HuggingFaceTokenizer


class QuestionAnswering:
    def __init__(
        self,
        tokenizer: TokenizerPort,
        model: QuestionAnsweringModelPort,
    ) -> None:
        self._answer_question = AnswerQuestion(tokenizer, model)

    @classmethod
    def from_pretrained(
        cls,
        source: str | Path | None = None,
        config: BertConfig | None = None,
    ) -> QuestionAnswering:
        settings = config or BertConfig.bert_base()
        tokenizer = HuggingFaceTokenizer.from_pretrained(source, config=settings)
        model = BertQuestionAnsweringModel.from_pretrained(source, config=settings)
        return cls(tokenizer, model)

    def answer(self, question: str, context: str) -> AnswerSpan:
        return self._answer_question.answer(question, context)
