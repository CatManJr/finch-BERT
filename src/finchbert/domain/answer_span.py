"""Extractive answer span over encoder tokens."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from finchbert.domain.token_batch import TokenBatch

_DEFAULT_MAX_ANSWER_TOKENS = 30


@dataclass(frozen=True, slots=True)
class AnswerSpan:
    start_token: int
    end_token: int
    text: str

    def __post_init__(self) -> None:
        if self.start_token < 0 or self.end_token < 0:
            raise ValueError("Answer token offsets must be non-negative.")
        if self.end_token < self.start_token:
            raise ValueError("Answer end must not precede start.")


def highest_scoring_span(
    start_logits: Sequence[float],
    end_logits: Sequence[float],
    tokens: TokenBatch,
    *,
    max_answer_tokens: int = _DEFAULT_MAX_ANSWER_TOKENS,
) -> tuple[int, int]:
    if len(start_logits) != tokens.length or len(end_logits) != tokens.length:
        raise ValueError("Logits length must match the token batch.")
    if max_answer_tokens <= 0:
        raise ValueError("max_answer_tokens must be positive.")

    best_start = 0
    best_end = 0
    best_score: float | None = None
    for start in range(tokens.length):
        if tokens.attention_mask[start] == 0:
            continue
        last_end = min(tokens.length - 1, start + max_answer_tokens - 1)
        for end in range(start, last_end + 1):
            if tokens.attention_mask[end] == 0:
                continue
            score = start_logits[start] + end_logits[end]
            if best_score is None or score > best_score:
                best_score = score
                best_start = start
                best_end = end
    if best_score is None:
        raise ValueError("Token batch has no attended positions.")
    return best_start, best_end
