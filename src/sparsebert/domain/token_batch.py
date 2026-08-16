"""Tokenized question and context for one encoder forward pass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenBatch:
    input_ids: tuple[int, ...]
    token_type_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.token_type_ids),
            len(self.attention_mask),
        }
        if len(lengths) != 1:
            raise ValueError("Token batch fields must share one length.")
        if any(flag not in (0, 1) for flag in self.attention_mask):
            raise ValueError("Attention mask values must be 0 or 1.")

    @property
    def length(self) -> int:
        return len(self.input_ids)
