"""Tokenized question and context for one encoder forward pass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenBatch:
    input_ids: tuple[int, ...]
    token_type_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    # Uncased WordPiece decode would lowercase; offsets slice the original passage.
    offset_mapping: tuple[tuple[int, int], ...] = ()
    context: str = ""

    def __post_init__(self) -> None:
        lengths = {
            len(self.input_ids),
            len(self.token_type_ids),
            len(self.attention_mask),
        }
        if self.offset_mapping:
            lengths.add(len(self.offset_mapping))
        if len(lengths) != 1:
            raise ValueError("Token batch fields must share one length.")
        if any(flag not in (0, 1) for flag in self.attention_mask):
            raise ValueError("Attention mask values must be 0 or 1.")
        for start, end in self.offset_mapping:
            if start < 0 or end < 0:
                raise ValueError("Offset mapping values must be non-negative.")
            if end < start:
                raise ValueError("Offset mapping end must not precede start.")

    @property
    def length(self) -> int:
        return len(self.input_ids)
