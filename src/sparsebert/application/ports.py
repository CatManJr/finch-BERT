"""Ports for extractive question answering. Implementations live in infrastructure."""

from __future__ import annotations

from typing import Protocol

from sparsebert.domain import SparsityReport, TokenBatch


class TokenizerPort(Protocol):
    def encode(self, question: str, context: str) -> TokenBatch:
        """Turn a question and passage into encoder inputs."""

    def decode_span(self, tokens: TokenBatch, start_token: int, end_token: int) -> str:
        """Map a token span back to passage text."""


class QuestionAnsweringModelPort(Protocol):
    def span_logits(
        self, tokens: TokenBatch
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return start and end logits aligned with the token batch."""


class SparseLinearPort(Protocol):
    def transform(
        self, activations: tuple[tuple[float, ...], ...]
    ) -> tuple[tuple[float, ...], ...]:
        """Apply a pruned linear map. Runtime must use Finch SpMM."""


class WeightCatalogPort(Protocol):
    def sparsity_reports(self) -> tuple[SparsityReport, ...]:
        """Describe nonzero structure of pruned linear layers."""

    def sparse_linear(self, layer_name: str) -> SparseLinearPort:
        """Return the Finch-backed linear map for one named layer."""
