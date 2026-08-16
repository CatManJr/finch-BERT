"""Sparsity of one pruned linear layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SparsityReport:
    layer_name: str
    nonzero_count: int
    parameter_count: int

    def __post_init__(self) -> None:
        if not self.layer_name:
            raise ValueError("layer_name must be non-empty.")
        if self.nonzero_count < 0:
            raise ValueError("nonzero_count cannot be negative.")
        if self.parameter_count <= 0:
            raise ValueError("parameter_count must be positive.")
        if self.nonzero_count > self.parameter_count:
            raise ValueError("nonzero_count cannot exceed parameter_count.")

    @property
    def sparsity(self) -> float:
        return 1.0 - self.nonzero_count / self.parameter_count
