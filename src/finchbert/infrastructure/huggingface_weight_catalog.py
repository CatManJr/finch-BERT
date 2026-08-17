"""Load oBERT HuggingFace checkpoints into Finch sparse linear maps."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from safetensors.numpy import load_file

from finchbert.domain import SparsityReport
from finchbert.infrastructure.finch_sparse_linear import FinchSparseLinear

DEFAULT_OBERT_SQUAD_MODEL = (
    "RedHatAI/oBERT-12-upstream-pruned-unstructured-90-finetuned-squadv1"
)
_WEIGHT_DTYPE = np.float32
_WEIGHT_SUFFIX = ".weight"
_LINEAR_WEIGHT_SUFFIXES = (
    ".query.weight",
    ".key.weight",
    ".value.weight",
    ".dense.weight",
    "qa_outputs.weight",
)


class HuggingFaceWeightCatalog:
    def __init__(self, tensors: Mapping[str, np.ndarray]) -> None:
        self._tensors = {
            name: np.array(value, dtype=_WEIGHT_DTYPE, copy=True)
            for name, value in tensors.items()
        }
        self._linear_names = tuple(
            _layer_name(name)
            for name in self._tensors
            if _is_pruned_linear_weight(name)
        )
        if not self._linear_names:
            raise ValueError("checkpoint has no pruned linear layers.")
        for layer_name in self._linear_names:
            weight = self._tensors[f"{layer_name}{_WEIGHT_SUFFIX}"]
            if weight.ndim != 2:
                raise ValueError(f"{layer_name} must be a 2-D linear weight.")
        self._linears: dict[str, FinchSparseLinear] = {}

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, ArrayLike]) -> HuggingFaceWeightCatalog:
        arrays = {name: np.asarray(value) for name, value in tensors.items()}
        return cls(arrays)

    @classmethod
    def from_pretrained(
        cls, source: str | Path | None = None
    ) -> HuggingFaceWeightCatalog:
        checkpoint = DEFAULT_OBERT_SQUAD_MODEL if source is None else source
        return cls.from_tensors(_load_checkpoint_tensors(checkpoint))

    def sparsity_reports(self) -> tuple[SparsityReport, ...]:
        return tuple(
            _sparsity_report(layer_name, self._tensors[f"{layer_name}{_WEIGHT_SUFFIX}"])
            for layer_name in self._linear_names
        )

    def sparse_linear(self, layer_name: str) -> FinchSparseLinear:
        if layer_name not in self._linear_names:
            raise ValueError(f"unknown linear layer: {layer_name}.")
        linear = self._linears.get(layer_name)
        if linear is None:
            weight = self._tensors[f"{layer_name}{_WEIGHT_SUFFIX}"]
            bias_name = f"{layer_name}.bias"
            bias = self._tensors.get(bias_name)
            linear = FinchSparseLinear.from_dense_weight(weight, bias)
            self._linears[layer_name] = linear
        return linear

    def numpy_tensor(self, name: str) -> np.ndarray:
        tensor = self._tensors.get(name)
        if tensor is None:
            raise ValueError(f"unknown tensor: {name}.")
        return tensor


def _is_pruned_linear_weight(tensor_name: str) -> bool:
    return tensor_name.endswith(_LINEAR_WEIGHT_SUFFIXES)


def _layer_name(tensor_name: str) -> str:
    return tensor_name[: -len(_WEIGHT_SUFFIX)]


def _sparsity_report(layer_name: str, weight: np.ndarray) -> SparsityReport:
    return SparsityReport(
        layer_name=layer_name,
        nonzero_count=int(np.count_nonzero(weight)),
        parameter_count=int(weight.size),
    )


def _load_checkpoint_tensors(source: str | Path) -> dict[str, np.ndarray]:
    path = Path(source)
    if path.exists():
        if path.is_file():
            return dict(load_file(path))
        return _load_directory(path)
    if _looks_like_local_path(path):
        raise FileNotFoundError(f"checkpoint not found: {path}.")
    return _load_hub_repository(str(source))


def _looks_like_local_path(path: Path) -> bool:
    text = str(path)
    return path.is_absolute() or path.suffix == ".safetensors" or text.startswith(".")


def _load_directory(directory: Path) -> dict[str, np.ndarray]:
    single = directory / "model.safetensors"
    if single.is_file():
        return dict(load_file(single))
    index_path = directory / "model.safetensors.index.json"
    if index_path.is_file():
        return _load_sharded(directory, index_path)
    raise FileNotFoundError(f"no safetensors checkpoint in {directory}.")


def _load_sharded(directory: Path, index_path: Path) -> dict[str, np.ndarray]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    tensors: dict[str, np.ndarray] = {}
    for shard in dict.fromkeys(weight_map.values()):
        tensors.update(load_file(directory / str(shard)))
    return tensors


def _load_hub_repository(repo_id: str) -> dict[str, np.ndarray]:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        weights = Path(hf_hub_download(repo_id, filename="model.safetensors"))
        return dict(load_file(weights))
    except EntryNotFoundError:
        try:
            index_path = Path(
                hf_hub_download(repo_id, filename="model.safetensors.index.json")
            )
        except EntryNotFoundError as exc:
            raise FileNotFoundError(
                f"repository {repo_id!r} has no safetensors checkpoint."
            ) from exc
        index = json.loads(index_path.read_text(encoding="utf-8"))
        tensors: dict[str, np.ndarray] = {}
        for shard in dict.fromkeys(index["weight_map"].values()):
            shard_path = hf_hub_download(repo_id, filename=str(shard))
            tensors.update(load_file(shard_path))
        return tensors
