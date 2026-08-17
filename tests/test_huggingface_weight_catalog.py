from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from finchbert.application import WeightCatalogPort
from finchbert.infrastructure import HuggingFaceWeightCatalog

_ORACLE_ABSOLUTE_TOLERANCE = 1e-4
_QUERY = "bert.encoder.layer.0.attention.self.query"
_QA_HEAD = "qa_outputs"


def test_reports_sparsity_of_pruned_linear_weights_only() -> None:
    catalog: WeightCatalogPort = HuggingFaceWeightCatalog.from_tensors(
        _tiny_checkpoint()
    )
    reports = {report.layer_name: report for report in catalog.sparsity_reports()}
    assert set(reports) == {_QUERY, _QA_HEAD}
    query = reports[_QUERY]
    assert query.parameter_count == 12
    assert query.nonzero_count == 5
    assert query.sparsity == pytest.approx(1.0 - 5 / 12)


def test_sparse_linear_matches_dense_numpy_oracle() -> None:
    tensors = _tiny_checkpoint()
    catalog = HuggingFaceWeightCatalog.from_tensors(tensors)
    activations = ((1.0, 2.0, 3.0), (0.5, 0.0, 1.0))
    result = catalog.sparse_linear(_QUERY).transform(activations)
    weight = tensors[f"{_QUERY}.weight"]
    bias = tensors[f"{_QUERY}.bias"]
    expected = np.asarray(activations, dtype=np.float32) @ weight.T + bias
    np.testing.assert_allclose(
        np.asarray(result, dtype=np.float32),
        expected,
        atol=_ORACLE_ABSOLUTE_TOLERANCE,
    )


def test_from_pretrained_reads_local_safetensors(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    save_file(_tiny_checkpoint(), checkpoint)
    catalog = HuggingFaceWeightCatalog.from_pretrained(tmp_path)
    assert {report.layer_name for report in catalog.sparsity_reports()} == {
        _QUERY,
        _QA_HEAD,
    }


def test_unknown_linear_layer_is_rejected() -> None:
    catalog = HuggingFaceWeightCatalog.from_tensors(_tiny_checkpoint())
    with pytest.raises(ValueError, match="unknown linear layer"):
        catalog.sparse_linear("bert.encoder.layer.0.attention.self.key")


def test_checkpoint_without_linear_layers_is_rejected() -> None:
    with pytest.raises(ValueError, match="no pruned linear"):
        HuggingFaceWeightCatalog.from_tensors(
            {
                "bert.embeddings.word_embeddings.weight": np.ones(
                    (4, 3), dtype=np.float32
                )
            }
        )


def test_missing_local_checkpoint_is_rejected(tmp_path: Path) -> None:
    missing = tmp_path / "model.safetensors"
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        HuggingFaceWeightCatalog.from_pretrained(missing)


def test_numpy_tensor_returns_dense_embeddings() -> None:
    tensors = _tiny_checkpoint()
    catalog = HuggingFaceWeightCatalog.from_tensors(tensors)
    word = catalog.numpy_tensor("bert.embeddings.word_embeddings.weight")
    np.testing.assert_array_equal(
        word,
        tensors["bert.embeddings.word_embeddings.weight"],
    )


def test_unknown_dense_tensor_is_rejected() -> None:
    catalog = HuggingFaceWeightCatalog.from_tensors(_tiny_checkpoint())
    with pytest.raises(ValueError, match="unknown tensor"):
        catalog.numpy_tensor("bert.embeddings.position_embeddings.weight")


def _tiny_checkpoint() -> dict[str, np.ndarray]:
    query_weight = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    return {
        "bert.embeddings.word_embeddings.weight": np.ones((6, 3), dtype=np.float32),
        "bert.encoder.layer.0.attention.output.LayerNorm.weight": np.ones(
            3, dtype=np.float32
        ),
        f"{_QUERY}.weight": query_weight,
        f"{_QUERY}.bias": np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32),
        f"{_QA_HEAD}.weight": np.array(
            [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        f"{_QA_HEAD}.bias": np.zeros(2, dtype=np.float32),
    }
