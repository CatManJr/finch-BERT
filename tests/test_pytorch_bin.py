from pathlib import Path

import numpy as np
import pytest

from finchbert.infrastructure.huggingface_weight_catalog import HuggingFaceWeightCatalog
from finchbert.infrastructure.pytorch_bin import load_pytorch_bin, write_pytorch_bin

_QUERY = "bert.encoder.layer.0.attention.self.query"
_QA_HEAD = "qa_outputs"


def test_round_trip_pytorch_bin(tmp_path: Path) -> None:
    tensors = _tiny_tensors()
    checkpoint = tmp_path / "pytorch_model.bin"
    write_pytorch_bin(checkpoint, tensors)
    loaded = load_pytorch_bin(checkpoint)
    assert set(loaded) == set(tensors)
    for name, expected in tensors.items():
        np.testing.assert_array_equal(loaded[name], expected)


def test_from_pretrained_reads_local_pytorch_bin(tmp_path: Path) -> None:
    write_pytorch_bin(tmp_path / "pytorch_model.bin", _tiny_tensors())
    catalog = HuggingFaceWeightCatalog.from_pretrained(tmp_path)
    assert {report.layer_name for report in catalog.sparsity_reports()} == {
        _QUERY,
        _QA_HEAD,
    }


def test_rejects_non_zip_pytorch_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="zip-format"):
        load_pytorch_bin(checkpoint)


def _tiny_tensors() -> dict[str, np.ndarray]:
    return {
        "bert.embeddings.word_embeddings.weight": np.ones((6, 3), dtype=np.float32),
        f"{_QUERY}.weight": np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 0.0, 5.0],
            ],
            dtype=np.float32,
        ),
        f"{_QUERY}.bias": np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32),
        f"{_QA_HEAD}.weight": np.array(
            [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        f"{_QA_HEAD}.bias": np.zeros(2, dtype=np.float32),
    }
