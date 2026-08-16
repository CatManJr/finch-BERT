import pytest

from sparsebert.domain import SparsityReport


def test_sparsity_is_one_minus_density() -> None:
    report = SparsityReport("query", nonzero_count=10, parameter_count=100)
    assert report.sparsity == pytest.approx(0.9)


def test_nonzero_cannot_exceed_parameter_count() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        SparsityReport("query", nonzero_count=5, parameter_count=4)
