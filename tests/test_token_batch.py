import pytest

from sparsebert.domain import TokenBatch


def test_token_batch_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="share one length"):
        TokenBatch((1, 2), (0,), (1, 1))


def test_attention_mask_must_be_binary() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        TokenBatch((1, 2), (0, 1), (1, 2))
