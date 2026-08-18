import pytest

from finchbert.domain import TokenBatch


def test_token_batch_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="share one length"):
        TokenBatch((1, 2), (0,), (1, 1))


def test_attention_mask_must_be_binary() -> None:
    with pytest.raises(ValueError, match="0 or 1"):
        TokenBatch((1, 2), (0, 1), (1, 2))


def test_offset_mapping_must_match_token_length() -> None:
    with pytest.raises(ValueError, match="share one length"):
        TokenBatch((1, 2), (0, 1), (1, 1), offset_mapping=((0, 1),))


def test_offset_mapping_end_must_not_precede_start() -> None:
    with pytest.raises(ValueError, match="precede"):
        TokenBatch((1,), (0,), (1,), offset_mapping=((4, 1),))
