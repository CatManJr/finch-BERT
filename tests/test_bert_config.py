import pytest

from sparsebert.domain import BertConfig


def test_bert_base_matches_published_shape() -> None:
    config = BertConfig.bert_base()
    assert config.hidden_size == 768
    assert config.hidden_layer_count == 12
    assert config.attention_head_count == 12
    assert config.intermediate_size == 3072
    assert config.attention_head_size == 64
    assert config.max_sequence_length == 128


def test_hidden_size_must_split_into_heads() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        BertConfig(
            hidden_size=100,
            hidden_layer_count=12,
            attention_head_count=12,
            intermediate_size=3072,
            vocab_size=30522,
            max_position_count=512,
            type_vocab_size=2,
            max_sequence_length=128,
        )


def test_sequence_length_cannot_exceed_positions() -> None:
    with pytest.raises(ValueError, match="max_position_count"):
        BertConfig(
            hidden_size=768,
            hidden_layer_count=12,
            attention_head_count=12,
            intermediate_size=3072,
            vocab_size=30522,
            max_position_count=32,
            type_vocab_size=2,
            max_sequence_length=128,
        )
