from sparsebert.domain import TokenBatch, highest_scoring_span


def test_highest_scoring_span_respects_mask_and_length() -> None:
    tokens = TokenBatch(
        input_ids=(101, 10, 11, 12, 0),
        token_type_ids=(0, 0, 1, 1, 0),
        attention_mask=(1, 1, 1, 1, 0),
    )
    start_logits = (0.0, 5.0, 1.0, 0.0, 99.0)
    end_logits = (0.0, 1.0, 4.0, 0.5, 99.0)
    start, end = highest_scoring_span(start_logits, end_logits, tokens)
    assert (start, end) == (1, 2)


def test_highest_scoring_span_can_limit_answer_length() -> None:
    tokens = TokenBatch((1, 2, 3), (0, 0, 0), (1, 1, 1))
    start_logits = (10.0, 0.0, 0.0)
    end_logits = (0.0, 0.0, 10.0)
    start, end = highest_scoring_span(
        start_logits,
        end_logits,
        tokens,
        max_answer_tokens=1,
    )
    assert (start, end) == (0, 0)
