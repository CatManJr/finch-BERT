from io import StringIO
from pathlib import Path

from finchbert.domain import BertConfig, TokenBatch
from finchbert.infrastructure.question_answering import QuestionAnswering
from infer import main

_QUESTION = "How many?"
_CONTEXT = "There are two cats."


def test_prints_answer_text() -> None:
    stdout = StringIO()
    stderr = StringIO()
    code = main(
        ["--question", _QUESTION, "--context", _CONTEXT, "--quiet"],
        load_qa=_fake_load,
        stdout=stdout,
        stderr=stderr,
    )
    assert code == 0
    assert stdout.getvalue() == "two\n"
    assert stderr.getvalue() == ""


def test_json_includes_span_offsets() -> None:
    stdout = StringIO()
    code = main(
        ["-q", _QUESTION, "-c", _CONTEXT, "--json", "--quiet"],
        load_qa=_fake_load,
        stdout=stdout,
        stderr=StringIO(),
    )
    assert code == 0
    assert '"text": "two"' in stdout.getvalue()
    assert '"start_token": 1' in stdout.getvalue()
    assert '"end_token": 1' in stdout.getvalue()


def test_reads_question_and_context_from_utf8_files(tmp_path: Path) -> None:
    question_file = tmp_path / "question.txt"
    context_file = tmp_path / "context.txt"
    question_file.write_text(_QUESTION, encoding="utf-8")
    context_file.write_text(_CONTEXT, encoding="utf-8")
    stdout = StringIO()
    code = main(
        [
            "--question-file",
            str(question_file),
            "--context-file",
            str(context_file),
            "--quiet",
        ],
        load_qa=_fake_load,
        stdout=stdout,
        stderr=StringIO(),
    )
    assert code == 0
    assert stdout.getvalue() == "two\n"


def test_rejects_blank_question() -> None:
    stderr = StringIO()
    code = main(
        ["--question", "   ", "--context", _CONTEXT, "--quiet"],
        load_qa=_fake_load,
        stdout=StringIO(),
        stderr=stderr,
    )
    assert code == 1
    assert "question" in stderr.getvalue()


def test_rejects_missing_input_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    stderr = StringIO()
    code = main(
        [
            "--question-file",
            str(missing),
            "--context",
            _CONTEXT,
            "--quiet",
        ],
        load_qa=_fake_load,
        stdout=StringIO(),
        stderr=stderr,
    )
    assert code == 1
    assert "file not found" in stderr.getvalue()


def test_rejects_question_and_question_file_together(tmp_path: Path) -> None:
    question_file = tmp_path / "question.txt"
    question_file.write_text(_QUESTION, encoding="utf-8")
    code = main(
        [
            "--question",
            _QUESTION,
            "--question-file",
            str(question_file),
            "--context",
            _CONTEXT,
        ],
        load_qa=_fake_load,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert code == 2


def test_rejects_non_positive_max_sequence_length() -> None:
    stderr = StringIO()
    code = main(
        [
            "-q",
            _QUESTION,
            "-c",
            _CONTEXT,
            "--max-sequence-length",
            "0",
            "--quiet",
        ],
        load_qa=_fake_load,
        stdout=StringIO(),
        stderr=stderr,
    )
    assert code == 1
    assert "max_sequence_length" in stderr.getvalue()


def test_missing_required_flags_is_a_usage_error() -> None:
    code = main([], load_qa=_fake_load, stdout=StringIO(), stderr=StringIO())
    assert code == 2


def test_help_exits_successfully() -> None:
    stdout = StringIO()
    code = main(["--help"], load_qa=_fake_load, stdout=stdout, stderr=StringIO())
    assert code == 0
    assert "--question" in stdout.getvalue()


def test_passes_model_source_to_loader() -> None:
    seen: dict[str, object] = {}

    def load(source: str | Path | None, config: BertConfig | None) -> QuestionAnswering:
        seen["source"] = source
        seen["config"] = config
        return _fake_load(source, config)

    code = main(
        ["-q", _QUESTION, "-c", _CONTEXT, "--model", "local/ckpt", "--quiet"],
        load_qa=load,
        stdout=StringIO(),
        stderr=StringIO(),
    )
    assert code == 0
    assert seen["source"] == "local/ckpt"


def test_unexpected_error_hides_traceback_without_debug() -> None:
    def load(source: str | Path | None, config: BertConfig | None) -> QuestionAnswering:
        del source, config
        raise RuntimeError("Finch CSR SpMM failed.")

    stderr = StringIO()
    code = main(
        ["-q", _QUESTION, "-c", _CONTEXT, "--quiet"],
        load_qa=load,
        stdout=StringIO(),
        stderr=stderr,
    )
    assert code == 1
    assert "Finch CSR SpMM failed." in stderr.getvalue()
    assert "Traceback" not in stderr.getvalue()


def _fake_load(
    source: str | Path | None, config: BertConfig | None
) -> QuestionAnswering:
    del source, config
    return QuestionAnswering(_PassThroughTokenizer(), _FixedModel())


class _PassThroughTokenizer:
    def encode(self, question: str, context: str) -> TokenBatch:
        del question, context
        return TokenBatch((101, 7, 8, 102), (0, 0, 1, 1), (1, 1, 1, 1))

    def decode_span(self, tokens: TokenBatch, start_token: int, end_token: int) -> str:
        del tokens, start_token, end_token
        return "two"


class _FixedModel:
    def span_logits(
        self, tokens: TokenBatch
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        start = [0.0] * tokens.length
        end = [0.0] * tokens.length
        start[1] = 4.0
        end[1] = 5.0
        return tuple(start), tuple(end)
