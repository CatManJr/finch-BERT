"""Extractive question answering with Finch SpMM oBERT."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

_SUCCESS = 0
_RUNTIME_ERROR = 1
_USAGE_ERROR = 2
_INTERRUPTED = 130


def _ensure_src_on_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)


_ensure_src_on_path()

from finchbert import QuestionAnswering  # noqa: E402
from finchbert.domain import AnswerSpan, BertConfig  # noqa: E402
from finchbert.infrastructure.huggingface_weight_catalog import (  # noqa: E402
    DEFAULT_OBERT_SQUAD_MODEL,
)

LoadQuestionAnswering = Callable[
    [str | Path | None, BertConfig | None],
    QuestionAnswering,
]


def main(
    argv: Sequence[str] | None = None,
    *,
    load_qa: LoadQuestionAnswering | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    out = sys.stdout if stdout is None else stdout
    err = sys.stderr if stderr is None else stderr
    parser = _build_parser()
    try:
        args = _parse_args(parser, argv, out, err)
    except SystemExit as exc:
        return _system_exit_code(exc)

    try:
        _configure_stdio(out, err)
        question = _resolve_text(args.question, args.question_file, "question")
        context = _resolve_text(args.context, args.context_file, "context")
        config = _bert_config(args.max_sequence_length)
        source = args.model or None
        if not args.quiet:
            print(f"loading {source or DEFAULT_OBERT_SQUAD_MODEL} ...", file=err)
        loader = load_qa or QuestionAnswering.from_pretrained
        span = loader(source, config).answer(question, context)
        _write_span(out, span, as_json=args.json)
        return _SUCCESS
    except KeyboardInterrupt:
        print("interrupted.", file=err)
        return _INTERRUPTED
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=err)
        return _RUNTIME_ERROR
    except Exception as exc:
        print(f"error: {exc}", file=err)
        if args.debug:
            traceback.print_exc(file=err)
        return _RUNTIME_ERROR


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer.py",
        description=(
            "Answer a question from a passage with unstructured-sparse oBERT "
            "and Finch CSR SpMM."
        ),
    )
    question = parser.add_mutually_exclusive_group(required=True)
    question.add_argument("-q", "--question", help="Question text.")
    question.add_argument(
        "--question-file",
        type=Path,
        help="UTF-8 file containing the question.",
    )
    context = parser.add_mutually_exclusive_group(required=True)
    context.add_argument("-c", "--context", help="Passage text.")
    context.add_argument(
        "--context-file",
        type=Path,
        help="UTF-8 file containing the passage.",
    )
    parser.add_argument(
        "--model",
        help=(
            "HuggingFace repo id or local checkpoint directory "
            f"(default: {DEFAULT_OBERT_SQUAD_MODEL})."
        ),
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        help="Encoder sequence length (default: 128, maximum: 512).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write the answer span as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print checkpoint loading status.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a traceback for unexpected errors.",
    )
    return parser


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    stdout: TextIO,
    stderr: TextIO,
) -> argparse.Namespace:
    previous_out = sys.stdout
    previous_err = sys.stderr
    try:
        sys.stdout = stdout
        sys.stderr = stderr
        return parser.parse_args(None if argv is None else list(argv))
    finally:
        sys.stdout = previous_out
        sys.stderr = previous_err


def _resolve_text(value: str | None, path: Path | None, name: str) -> str:
    if value is not None and path is not None:
        raise ValueError(f"pass only one of --{name} or --{name}-file.")
    if value is not None:
        return value
    if path is None:
        raise ValueError(f"--{name} or --{name}-file is required.")
    return _read_utf8(path)


def _read_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"file not found: {path}.") from exc
    except IsADirectoryError as exc:
        raise ValueError(f"{path} is a directory.") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8.") from exc
    except OSError as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc


def _bert_config(max_sequence_length: int | None) -> BertConfig | None:
    if max_sequence_length is None:
        return None
    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive.")
    return replace(
        BertConfig.bert_base(),
        max_sequence_length=max_sequence_length,
    )


def _write_span(stream: TextIO, span: AnswerSpan, *, as_json: bool) -> None:
    if as_json:
        json.dump(
            {
                "text": span.text,
                "start_token": span.start_token,
                "end_token": span.end_token,
            },
            stream,
            ensure_ascii=False,
        )
        stream.write("\n")
        return
    stream.write(f"{span.text}\n")


def _configure_stdio(stdout: TextIO, stderr: TextIO) -> None:
    for stream in (stdout, stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def _system_exit_code(exc: SystemExit) -> int:
    code = exc.code
    if code is None:
        return _SUCCESS
    if isinstance(code, int):
        return code
    return _USAGE_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
