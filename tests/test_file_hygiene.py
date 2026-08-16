from pathlib import Path

from file_hygiene import (
    check_json,
    debug_statements,
    end_of_file,
    merge_conflict,
    trailing_whitespace,
)


def test_trailing_whitespace_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello  \nworld\n", encoding="utf-8")
    assert trailing_whitespace([str(path)]) == 1
    assert path.read_text(encoding="utf-8") == "hello\nworld\n"


def test_end_of_file_adds_newline(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("hello", encoding="utf-8")
    assert end_of_file([str(path)]) == 1
    assert path.read_text(encoding="utf-8") == "hello\n"


def test_merge_conflict_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("<<<<<<< HEAD\n", encoding="utf-8")
    assert merge_conflict([str(path)]) == 1


def test_debug_statement_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text("breakpoint()\n", encoding="utf-8")
    assert debug_statements([str(path)]) == 1


def test_debug_string_is_not_a_call(tmp_path: Path) -> None:
    path = tmp_path / "sample.py"
    path.write_text('print("breakpoint()")\n', encoding="utf-8")
    assert debug_statements([str(path)]) == 0


def test_valid_json_passes(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text('{"ok": true}\n', encoding="utf-8")
    assert check_json([str(path)]) == 0
