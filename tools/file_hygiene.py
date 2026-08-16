"""File hygiene checks used by local pre-commit hooks."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable, Sequence
from pathlib import Path

MAX_FILE_BYTES = 500 * 1024
MERGE_CONFLICT = re.compile(r"^(<<<<<<<|=======|>>>>>>>)", re.MULTILINE)
PRIVATE_KEY = re.compile(r"BEGIN (RSA |OPENSSH |DSA |EC |PGP )?PRIVATE KEY")
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".pdf",
    ".whl",
    ".so",
}


def _paths(files: Sequence[str]) -> list[Path]:
    return [Path(name) for name in files if Path(name).is_file()]


def _read_text(path: Path) -> str | None:
    if path.suffix.lower() in SKIP_SUFFIXES:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def trailing_whitespace(files: Sequence[str]) -> int:
    changed = False
    for path in _paths(files):
        text = _read_text(path)
        if text is None:
            continue
        lines = text.splitlines(keepends=True)
        rewritten: list[str] = []
        for line in lines:
            if line.endswith("\r\n"):
                rewritten.append(line[:-2].rstrip() + "\r\n")
            elif line.endswith("\n"):
                rewritten.append(line[:-1].rstrip() + "\n")
            else:
                rewritten.append(line.rstrip())
        new_text = "".join(rewritten)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed = True
    return 1 if changed else 0


def end_of_file(files: Sequence[str]) -> int:
    changed = False
    for path in _paths(files):
        text = _read_text(path)
        if text is None or text == "":
            continue
        new_text = text.rstrip("\n") + "\n"
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            changed = True
    return 1 if changed else 0


def mixed_line_ending(files: Sequence[str]) -> int:
    changed = False
    for path in _paths(files):
        raw = path.read_bytes()
        if b"\r\n" not in raw:
            continue
        path.write_bytes(raw.replace(b"\r\n", b"\n"))
        changed = True
    return 1 if changed else 0


def check_json(files: Sequence[str]) -> int:
    failed = False
    for path in _paths(files):
        if path.suffix != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            print(f"{path}: invalid JSON ({error})")
            failed = True
    return 1 if failed else 0


def check_toml(files: Sequence[str]) -> int:
    failed = False
    for path in _paths(files):
        if path.suffix != ".toml":
            continue
        try:
            tomllib.loads(path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as error:
            print(f"{path}: invalid TOML ({error})")
            failed = True
    return 1 if failed else 0


def check_yaml(files: Sequence[str]) -> int:
    import yaml  # type: ignore[import-untyped]

    failed = False
    for path in _paths(files):
        if path.suffix not in {".yml", ".yaml"}:
            continue
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as error:
            print(f"{path}: invalid YAML ({error})")
            failed = True
    return 1 if failed else 0


def merge_conflict(files: Sequence[str]) -> int:
    failed = False
    for path in _paths(files):
        text = _read_text(path)
        if text is None:
            continue
        if MERGE_CONFLICT.search(text):
            print(f"{path}: merge conflict marker")
            failed = True
    return 1 if failed else 0


def large_files(files: Sequence[str]) -> int:
    failed = False
    for path in _paths(files):
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            print(f"{path}: {size} bytes exceeds {MAX_FILE_BYTES}")
            failed = True
    return 1 if failed else 0


def private_key(files: Sequence[str]) -> int:
    failed = False
    for path in _paths(files):
        text = _read_text(path)
        if text is None:
            continue
        if PRIVATE_KEY.search(text):
            print(f"{path}: possible private key")
            failed = True
    return 1 if failed else 0


def _contains_debug_call(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "breakpoint":
            return True
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "set_trace"
            and isinstance(func.value, ast.Name)
            and func.value.id == "pdb"
        ):
            return True
    return False


def debug_statements(files: Sequence[str]) -> int:
    failed = False
    for path in _paths(files):
        if path.suffix != ".py":
            continue
        text = _read_text(path)
        if text is None:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        if _contains_debug_call(tree):
            print(f"{path}: debug statement")
            failed = True
    return 1 if failed else 0


def symlinks(files: Sequence[str]) -> int:
    failed = False
    for path in _paths(files):
        if path.is_symlink() and not path.exists():
            print(f"{path}: broken symlink")
            failed = True
    return 1 if failed else 0


def case_conflict(files: Sequence[str]) -> int:
    seen: dict[str, Path] = {}
    failed = False
    for path in _paths(files):
        key = str(path).lower()
        previous = seen.get(key)
        if previous is not None and previous != path:
            print(f"case conflict: {previous} vs {path}")
            failed = True
        seen[key] = path
    return 1 if failed else 0


def no_commit_to_main(_: Sequence[str]) -> int:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"],
        text=True,
    ).strip()
    if branch == "main":
        print("Direct commits to main are not allowed.")
        return 1
    return 0


HOOKS: dict[str, Callable[[Sequence[str]], int]] = {
    "trailing-whitespace": trailing_whitespace,
    "end-of-file": end_of_file,
    "mixed-line-ending": mixed_line_ending,
    "check-json": check_json,
    "check-toml": check_toml,
    "check-yaml": check_yaml,
    "merge-conflict": merge_conflict,
    "large-files": large_files,
    "private-key": private_key,
    "debug-statements": debug_statements,
    "symlinks": symlinks,
    "case-conflict": case_conflict,
    "no-commit-to-main": no_commit_to_main,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hook", choices=sorted(HOOKS))
    parser.add_argument("files", nargs="*")
    args = parser.parse_args(argv[1:])
    return HOOKS[args.hook](args.files)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
