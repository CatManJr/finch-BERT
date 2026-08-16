"""Reject commit messages that contain emoji or break Conventional Commits."""

from __future__ import annotations

import re
import sys
from pathlib import Path

CONVENTIONAL_SUBJECT = re.compile(
    r"^(feat|fix|test|docs|chore|refactor|ci)(\([a-z0-9._-]+\))?!?: .+"
)

EMOJI = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001faff"
    "\U00002700-\U000027bf"
    "\U00002600-\U000026ff"
    "\U0001f1e0-\U0001f1ff"
    "]+"
)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("Usage: check_commit_msg.py COMMIT_EDITMSG")
        return 2
    text = Path(argv[1]).read_text(encoding="utf-8")
    if EMOJI.search(text):
        print("Commit message must not contain emoji.")
        return 1
    subject = next((line for line in text.splitlines() if line.strip()), "")
    if not CONVENTIONAL_SUBJECT.match(subject):
        print(
            "Subject must be a Conventional Commit, for example: "
            "chore: add pre-commit workflow"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
