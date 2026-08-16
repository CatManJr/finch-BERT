from pathlib import Path

from check_commit_msg import main


def test_accepts_conventional_subject(tmp_path: Path) -> None:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text("chore: add pre-commit workflow\n", encoding="utf-8")
    assert main(["check_commit_msg.py", str(path)]) == 0


def test_rejects_emoji(tmp_path: Path) -> None:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text("chore: add hooks \U0001f600\n", encoding="utf-8")
    assert main(["check_commit_msg.py", str(path)]) == 1


def test_rejects_non_conventional_subject(tmp_path: Path) -> None:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text("added hooks\n", encoding="utf-8")
    assert main(["check_commit_msg.py", str(path)]) == 1
