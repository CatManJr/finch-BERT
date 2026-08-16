from pathlib import Path

FORBIDDEN = (
    "import finch",
    "import transformers",
    "import torch",
    "from finch",
    "from transformers",
    "from torch",
)


def _python_files(package: str) -> list[Path]:
    root = Path("src/sparsebert") / package
    return sorted(root.rglob("*.py"))


def test_domain_and_application_stay_framework_free() -> None:
    files = _python_files("domain") + _python_files("application")
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN:
            assert needle not in text, f"{path} imports a framework: {needle}"
