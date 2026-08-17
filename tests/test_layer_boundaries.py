from pathlib import Path

FORBIDDEN_MODULES = (
    "finch",
    "transformers",
    "torch",
    "huggingface_hub",
    "safetensors",
)


def _python_files(package: str) -> list[Path]:
    root = Path("src/finchbert") / package
    return sorted(root.rglob("*.py"))


def _imports_module(text: str, module: str) -> bool:
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("import "):
            imported = line.removeprefix("import ").split()[0].split(",")[0]
        elif line.startswith("from "):
            imported = line.removeprefix("from ").split()[0]
        else:
            continue
        if imported == module or imported.startswith(f"{module}."):
            return True
    return False


def test_domain_and_application_stay_framework_free() -> None:
    files = _python_files("domain") + _python_files("application")
    assert files
    for path in files:
        text = path.read_text(encoding="utf-8")
        for module in FORBIDDEN_MODULES:
            assert not _imports_module(text, module), (
                f"{path} imports a framework: {module}"
            )
