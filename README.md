# finch-BERT

CPU hosted question answering with unstructured-sparse oBERT and [finch-tensor](https://github.com/finch-tensor/finch-tensor) SpMM.

This is an experimental Finch project testing finch-tensor for language modeling. Pruned Linear layers run as CSR SpMM in Finch.

The default checkpoint is [RedHatAI/oBERT-12-upstream-pruned-unstructured-90-finetuned-squadv1](https://huggingface.co/RedHatAI/oBERT-12-upstream-pruned-unstructured-90-finetuned-squadv1).

## Setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.
Alternatively, you could manually build virtual environment and use `pip install`.

```bash
UV_HTTP_TIMEOUT=300 uv sync --group dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
```

If PyPI is unstable in some regions (e.g., China):

```bash
set UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
UV_HTTP_TIMEOUT=300 uv sync --group dev
```
Before commit, unset mirror site to pass the pre-commit check.
```bash
set UV_INDEX_URL=
uv lock --default-index https://pypi.org/simple --refresh
```

## Checks

```bash
uv run pre-commit run --all-files
uv run pytest -m "not slow"
```

Direct commits to `main` are blocked by a local hook. Create a feature branch before committing. GitHub Actions skips that hook because it runs on `main` after merge.

## Inference

```bash
uv run python scripts/infer.py --question "How many cats?" --context "There are two cats in the garden."
```

Long passages can be read from UTF-8 files:

```bash
uv run python scripts/infer.py --question-file question.txt --context-file passage.txt
```

`--model` selects a HuggingFace repo id or a local checkpoint directory. `--json` writes the span as JSON. `--max-sequence-length` overrides the default 128-token encoder limit.
