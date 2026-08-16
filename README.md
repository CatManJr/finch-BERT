# finch-sparseBERT

CPU extractive question answering with unstructured-sparse oBERT and [finch-tensor](https://github.com/finch-tensor/finch-tensor) SpMM.

This is an experimental Finch project. Pruned Linear layers run as CSR SpMM in Finch. There is no SciPy, NumPy, or PyTorch runtime fallback for that kernel.

The default checkpoint is [RedHatAI/oBERT-12-upstream-pruned-unstructured-90-finetuned-squadv1](https://huggingface.co/RedHatAI/oBERT-12-upstream-pruned-unstructured-90-finetuned-squadv1).

## Setup

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required. On macOS ARM, Numba pulls a prebuilt `llvmlite` wheel; raise the download timeout instead of installing LLVM.

```bash
UV_HTTP_TIMEOUT=300 uv sync --group dev
uv run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
```

If PyPI is slow:

```bash
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
UV_HTTP_TIMEOUT=300 uv sync --group dev
```

## Checks

```bash
uv run pre-commit run --all-files
uv run pytest -m "not slow"
```

Direct commits to `main` are blocked by a local hook. Create a feature branch before committing. GitHub Actions skips that hook because it runs on `main` after merge.

## Inference

The question-answering CLI lands in a later slice:

```bash
uv run python scripts/infer.py --question "..." --context "..."
```
