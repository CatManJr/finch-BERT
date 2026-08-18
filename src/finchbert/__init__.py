"""BERT-based extractive question answering (SQuAD) with Finch SpMM
for pruned Linear layers."""

from finchbert.infrastructure.question_answering import QuestionAnswering

__all__ = ["QuestionAnswering", "__version__"]
__version__ = "0.1.0"
