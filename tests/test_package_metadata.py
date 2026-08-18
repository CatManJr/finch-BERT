from finchbert import QuestionAnswering, __version__


def test_version_is_present() -> None:
    assert __version__ == "0.1.0"


def test_question_answering_is_exported() -> None:
    assert QuestionAnswering.__module__ == "finchbert.infrastructure.question_answering"
