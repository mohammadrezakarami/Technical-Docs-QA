from pathlib import Path

from src.real_qa.build import make_doc_id, split_into_sentences
from src.real_qa.pipeline import RealQAPipeline
from scripts.evaluate_real_qa import normalize_answer, token_f1


def test_make_doc_id_is_deterministic() -> None:
    url = "https://fastapi.tiangolo.com/advanced/response-headers"
    first = make_doc_id("FastAPI", url)
    second = make_doc_id("FastAPI", url)
    assert first == second
    assert first.startswith("fastapi_")


def test_normalize_answer_removes_articles_and_punctuation() -> None:
    assert normalize_answer("The X- prefix.") == "x prefix"


def test_token_f1_matches_exact_answers() -> None:
    assert token_f1("StreamingResponse", "StreamingResponse") == 1.0


def test_split_into_sentences_preserves_multiple_sentences() -> None:
    assert split_into_sentences("Alpha. Beta? Gamma!") == ["Alpha.", "Beta?", "Gamma!"]


def test_detect_question_style_marks_explanatory_queries() -> None:
    pipeline = RealQAPipeline.__new__(RealQAPipeline)
    assert pipeline.detect_question_style("What is the difference between loc and iloc?") == "explanatory"
