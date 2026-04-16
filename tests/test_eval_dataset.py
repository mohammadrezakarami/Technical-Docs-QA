import json
from pathlib import Path


def test_eval_dataset_has_required_fields() -> None:
    path = Path(__file__).resolve().parents[1] / "data" / "eval" / "real_qa_eval.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload
    for item in payload:
        assert "id" in item
        assert "question" in item
        assert "answerable" in item
        assert "gold_source_urls" in item
        if item.get("question_type") == "explanatory":
            assert item.get("answer_style") == "explanatory"
            assert item.get("required_terms")
