from pathlib import Path

from fastapi.testclient import TestClient

from src.real_qa.web import build_runtime_status, create_app


def test_build_runtime_status_reports_required_artifacts() -> None:
    project_root = Path(__file__).resolve().parents[1]
    payload = build_runtime_status(project_root)
    assert payload["project_root"] == str(project_root)
    assert "required_artifacts" in payload
    assert set(payload["required_artifacts"]) == {
        "processed_chunks",
        "bm25_index",
        "dense_faiss_index",
        "dense_chunk_ids",
    }


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_serves_browser_interface() -> None:
    client = TestClient(create_app())
    response = client.get("/")
    assert response.status_code == 200
    assert "Ask the corpus, not your memory." in response.text
