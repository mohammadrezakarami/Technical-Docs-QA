from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.real_qa.pipeline import RealQAPipeline
from src.real_qa.settings import BuildConfig


def resolve_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def build_runtime_status(project_root: Path) -> dict[str, Any]:
    cfg = BuildConfig(project_root=project_root)
    manifest_path = cfg.reports_dir / "manifest.json"
    evaluation_path = cfg.reports_dir / "evaluation_report.json"
    required_artifacts = {
        "processed_chunks": cfg.processed_dir / "processed_chunks.parquet",
        "bm25_index": cfg.index_dir / "bm25_retrieval_index.pkl",
        "dense_faiss_index": cfg.index_dir / "dense_faiss_index.faiss",
        "dense_chunk_ids": cfg.index_dir / "dense_chunk_ids.json",
    }
    return {
        "project_root": str(project_root),
        "artifacts_dir": str(cfg.artifacts_dir),
        "manifest_exists": manifest_path.exists(),
        "evaluation_exists": evaluation_path.exists(),
        "required_artifacts": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in required_artifacts.items()
        },
    }


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Question to answer from the QA corpus.")
    threshold: float = Field(0.01, ge=0.0, le=1.0, description="Confidence threshold for answer vs no_answer.")
    style: str = Field("auto", description="Answer style: auto, extractive, or explanatory.")


class AskResponse(BaseModel):
    question: str
    question_style: str
    answer_type: str
    final_answer: str
    confidence: float
    evidence: list[dict[str, Any]]
    support_sentences: list[dict[str, Any]]


def _read_ui_html() -> str:
    ui_path = Path(__file__).resolve().parent / "ui" / "index.html"
    return ui_path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_pipeline() -> RealQAPipeline:
    cfg = BuildConfig(project_root=resolve_project_root())
    return RealQAPipeline(cfg)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Real QA System API",
        version="1.0.0",
        description="API and browser interface for the technical documentation QA system.",
    )

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _read_ui_html()

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return build_runtime_status(resolve_project_root())

    @app.post("/api/ask", response_model=AskResponse)
    async def ask(payload: AskRequest) -> AskResponse:
        style = payload.style if payload.style in {"auto", "extractive", "explanatory"} else "auto"
        try:
            result = get_pipeline().answer(
                payload.question,
                threshold=float(payload.threshold),
                style=style,
            )
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=(
                    "QA artifacts are missing. Run the build step before starting the API. "
                    f"Details: {exc}"
                ),
            ) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"QA inference failed: {exc}") from exc
        return AskResponse(**result)

    return app


app = create_app()
