from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceConfig:
    source_name: str
    base_url: str
    allowed_netloc: str
    seed_pages: list[str]
    allow_patterns: list[str]


DEFAULT_SOURCES = [
    SourceConfig(
        source_name="FastAPI",
        base_url="https://fastapi.tiangolo.com/",
        allowed_netloc="fastapi.tiangolo.com",
        seed_pages=[
            "https://fastapi.tiangolo.com/tutorial/",
            "https://fastapi.tiangolo.com/reference/",
            "https://fastapi.tiangolo.com/advanced/custom-response/",
            "https://fastapi.tiangolo.com/advanced/response-headers/",
        ],
        allow_patterns=[r"/tutorial/", r"/reference/", r"/advanced/"],
    ),
    SourceConfig(
        source_name="Pandas",
        base_url="https://pandas.pydata.org/docs/",
        allowed_netloc="pandas.pydata.org",
        seed_pages=[
            "https://pandas.pydata.org/docs/user_guide/index.html",
            "https://pandas.pydata.org/docs/reference/index.html",
            "https://pandas.pydata.org/docs/user_guide/io.html",
            "https://pandas.pydata.org/docs/user_guide/indexing.html",
        ],
        allow_patterns=[r"/docs/user_guide/", r"/docs/reference/", r"/docs/getting_started/"],
    ),
    SourceConfig(
        source_name="Python",
        base_url="https://docs.python.org/3/",
        allowed_netloc="docs.python.org",
        seed_pages=[
            "https://docs.python.org/3/tutorial/index.html",
            "https://docs.python.org/3/library/index.html",
            "https://docs.python.org/3/tutorial/datastructures.html",
            "https://docs.python.org/3/tutorial/controlflow.html",
        ],
        allow_patterns=[r"/3/tutorial/", r"/3/library/", r"/3/reference/"],
    ),
    SourceConfig(
        source_name="Pydantic",
        base_url="https://docs.pydantic.dev/latest/",
        allowed_netloc="docs.pydantic.dev",
        seed_pages=[
            "https://docs.pydantic.dev/latest/concepts/models/",
            "https://docs.pydantic.dev/latest/concepts/fields/",
        ],
        allow_patterns=[r"/latest/concepts/", r"/latest/api/", r"/latest/examples/"],
    ),
    SourceConfig(
        source_name="Scikit-learn",
        base_url="https://scikit-learn.org/stable/",
        allowed_netloc="scikit-learn.org",
        seed_pages=[
            "https://scikit-learn.org/stable/user_guide.html",
            "https://scikit-learn.org/stable/modules/classes.html",
        ],
        allow_patterns=[r"/stable/user_guide\.html", r"/stable/modules/generated/", r"/stable/modules/classes", r"/stable/model_selection"],
    ),
]


@dataclass(frozen=True)
class BuildConfig:
    project_root: Path
    max_pages_per_source: int = 60
    max_urls_visited_per_source: int = 280
    max_crawl_depth: int = 2
    request_timeout: int = 20
    sleep_seconds: float = 0.15
    min_clean_chars: int = 140
    min_word_count: int = 40
    chunk_size_words: int = 180
    chunk_overlap_words: int = 35
    chunk_min_sentences: int = 4
    chunk_max_sentences: int = 8
    chunk_sentence_overlap: int = 2
    top_k_bm25: int = 20
    top_k_dense: int = 20
    top_k_rerank: int = 7
    max_context_chars: int = 2200
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reader_model_name: str = "deepset/roberta-base-squad2"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts" / "real_qa"

    @property
    def raw_dir(self) -> Path:
        return self.artifacts_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.artifacts_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "index"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def dense_model_dir(self) -> Path:
        return self.models_dir / "dense_retriever"

    @property
    def reranker_model_dir(self) -> Path:
        return self.models_dir / "reranker"

    @property
    def reader_model_dir(self) -> Path:
        return self.models_dir / "reader"

    @property
    def reports_dir(self) -> Path:
        return self.artifacts_dir / "reports"
