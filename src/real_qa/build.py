from __future__ import annotations

import json
import pickle
import re
import time
from hashlib import sha1
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import faiss
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.real_qa.settings import BuildConfig, DEFAULT_SOURCES, SourceConfig


HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; CodexRealQA/1.0)"}
BANNED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".tar", ".gz", ".pdf", ".xml", ".css", ".js")
NOISE_PATTERNS = [
    r"^\s*table of contents\s*$",
    r"^\s*previous\s*$",
    r"^\s*next\s*$",
    r"^\s*contents\s*$",
    r"^\s*navigation\s*$",
    r"^\s*search\s*$",
    r"^\s*skip to content\s*$",
    r"^\s*edit on github\s*$",
    r"^\s*view page source\s*$",
]
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to", "of", "in", "on", "at", "by", "with",
    "from", "as", "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those", "it", "its",
    "into", "than", "can", "could", "should", "would", "may", "might", "will", "shall", "do", "does", "did",
    "how", "what", "which", "when", "where", "why", "who", "whom", "about", "through", "using", "use",
}


def normalize_ws(text: Any) -> str:
    text = str(text)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    return value


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.findall(r"[a-zA-Z0-9_/.+-]+", normalize_ws(text).lower()) if tok not in STOPWORDS]


def split_into_sentences(text: str) -> list[str]:
    text = normalize_ws(text)
    if not text:
        return []
    parts = [normalize_ws(part) for part in re.split(r"(?<=[.!?])\s+|\n+", text) if normalize_ws(part)]
    sentences: list[str] = []
    buffer = ""
    for part in parts:
        candidate = normalize_ws(f"{buffer} {part}" if buffer else part)
        if len(candidate.split()) < 8 and not re.search(r"[.!?]$", part):
            buffer = candidate
            continue
        sentences.append(candidate)
        buffer = ""
    if buffer:
        if sentences:
            sentences[-1] = normalize_ws(f"{sentences[-1]} {buffer}")
        else:
            sentences.append(buffer)
    return sentences


def normalize_url(url: str) -> str:
    url = str(url).strip().split("#")[0]
    parsed = urlparse(url)
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    normalized = f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    return normalized


def make_doc_id(source_name: str, normalized_url: str) -> str:
    source_slug = source_name.lower().replace(" ", "_")
    digest = sha1(normalized_url.encode("utf-8")).hexdigest()[:12]
    return f"{source_slug}_{digest}"


def is_valid_url(url: str, source: SourceConfig) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if source.allowed_netloc not in parsed.netloc:
        return False
    if parsed.path.lower().endswith(BANNED_EXTENSIONS):
        return False
    return any(re.search(pattern, url) for pattern in source.allow_patterns)


def fetch_html(url: str, timeout: int) -> tuple[str | None, str | None]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout)
        if response.status_code == 200 and "text/html" in response.headers.get("Content-Type", ""):
            return response.text, normalize_url(response.url)
        return None, None
    except Exception:
        return None, None


def extract_text(html: str) -> tuple[str, str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for selector in [
        "nav", "header", "footer", "aside", ".sidebar", ".sphinxsidebar", ".wy-nav-side",
        ".related", ".toctree-wrapper", ".toc", ".breadcrumbs", ".headerlink", ".search",
        ".mobile-nav", ".topbar", ".navbar", ".prev-next-area", ".rst-footer-buttons",
        ".bd-sidebar", ".bd-header", ".bd-footer",
    ]:
        for tag in soup.select(selector):
            tag.decompose()
    title = normalize_ws(soup.title.text) if soup.title and soup.title.text else "Untitled"
    section = ""
    for tag_name in ["h1", "h2"]:
        node = soup.find(tag_name)
        if node:
            section = normalize_ws(node.get_text(" ", strip=True))
            break
    main_candidates = []
    for selector in ["main", "article", "[role='main']", ".document", ".body", ".content", ".section", ".bd-content", ".sk-page-content"]:
        try:
            main_candidates.extend(soup.select(selector))
        except Exception:
            pass
    if main_candidates:
        raw_text = max(main_candidates, key=lambda node: len(node.get_text(" ", strip=True))).get_text("\n", strip=True)
    else:
        raw_text = soup.get_text("\n", strip=True)
    lines = []
    for line in normalize_ws(raw_text).splitlines():
        low = line.strip().lower()
        if not line.strip():
            continue
        if any(re.match(pattern, low) for pattern in NOISE_PATTERNS):
            continue
        lines.append(line.strip())
    cleaned = normalize_ws("\n".join(lines))
    return title, section, cleaned


def collect_links(html: str, current_url: str, source: SourceConfig) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for anchor in soup.find_all("a", href=True):
        full = normalize_url(urljoin(current_url, anchor["href"].strip()))
        if is_valid_url(full, source):
            links.append(full)
    deduped = []
    seen = set()
    for link in links:
        if link not in seen:
            seen.add(link)
            deduped.append(link)
    return deduped


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> list[str]:
    words = normalize_ws(text).split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size_words)
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = max(start + 1, end - overlap_words)
    return chunks


def chunk_sentences(
    text: str,
    min_sentences: int,
    max_sentences: int,
    sentence_overlap: int,
    chunk_size_words: int,
    overlap_words: int,
) -> list[str]:
    sentences = split_into_sentences(text)
    if len(sentences) > max(80, len(normalize_ws(text).split()) // 6):
        return chunk_text(text, chunk_size_words, overlap_words)
    if len(sentences) < min_sentences:
        return chunk_text(text, chunk_size_words, overlap_words)
    chunks = []
    start = 0
    while start < len(sentences):
        end = min(len(sentences), start + max_sentences)
        candidate = sentences[start:end]
        if len(candidate) < min_sentences and chunks:
            tail = " ".join(candidate).strip()
            if tail:
                chunks[-1] = normalize_ws(f"{chunks[-1]} {tail}")
            break
        chunk = normalize_ws(" ".join(candidate))
        if chunk:
            chunks.append(chunk)
        if end >= len(sentences):
            break
        start = max(start + 1, end - sentence_overlap)
    return chunks or chunk_text(text, chunk_size_words, overlap_words)


def build_dense_text(row: pd.Series) -> str:
    parts = []
    if row.get("source_name"):
        parts.append(f"Source: {row['source_name']}")
    if row.get("title"):
        parts.append(f"Document: {row['title']}")
    if row.get("section_title"):
        parts.append(f"Section: {row['section_title']}")
    parts.append(row["chunk_text"])
    return "\n".join(parts).strip()


def crawl_sources(cfg: BuildConfig, sources: list[SourceConfig]) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    crawl_report: dict[str, Any] = {}
    for source in sources:
        queue = deque([(normalize_url(url), 0) for url in source.seed_pages])
        visited = set()
        accepted = 0
        fetched = 0
        while queue and accepted < cfg.max_pages_per_source and len(visited) < cfg.max_urls_visited_per_source:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)
            html, final_url = fetch_html(url, timeout=cfg.request_timeout)
            time.sleep(cfg.sleep_seconds)
            if html is None or final_url is None:
                continue
            fetched += 1
            title, section_title, clean_text = extract_text(html)
            if len(clean_text) < cfg.min_clean_chars or len(clean_text.split()) < cfg.min_word_count:
                continue
            raw_path = cfg.raw_dir / f"{source.source_name.lower().replace(' ', '_')}_{accepted:04d}.html"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(html, encoding="utf-8", errors="ignore")
            doc_id = make_doc_id(source.source_name, final_url)
            records.append({
                "doc_id": doc_id,
                "source_name": source.source_name,
                "source_url": final_url,
                "title": title,
                "section_title": section_title,
                "clean_text": clean_text,
                "crawl_depth": depth,
                "raw_file_path": str(raw_path),
            })
            accepted += 1
            if depth < cfg.max_crawl_depth:
                for link in collect_links(html, final_url, source):
                    if link not in visited:
                        queue.append((link, depth + 1))
        crawl_report[source.source_name] = {"visited": len(visited), "fetched": fetched, "accepted": accepted}
    df = pd.DataFrame(records).drop_duplicates(subset=["source_url"]).reset_index(drop=True)
    return df, crawl_report


def build_chunks_df(docs_df: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    rows = []
    for _, row in docs_df.iterrows():
        chunks = chunk_sentences(
            row["clean_text"],
            min_sentences=cfg.chunk_min_sentences,
            max_sentences=cfg.chunk_max_sentences,
            sentence_overlap=cfg.chunk_sentence_overlap,
            chunk_size_words=cfg.chunk_size_words,
            overlap_words=cfg.chunk_overlap_words,
        )
        for idx, chunk in enumerate(chunks):
            rows.append({
                "chunk_id": f"{row['doc_id']}_chunk_{idx:03d}",
                "doc_id": row["doc_id"],
                "source_name": row["source_name"],
                "title": row["title"],
                "section_title": row["section_title"],
                "source_url": row["source_url"],
                "chunk_text": chunk,
                "token_length": len(tokenize(chunk)),
                "sentence_count": len(split_into_sentences(chunk)),
                "position_in_doc": idx,
            })
    chunks_df = pd.DataFrame(rows)
    chunks_df["dense_text"] = chunks_df.apply(build_dense_text, axis=1)
    return chunks_df


def build_bm25_index(chunks_df: pd.DataFrame) -> dict[str, Any]:
    doc_freq = Counter()
    inverted_index = defaultdict(list)
    doc_len = {}
    for _, row in chunks_df.iterrows():
        chunk_id = row["chunk_id"]
        body_tokens = tokenize(row["chunk_text"])
        counts = Counter(body_tokens)
        doc_len[chunk_id] = len(body_tokens)
        for term in set(body_tokens):
            doc_freq[term] += 1
        for term, tf in counts.items():
            inverted_index[term].append((chunk_id, tf))
    avgdl = sum(doc_len.values()) / max(len(doc_len), 1)
    return {
        "doc_freq": dict(doc_freq),
        "inverted_index": dict(inverted_index),
        "doc_len": doc_len,
        "avgdl": avgdl,
        "n_docs": len(doc_len),
        "params": {"k1": 1.5, "b": 0.75},
    }


def build_dense_index(chunks_df: pd.DataFrame, cfg: BuildConfig) -> tuple[faiss.IndexFlatIP, list[str]]:
    model_path = cfg.dense_model_dir if cfg.dense_model_dir.exists() else cfg.dense_model_name
    model = SentenceTransformer(str(model_path))
    embeddings = model.encode(
        chunks_df["dense_text"].tolist(),
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks_df["chunk_id"].astype(str).tolist()


def persist_model_snapshots(cfg: BuildConfig) -> dict[str, str]:
    if not cfg.dense_model_dir.exists():
        dense_model = SentenceTransformer(cfg.dense_model_name)
        dense_model.save(str(cfg.dense_model_dir))

    if not cfg.reranker_model_dir.exists():
        reranker = CrossEncoder(cfg.reranker_model_name, max_length=384)
        reranker.save(str(cfg.reranker_model_dir))

    if not cfg.reader_model_dir.exists():
        reader_tokenizer = AutoTokenizer.from_pretrained(cfg.reader_model_name)
        reader_model = AutoModelForQuestionAnswering.from_pretrained(cfg.reader_model_name)
        reader_tokenizer.save_pretrained(str(cfg.reader_model_dir))
        reader_model.save_pretrained(str(cfg.reader_model_dir))

    return {
        "dense_model": str(cfg.dense_model_dir),
        "reranker_model": str(cfg.reranker_model_dir),
        "reader_model": str(cfg.reader_model_dir),
    }


def build_real_artifacts(cfg: BuildConfig) -> dict[str, Any]:
    for folder in [cfg.raw_dir, cfg.processed_dir, cfg.index_dir, cfg.models_dir, cfg.reports_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    docs_json = cfg.processed_dir / "documents.json"
    chunks_parquet = cfg.processed_dir / "processed_chunks.parquet"
    bm25_path = cfg.index_dir / "bm25_retrieval_index.pkl"
    faiss_path = cfg.index_dir / "dense_faiss_index.faiss"
    dense_ids_path = cfg.index_dir / "dense_chunk_ids.json"
    manifest_path = cfg.reports_dir / "manifest.json"

    if docs_json.exists():
        docs_df = pd.DataFrame(json.loads(docs_json.read_text(encoding="utf-8")))
        crawl_report = {"mode": "cached_documents"}
    else:
        docs_df, crawl_report = crawl_sources(cfg, DEFAULT_SOURCES)
        if len(docs_df) == 0:
            raise RuntimeError("No documentation pages were collected. Check network or crawl constraints.")
        docs_df.to_json(docs_json, orient="records", indent=2, force_ascii=False)

    chunks_df = build_chunks_df(docs_df, cfg)
    if len(chunks_df) == 0:
        raise RuntimeError("No chunks were created from the crawled documentation.")

    chunks_df.to_parquet(chunks_parquet, index=False)

    bm25_index = build_bm25_index(chunks_df)
    with bm25_path.open("wb") as handle:
        pickle.dump(bm25_index, handle)

    dense_index, dense_chunk_ids = build_dense_index(chunks_df, cfg)
    faiss.write_index(dense_index, str(faiss_path))
    dense_ids_path.write_text(json.dumps(dense_chunk_ids, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    model_paths = persist_model_snapshots(cfg)

    manifest = {
        "build_config": to_jsonable(asdict(cfg)),
        "crawl_report": crawl_report,
        "document_count": int(len(docs_df)),
        "chunk_count": int(len(chunks_df)),
        "artifacts": {
            "documents": str(docs_json),
            "chunks": str(chunks_parquet),
            "bm25_index": str(bm25_path),
            "dense_index": str(faiss_path),
            "dense_chunk_ids": str(dense_ids_path),
            **model_paths,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest
