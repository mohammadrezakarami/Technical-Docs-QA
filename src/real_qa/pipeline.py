from __future__ import annotations

import json
import math
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from src.real_qa.build import STOPWORDS, normalize_ws, split_into_sentences, tokenize
from src.real_qa.settings import BuildConfig


class RealQAPipeline:
    def __init__(self, cfg: BuildConfig) -> None:
        self.cfg = cfg
        self._answer_cache: dict[tuple[str, float, str], dict[str, Any]] = {}
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        chunks_path = self.cfg.processed_dir / "processed_chunks.parquet"
        bm25_path = self.cfg.index_dir / "bm25_retrieval_index.pkl"
        faiss_path = self.cfg.index_dir / "dense_faiss_index.faiss"
        dense_ids_path = self.cfg.index_dir / "dense_chunk_ids.json"
        dense_model_path = self.cfg.dense_model_dir if self.cfg.dense_model_dir.exists() else self.cfg.dense_model_name
        reranker_model_path = self.cfg.reranker_model_dir if self.cfg.reranker_model_dir.exists() else self.cfg.reranker_model_name
        reader_model_path = self.cfg.reader_model_dir if self.cfg.reader_model_dir.exists() else self.cfg.reader_model_name
        for path in [chunks_path, bm25_path, faiss_path, dense_ids_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing required artifact: {path}")
        self.chunks_df = pd.read_parquet(chunks_path).fillna("")
        self.chunk_lookup = {
            str(row["chunk_id"]): row.to_dict()
            for _, row in self.chunks_df.iterrows()
        }
        self.title_terms = {
            str(row["chunk_id"]): set(tokenize(f"{row['title']} {row['section_title']}"))
            for _, row in self.chunks_df.iterrows()
        }
        with bm25_path.open("rb") as handle:
            self.bm25_index = pickle.load(handle)
        self.dense_index = faiss.read_index(str(faiss_path))
        self.dense_chunk_ids = json.loads(dense_ids_path.read_text(encoding="utf-8"))
        self.dense_model = SentenceTransformer(str(dense_model_path))
        self.reranker = CrossEncoder(str(reranker_model_path), max_length=384)
        self.reader_tokenizer = AutoTokenizer.from_pretrained(str(reader_model_path), local_files_only=Path(str(reader_model_path)).exists())
        self.reader_model = AutoModelForQuestionAnswering.from_pretrained(str(reader_model_path), local_files_only=Path(str(reader_model_path)).exists())
        self.reader_model.eval()
        self.doc_freq = self.bm25_index["doc_freq"]
        self.inverted_index = self.bm25_index["inverted_index"]
        self.doc_len = self.bm25_index["doc_len"]
        self.avgdl = float(self.bm25_index["avgdl"])
        self.n_docs = int(self.bm25_index["n_docs"])
        self.k1 = float(self.bm25_index["params"]["k1"])
        self.b = float(self.bm25_index["params"]["b"])

    def detect_question_style(self, question: str) -> str:
        low = question.lower()
        explanatory_markers = [
            "why",
            "how",
            "difference between",
            "compare",
            "when should",
            "when is each",
            "explain",
            "what happens",
            "recommended",
            "best way",
            "versus",
            "vs",
        ]
        if any(marker in low for marker in explanatory_markers):
            return "explanatory"
        return "extractive"

    def normalize_compare_term(self, term: str) -> str:
        cleaned = normalize_ws(term).lower().strip(" .?")
        cleaned = re.sub(r"\b(in|with|for|within)\s+(pandas|fastapi|python|pydantic)\b", "", cleaned)
        cleaned = re.sub(r"\b(and|when|why|how|used|recommended|should|is each|each)\b.*$", "", cleaned)
        cleaned = re.sub(r"[^a-z0-9_./+-]+$", "", cleaned)
        return normalize_ws(cleaned).strip(" .?,")

    def expand_query(self, query: str) -> str:
        expansions = [query]
        low = query.lower()
        heuristics = {
            "response model": "response_model pydantic schema",
            "csv": "read_csv to_csv csv",
            "parquet": "read_parquet to_parquet parquet",
            "headers": "response headers header response",
            "stream": "StreamingResponse stream yield",
            "index and value": "enumerate index value",
            "two sequences": "zip iterable sequence",
            "group rows": "groupby split apply combine",
            "dictionary key and value": "items key value dictionary",
        }
        for phrase, expansion in heuristics.items():
            if phrase in low:
                expansions.append(expansion)
        code_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
        if code_tokens:
            expansions.extend(code_tokens[:4])
        return " ".join(dict.fromkeys(part for part in expansions if part))

    def sentence_split(self, text: str) -> list[str]:
        return split_into_sentences(text)

    def is_low_support_answer(self, payload: dict[str, Any], threshold: float) -> bool:
        answer = str(payload.get("answer_span", "")).strip()
        if payload.get("reranker_score", 0.0) < 0.0:
            return True
        if payload.get("retrieval_score", 0.0) < 0.02 and payload.get("reader_score", 0.0) < max(0.15, threshold):
            return True
        if len(answer) <= 1 and payload.get("reader_score", 0.0) < 0.2:
            return True
        if re.fullmatch(r"[\W_]+", answer):
            return True
        if re.fullmatch(r"\d{1,4}", answer) and payload.get("reader_score", 0.0) < 0.2:
            return True
        return False

    def best_support_sentence(self, query: str, context: str) -> tuple[str, float]:
        query_terms = set(tokenize(query))
        best_sentence = ""
        best_score = 0.0
        for sentence in self.sentence_split(context):
            sentence_terms = set(tokenize(sentence))
            if not sentence_terms:
                continue
            overlap = len(query_terms & sentence_terms) / max(len(query_terms), 1)
            if overlap > best_score:
                best_sentence = sentence
                best_score = overlap
        return best_sentence, best_score

    def sentence_support_score(self, query: str, sentence: str, candidate: dict[str, Any]) -> float:
        query_terms = set(tokenize(query))
        sentence_terms = set(tokenize(sentence))
        if not sentence_terms:
            return 0.0
        overlap = len(query_terms & sentence_terms) / max(len(query_terms), 1)
        title_terms = self.title_terms.get(candidate["chunk_id"], set())
        title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
        length_bonus = min(len(sentence.split()) / 24.0, 1.0)
        return (
            0.45 * overlap
            + 0.20 * max(float(candidate.get("reranker_score", 0.0)), 0.0) / 10.0
            + 0.15 * max(float(candidate.get("hybrid_score", 0.0)), 0.0)
            + 0.10 * title_overlap
            + 0.10 * length_bonus
        )

    def lookup_term_definition(self, term: str, preferred_urls: set[str]) -> str:
        patterns = [
            rf"\.{re.escape(term)}\s+is\s+primarily\s+([^.]+?)(?:\.|$)",
            rf"{re.escape(term)}\s+is\s+primarily\s+([^.]+?)(?:\.|$)",
        ]
        prioritized = []
        fallback = []
        for chunk in self.chunks_df.itertuples(index=False):
            target = prioritized if preferred_urls and str(chunk.source_url) in preferred_urls else fallback
            target.append(chunk)
        for pool in [prioritized, fallback]:
            for chunk in pool:
                text = normalize_ws(str(chunk.chunk_text))
                low = text.lower()
                if term not in low:
                    continue
                for pattern in patterns:
                    match = re.search(pattern, low)
                    if match:
                        return normalize_ws(match.group(1))
        return ""

    def collect_support_sentences(self, query: str, candidates: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen_sentences: set[str] = set()
        for candidate in candidates:
            for sentence in self.sentence_split(candidate["chunk_text"]):
                normalized = normalize_ws(sentence)
                if len(normalized.split()) < 6:
                    continue
                dedupe_key = normalized.lower()
                if dedupe_key in seen_sentences:
                    continue
                score = self.sentence_support_score(query, normalized, candidate)
                rows.append({
                    "sentence": normalized,
                    "score": round(score, 4),
                    "chunk_id": candidate["chunk_id"],
                    "source_name": candidate["source_name"],
                    "title": candidate["title"],
                    "section_title": candidate["section_title"],
                    "source_url": candidate["source_url"],
                })
                seen_sentences.add(dedupe_key)
        rows.extend(self.compare_term_support_sentences(query, candidates, seen_sentences))
        rows.sort(key=lambda item: item["score"], reverse=True)
        return rows[:limit]

    def compare_term_support_sentences(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        seen_sentences: set[str],
    ) -> list[dict[str, Any]]:
        query_low = query.lower()
        match = re.search(r"difference between (?P<a>.+?) and (?P<b>.+?)(?:,|\?|$)", query_low)
        if not match or not candidates:
            return []
        left = self.normalize_compare_term(match.group("a"))
        right = self.normalize_compare_term(match.group("b"))
        terms = [left, right]
        preferred_urls = {item["source_url"] for item in candidates[:3]}
        fallback_rows: list[dict[str, Any]] = []
        for term in terms:
            best_row = None
            best_score = 0.0
            for chunk in self.chunks_df.itertuples(index=False):
                chunk_text = str(chunk.chunk_text)
                chunk_low = chunk_text.lower()
                if term not in chunk_low:
                    continue
                if preferred_urls and str(chunk.source_url) not in preferred_urls:
                    continue
                for sentence in self.sentence_split(chunk_text):
                    sentence_low = sentence.lower()
                    if term not in sentence_low:
                        continue
                    dedupe_key = normalize_ws(sentence).lower()
                    if dedupe_key in seen_sentences:
                        continue
                    pseudo_candidate = {
                        "chunk_id": str(chunk.chunk_id),
                        "source_name": str(chunk.source_name),
                        "title": str(chunk.title),
                        "section_title": str(chunk.section_title),
                        "source_url": str(chunk.source_url),
                        "hybrid_score": 0.04,
                        "reranker_score": 1.5,
                    }
                    score = self.sentence_support_score(query, sentence, pseudo_candidate) + 0.12
                    if score > best_score:
                        best_score = score
                        best_row = {
                            "sentence": normalize_ws(sentence),
                            "score": round(score, 4),
                            "chunk_id": str(chunk.chunk_id),
                            "source_name": str(chunk.source_name),
                            "title": str(chunk.title),
                            "section_title": str(chunk.section_title),
                            "source_url": str(chunk.source_url),
                        }
            if best_row is not None:
                seen_sentences.add(best_row["sentence"].lower())
                fallback_rows.append(best_row)
        return fallback_rows

    def synthesize_explanatory_answer(self, question: str, support_sentences: list[dict[str, Any]]) -> tuple[str, float]:
        if not support_sentences:
            return "", 0.0
        question_low = question.lower()
        selected = support_sentences[:3]
        sentences = [item["sentence"] for item in selected]
        if "difference between" in question_low or "compare" in question_low or " versus " in question_low or " vs " in question_low:
            m = re.search(r"difference between (?P<a>.+?) and (?P<b>.+?)(?:\?|$)", question_low)
            if m:
                left = self.normalize_compare_term(m.group("a"))
                right = self.normalize_compare_term(m.group("b"))
                preferred_urls = {item["source_url"] for item in support_sentences[:4]}
                left_definition = self.lookup_term_definition(left, preferred_urls)
                right_definition = self.lookup_term_definition(right, preferred_urls)
                if left_definition and right_definition:
                    compare_answer = (
                        f"{left} is primarily {left_definition}, while {right} is primarily {right_definition}. "
                        f"Use {left} when you want {left_definition}, and use {right} when you want {right_definition}."
                    )
                    confidence = min(0.99, max(item["score"] for item in support_sentences[:2]))
                    return normalize_ws(compare_answer), confidence
                left_sentences = [item["sentence"] for item in support_sentences if left in item["sentence"].lower()]
                right_sentences = [item["sentence"] for item in support_sentences if right in item["sentence"].lower()]
                if left_sentences and right_sentences:
                    compare_answer = (
                        f"{left} is described as {left_sentences[0].strip('. ')} "
                        f"while {right} is described as {right_sentences[0].strip('. ')}"
                    )
                    confidence = min(0.99, (support_sentences[0]["score"] + support_sentences[1]["score"]) / 2)
                    return normalize_ws(compare_answer), confidence
                merged = []
                if left_sentences:
                    merged.append(left_sentences[0])
                if right_sentences:
                    merged.append(right_sentences[0])
                for sentence in sentences:
                    if sentence not in merged:
                        merged.append(sentence)
                sentences = merged[:3]
        if "why" in question_low:
            prefix = "Based on the documentation, "
        elif "how" in question_low:
            prefix = "According to the retrieved documentation, "
        else:
            prefix = ""
        answer = normalize_ws(" ".join(sentences))
        if prefix and answer:
            answer = normalize_ws(f"{prefix}{answer[0].lower() + answer[1:] if len(answer) > 1 else answer.lower()}")
        confidence = min(0.99, sum(item["score"] for item in selected) / max(len(selected), 1))
        return answer, confidence

    def read_answer(self, question: str, context: str) -> dict[str, Any]:
        encoded = self.reader_tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=384,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        sequence_ids = encoded.sequence_ids(0)
        offset_mapping = encoded["offset_mapping"][0].tolist()
        model_inputs = {
            key: value
            for key, value in encoded.items()
            if key != "offset_mapping"
        }
        with torch.no_grad():
            outputs = self.reader_model(**model_inputs)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)
        context_positions = [idx for idx, seq_id in enumerate(sequence_ids) if seq_id == 1]
        if not context_positions:
            return {"answer": "", "score": 0.0}
        start_indices = sorted(context_positions, key=lambda idx: float(start_probs[idx]), reverse=True)[:12]
        end_indices = sorted(context_positions, key=lambda idx: float(end_probs[idx]), reverse=True)[:12]

        best = None
        best_score = -1.0
        for start_idx in start_indices:
            for end_idx in end_indices:
                if end_idx < start_idx:
                    continue
                if end_idx - start_idx > 30:
                    continue
                score = float(start_probs[start_idx] * end_probs[end_idx])
                if score > best_score:
                    best_score = score
                    best = (start_idx, end_idx)
        if best is None:
            return {"answer": "", "score": 0.0}

        start_char = offset_mapping[best[0]][0]
        end_char = offset_mapping[best[1]][1]
        if end_char <= start_char:
            return {"answer": "", "score": 0.0}
        answer = normalize_ws(context[start_char:end_char])
        if answer.lower() in {"", "[cls]", "[sep]"}:
            return {"answer": "", "score": 0.0}
        return {"answer": answer, "score": best_score}

    def _bm25_idf(self, df: int) -> float:
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def bm25_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        expanded_query = self.expand_query(query)
        q_tokens = [tok for tok in tokenize(expanded_query) if tok not in STOPWORDS]
        if not q_tokens:
            return []
        scores: dict[str, float] = defaultdict(float)
        for term in q_tokens:
            postings = self.inverted_index.get(term, [])
            if not postings:
                continue
            idf = self._bm25_idf(self.doc_freq.get(term, 0))
            for chunk_id, tf in postings:
                dl = self.doc_len[chunk_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                scores[str(chunk_id)] += idf * ((tf * (self.k1 + 1)) / denom)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        rows = []
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            meta = self.chunk_lookup[chunk_id]
            title_overlap = len(set(q_tokens) & self.title_terms[chunk_id]) / max(len(set(q_tokens)), 1)
            rows.append({
                "chunk_id": chunk_id,
                "rank": rank,
                "score": float(score + 0.15 * title_overlap),
                "source_name": meta["source_name"],
                "title": meta["title"],
                "section_title": meta["section_title"],
                "source_url": meta["source_url"],
                "chunk_text": meta["chunk_text"],
            })
        return rows

    def dense_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        expanded_query = self.expand_query(query)
        if expanded_query in self._embedding_cache:
            q_emb = self._embedding_cache[expanded_query]
        else:
            q_emb = self.dense_model.encode([expanded_query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype("float32")
            self._embedding_cache[expanded_query] = q_emb
        scores, indices = self.dense_index.search(q_emb, top_k)
        rows = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.dense_chunk_ids):
                continue
            chunk_id = str(self.dense_chunk_ids[idx])
            meta = self.chunk_lookup[chunk_id]
            query_terms = set(tokenize(expanded_query))
            title_overlap = len(query_terms & self.title_terms[chunk_id]) / max(len(query_terms), 1)
            rows.append({
                "chunk_id": chunk_id,
                "rank": rank,
                "score": float(score + 0.10 * title_overlap),
                "source_name": meta["source_name"],
                "title": meta["title"],
                "section_title": meta["section_title"],
                "source_url": meta["source_url"],
                "chunk_text": meta["chunk_text"],
            })
        return rows

    def hybrid_search(self, query: str) -> list[dict[str, Any]]:
        bm25_rows = self.bm25_search(query, self.cfg.top_k_bm25)
        dense_rows = self.dense_search(query, self.cfg.top_k_dense)
        bm25_map = {row["chunk_id"]: row for row in bm25_rows}
        dense_map = {row["chunk_id"]: row for row in dense_rows}
        rows = []
        query_terms = set(tokenize(self.expand_query(query)))
        for chunk_id in set(bm25_map) | set(dense_map):
            b = bm25_map.get(chunk_id)
            d = dense_map.get(chunk_id)
            meta = self.chunk_lookup[chunk_id]
            score = 0.0
            if b:
                score += 1.0 / (60 + b["rank"])
            if d:
                score += 1.0 / (60 + d["rank"])
            score += 0.10 * (len(query_terms & self.title_terms[chunk_id]) / max(len(query_terms), 1))
            rows.append({
                "chunk_id": chunk_id,
                "hybrid_score": score,
                "source_name": meta["source_name"],
                "title": meta["title"],
                "section_title": meta["section_title"],
                "source_url": meta["source_url"],
                "chunk_text": meta["chunk_text"],
            })
        rows = sorted(rows, key=lambda item: item["hybrid_score"], reverse=True)[: max(self.cfg.top_k_bm25, self.cfg.top_k_dense)]
        return rows

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs = []
        for candidate in candidates:
            passage = normalize_ws(candidate["chunk_text"])[: self.cfg.max_context_chars]
            pair_text = f"Title: {candidate['title']}\nSection: {candidate['section_title']}\nPassage: {passage}"
            pairs.append((query, pair_text))
        if not pairs:
            return []
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The CrossEncoder\.predict `num_workers` argument is deprecated.*",
            )
            scores = self.reranker.predict(pairs)
        reranked = []
        for candidate, score in zip(candidates, scores):
            item = dict(candidate)
            item["reranker_score"] = float(score)
            reranked.append(item)
        reranked = sorted(reranked, key=lambda item: item["reranker_score"], reverse=True)[: self.cfg.top_k_rerank]
        return reranked

    def answer(self, question: str, threshold: float = 0.01, style: str = "auto") -> dict[str, Any]:
        normalized_style = style if style in {"auto", "extractive", "explanatory"} else "auto"
        cache_key = (normalize_ws(question), round(float(threshold), 4), normalized_style)
        if cache_key in self._answer_cache:
            return self._answer_cache[cache_key]
        query = normalize_ws(question)
        question_style = self.detect_question_style(query) if normalized_style == "auto" else normalized_style
        hybrid = self.hybrid_search(query)
        reranked = self.rerank(query, hybrid)
        evidence = []
        best_payload = None
        for candidate in reranked[:5]:
            context = normalize_ws(candidate["chunk_text"])[: self.cfg.max_context_chars]
            try:
                pred = self.read_answer(query, context)
            except Exception:
                continue
            fallback_sentence, fallback_score = self.best_support_sentence(query, context)
            answer_span = normalize_ws(pred.get("answer", ""))
            if fallback_score >= 0.34 and len(answer_span) < 2:
                answer_span = fallback_sentence
            title_match = len(set(tokenize(query)) & self.title_terms[candidate["chunk_id"]]) / max(len(set(tokenize(query))), 1)
            combined_reader_score = max(float(pred.get("score", 0.0)), 0.65 * float(pred.get("score", 0.0)) + 0.20 * fallback_score + 0.15 * title_match)
            item = {
                "chunk_id": candidate["chunk_id"],
                "source_name": candidate["source_name"],
                "title": candidate["title"],
                "section_title": candidate["section_title"],
                "source_url": candidate["source_url"],
                "retrieval_score": round(float(candidate["hybrid_score"]), 4),
                "reranker_score": round(float(candidate["reranker_score"]), 4),
                "reader_score": round(combined_reader_score, 4),
                "answer_span": answer_span,
                "context_preview": context[:500],
            }
            evidence.append(item)
            if best_payload is None or item["reader_score"] > best_payload["reader_score"]:
                best_payload = item
        support_sentences = self.collect_support_sentences(query, reranked, limit=6)
        synthesized_answer, synthesized_confidence = self.synthesize_explanatory_answer(query, support_sentences)
        should_use_synthesis = question_style == "explanatory" and synthesized_answer and synthesized_confidence >= max(0.18, threshold)
        if should_use_synthesis:
            payload = {
                "question": query,
                "question_style": question_style,
                "answer_type": "answer",
                "final_answer": synthesized_answer,
                "confidence": round(float(synthesized_confidence), 4),
                "evidence": evidence,
                "support_sentences": support_sentences,
            }
            self._answer_cache[cache_key] = payload
            return payload
        if best_payload is None or best_payload["reader_score"] < threshold or self.is_low_support_answer(best_payload, threshold):
            payload = {
                "question": query,
                "question_style": question_style,
                "answer_type": "no_answer",
                "final_answer": "",
                "confidence": 0.0,
                "evidence": evidence,
                "support_sentences": support_sentences,
            }
            self._answer_cache[cache_key] = payload
            return payload
        payload = {
            "question": query,
            "question_style": question_style,
            "answer_type": "answer",
            "final_answer": best_payload["answer_span"],
            "confidence": best_payload["reader_score"],
            "evidence": evidence,
            "support_sentences": support_sentences,
        }
        self._answer_cache[cache_key] = payload
        return payload
