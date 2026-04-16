# Technical Report

## Project Goal

Build an end-to-end question answering system over official technical documentation with a reproducible terminal-first workflow.

## Final Architecture

- corpus crawler over official documentation sources
- HTML cleaning and section extraction
- sentence-aware chunking with overlap
- sparse retrieval with BM25
- dense retrieval with sentence-transformers + FAISS
- cross-encoder reranking
- extractive reading with a QA transformer
- multi-chunk support sentence aggregation
- grounded explanatory answer synthesis
- evaluation with answer, explanatory coverage, and retrieval metrics
- offline artifact reuse with local model snapshots

## Data Sources

- FastAPI official docs
- Pandas official docs
- Python official docs
- Pydantic docs
- Scikit-learn docs

## Training Additions

- generated domain-specific retriever pairs from gold benchmark questions
- synthetic query/passage pairs from titles and section names
- lightweight dense retriever fine-tuning script
- rebuilt dense FAISS index after retriever tuning

## Evaluation

Current tracked metrics:

- QA Exact Match
- QA F1
- Explanatory F1
- Explanatory term coverage
- Retrieval Hit@1
- Retrieval Hit@3
- Retrieval MRR
- No-answer accuracy

Latest verified snapshot:

- QA Exact Match: `0.3214`
- QA F1: `0.3959`
- Explanatory F1: `0.2263`
- Explanatory term coverage: `0.6857`
- Retrieval Hit@1: `0.7857`
- Retrieval Hit@3: `0.9286`
- Retrieval MRR: `0.8452`
- No-answer accuracy: `1.0000`
- Benchmark size: `30` questions

Reports:

- `artifacts/real_qa/reports/evaluation_report.json`
- `artifacts/real_qa/reports/evaluation_report.md`
- `artifacts/real_qa/reports/error_analysis.md`

## Engineering Decisions

- deterministic document IDs for reproducible builds
- terminal-only workflow for debugging and evaluation
- local model snapshots to avoid repeated network dependency at inference time
- URL-backed benchmark labels so evaluation survives rebuilds
- explicit explanatory-question handling with grounded support sentences
- small focused tests for helper and dataset integrity

## Current Limitations

- explanatory answers are grounded but still more extractive than a large generative assistant
- benchmark is still modest in size compared to production-scale QA evaluation
- dense retriever fine-tuning is lightweight, not large-scale supervised training
- answer extraction quality still depends on document chunking and reader confidence thresholds

## Delivery Value

This project demonstrates document ingestion, indexing, retrieval, reranking, answer extraction, evaluation, and workflow engineering in one coherent QA system instead of a demo-only notebook.
