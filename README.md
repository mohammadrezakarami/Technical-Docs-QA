# Technical Docs QA

Question answering system over official technical documentation with hybrid retrieval, reranking, extractive reading, explanatory answer synthesis, benchmark evaluation, and deployable web delivery.

## Live Links

- GitHub repository: [mohammadrezakarami/Technical-Docs-QA](https://github.com/mohammadrezakarami/Technical-Docs-QA)
- Hugging Face Space: [mokarami/Technical-Docs-QA](https://hf.co/spaces/mokarami/Technical-Docs-QA)

## What This Project Is

This project is a real QA pipeline built on top of official technical documentation, not a notebook demo and not a toy baseline. It covers the full path from corpus building and indexing to retrieval, answer generation, evaluation, and web deployment.

Current corpus coverage:

- FastAPI official docs
- Pandas official docs
- Python official docs
- Pydantic docs
- Scikit-learn docs

## Main Features

- official documentation crawling and cleaning
- sentence-aware chunking with overlap
- BM25 sparse retrieval
- FAISS dense retrieval with sentence-transformers
- cross-encoder reranking
- extractive QA reader
- grounded explanatory answer synthesis
- no-answer handling
- benchmark evaluation and error analysis
- FastAPI backend
- browser interface
- Docker deployment
- Hugging Face Space deployment

## System Architecture

1. Crawl official documentation pages
2. Clean and normalize the extracted text
3. Build sentence-aware chunks
4. Create sparse and dense retrieval indexes
5. Retrieve candidate chunks with BM25 and dense search
6. Rerank candidates with a cross-encoder
7. Read answer spans with an extractive QA model
8. Aggregate support sentences for explanatory questions
9. Return a grounded final answer through API or web interface
10. Evaluate the system with answer and retrieval metrics

## Web App And API

This repository now exposes the QA system through a web app and API, so the primary presentation path is no longer the terminal.

Available endpoints:

- `GET /`: browser interface
- `GET /api/health`: health check
- `GET /api/status`: artifact readiness summary
- `POST /api/ask`: inference endpoint

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do you set custom response headers in FastAPI, and why does using a Response parameter work?",
    "style": "explanatory",
    "threshold": 0.01
  }'
```

## Run Locally

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the web app locally:

```bash
.venv/bin/python scripts/run_real_qa_api.py
```

Then open:

- [http://127.0.0.1:7860](http://127.0.0.1:7860)

## Docker

Build and run locally with Docker:

```bash
docker build -t technical-docs-qa .
docker run -p 7860:7860 technical-docs-qa
```

## Hugging Face Space Deployment

The project is prepared for a Docker-based Hugging Face Space deployment.

Deployment-related files:

- `Dockerfile`
- `.dockerignore`
- `scripts/run_real_qa_api.py`
- `docs/HF_SPACE_DEPLOYMENT.md`

Recommended deployment strategy:

- keep `artifacts/real_qa/index/` in the deployed repo
- keep `artifacts/real_qa/processed/` in the deployed repo
- keep `artifacts/real_qa/models/` out of the repo
- let public model weights download at runtime when needed

## Evaluation Snapshot

Latest verified benchmark snapshot:

- QA Exact Match: `0.3214`
- QA F1: `0.3959`
- Explanatory F1: `0.2263`
- Explanatory term coverage: `0.6857`
- Retrieval Hit@1: `0.7857`
- Retrieval Hit@3: `0.9286`
- Retrieval MRR: `0.8452`
- No-answer accuracy: `1.0000`
- Example count: `30`

## Reports

Generated reports:

- `artifacts/real_qa/reports/evaluation_report.json`
- `artifacts/real_qa/reports/evaluation_report.md`
- `artifacts/real_qa/reports/error_analysis.md`

Delivery documents:

- `docs/TECHNICAL_REPORT.md`
- `docs/PRESENTATION_SCRIPT.md`

## Repository Layout

- `src/real_qa/`: core QA pipeline
- `src/real_qa/web.py`: FastAPI app
- `src/real_qa/ui/index.html`: browser interface
- `scripts/run_real_qa_api.py`: web server entrypoint
- `scripts/evaluate_real_qa.py`: benchmark evaluation
- `scripts/analyze_real_qa_errors.py`: error analysis
- `data/eval/real_qa_eval.json`: evaluation benchmark
- `artifacts/real_qa/index/`: retrieval indexes
- `artifacts/real_qa/processed/`: processed corpus
- `artifacts/real_qa/reports/`: generated evaluation reports

## Development Utilities

CLI tooling still exists for development, reproducibility, and evaluation, but it is no longer the main presentation path.

Useful commands:

```bash
make build
make train
make serve
make eval
make analyze
make test
```

## Notes

- The benchmark now includes both extractive and explanatory questions.
- Retrieval quality is stronger than final answer quality, which means the main current bottleneck is reading and synthesis rather than document matching.
- The repository is designed to be delivery-oriented and reproducible, not a state-of-the-art research claim.
