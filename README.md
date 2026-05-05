# Real QA System

Terminal-first question answering system over official technical documentation with domain-specific retriever tuning, benchmark evaluation, delivery documents, and a lightweight web layer.

## Scope

Current corpus coverage:

- FastAPI official docs
- Pandas official docs
- Python official docs
- Pydantic docs
- Scikit-learn docs

The repository is intentionally focused on one serious QA path. The terminal workflow remains the main engineering path, and a thin FastAPI plus browser interface now sits on top of the same QA pipeline.

## Architecture

- official-doc crawl
- cleaning and chunking
- sentence-aware chunking for longer explanatory spans
- BM25 sparse retrieval
- dense retrieval with FAISS + sentence-transformers
- domain-specific dense retriever tuning from project-generated pairs
- cross-encoder reranking
- extractive answer reading
- multi-chunk support sentence collection
- grounded explanatory answer synthesis
- benchmark evaluation
- automatic error analysis

## Project Layout

- `src/real_qa/`: core build and inference logic
- `scripts/qa_cli.py`: unified CLI
- `scripts/run_real_qa_api.py`: local API server entrypoint
- `scripts/generate_domain_training_data.py`: training-pair generation
- `scripts/fine_tune_dense_retriever.py`: dense retriever tuning
- `scripts/evaluate_real_qa.py`: benchmark evaluation
- `scripts/analyze_real_qa_errors.py`: error analysis
- `src/real_qa/web.py`: FastAPI app
- `src/real_qa/ui/index.html`: browser interface
- `data/eval/real_qa_eval.json`: benchmark set
- `data/training/dense_retriever_pairs.jsonl`: generated training pairs
- `docs/TECHNICAL_REPORT.md`: technical write-up
- `docs/PRESENTATION_SCRIPT.md`: presentation outline
- `artifacts/real_qa/`: generated artifacts, models, and reports

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Unified CLI

Build the documentation corpus, indexes, and local model snapshots:

```bash
.venv/bin/python scripts/qa_cli.py build
```

Generate project-specific retriever training pairs:

```bash
.venv/bin/python scripts/qa_cli.py generate-train
```

Fine-tune the dense retriever and rebuild the dense index:

```bash
.venv/bin/python scripts/qa_cli.py train
```

Ask a question:

```bash
.venv/bin/python scripts/qa_cli.py ask "Which parameter type can you declare in a FastAPI path operation to set response headers?"
```

Run the API and browser interface:

```bash
.venv/bin/python scripts/qa_cli.py serve
```

Then open `http://127.0.0.1:8000`.

Ask for a grounded explanatory answer:

```bash
.venv/bin/python scripts/qa_cli.py ask --style explanatory "How do you set custom response headers in FastAPI, and why does using a Response parameter work?"
```

Evaluate on the benchmark:

```bash
.venv/bin/python scripts/qa_cli.py eval
.venv/bin/python scripts/qa_cli.py eval --sweep
```

Generate error analysis:

```bash
.venv/bin/python scripts/qa_cli.py analyze
```

Check artifact status:

```bash
.venv/bin/python scripts/qa_cli.py status
```

## Make Targets

```bash
make build
make generate-train
make train
make serve
make ask Q="Which Python function pairs entries from two sequences for looping at the same time?"
make eval
make eval-sweep
make analyze
make status
make test
```

## Latest Verified Snapshot

Current benchmark snapshot after explanatory benchmark expansion and grounded synthesis:

- QA Exact Match: `0.3214`
- QA F1: `0.3959`
- Explanatory F1: `0.2263`
- Explanatory term coverage: `0.6857`
- Retrieval Hit@1: `0.7857`
- Retrieval Hit@3: `0.9286`
- Retrieval MRR: `0.8452`
- No-answer accuracy: `1.0000`
- Example count: `30`

## API

Available endpoints:

- `GET /`: browser interface
- `GET /api/health`: health check
- `GET /api/status`: artifact readiness summary
- `POST /api/ask`: ask a question

Example request:

```bash
curl -X POST http://127.0.0.1:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do you set custom response headers in FastAPI, and why does using a Response parameter work?",
    "style": "explanatory",
    "threshold": 0.01
  }'
```

## Docker And Hugging Face Spaces

This repository now includes:

- `Dockerfile`
- `.dockerignore`
- `scripts/run_real_qa_api.py`
- `docs/HF_SPACE_DEPLOYMENT.md`

Recommended deployment strategy:

- keep `artifacts/real_qa/index/` and `artifacts/real_qa/processed/` in the deployed repo
- keep `artifacts/real_qa/models/` out of the repo
- let the app download public model weights when needed

Local Docker run:

```bash
docker build -t real-qa-system .
docker run -p 7860:7860 real-qa-system
```

For Hugging Face Spaces, use a Docker Space and follow:

- `docs/HF_SPACE_DEPLOYMENT.md`

## Reports

Generated reports:

- `artifacts/real_qa/reports/evaluation_report.json`
- `artifacts/real_qa/reports/evaluation_report.md`
- `artifacts/real_qa/reports/error_analysis.md`

Delivery documents:

- `docs/TECHNICAL_REPORT.md`
- `docs/PRESENTATION_SCRIPT.md`

## CI

GitHub Actions CI is included at `.github/workflows/ci.yml` and runs syntax checks plus tests on push and pull request.

## Notes

- Built artifact directories are ignored in `.gitignore` because they can be regenerated.
- The benchmark now includes both extractive and explanatory questions, so token-level answer metrics are stricter than the earlier extractive-only snapshot.
- The benchmark is reproducible and delivery-oriented; it is not intended to claim state-of-the-art research results.
