# Presentation Script

## One-line Pitch

I built a grounded QA system over official technical documentation with reproducible artifact generation, benchmark evaluation, and terminal-first usage for both factoid and explanatory questions.

## What It Does

- crawls official docs
- converts them into searchable chunks
- retrieves evidence with sparse and dense search
- fine-tunes the dense retriever on project-generated domain pairs
- reranks passages
- extracts answers with a QA reader
- synthesizes grounded explanatory answers from multiple supporting chunks
- evaluates quality with benchmark metrics

## Why It Is Not Just a Demo

- it has a real artifact pipeline
- it stores local model snapshots and indexes
- it includes benchmark evaluation and error analysis
- it has deterministic rebuild behavior
- it has CI and tests

## Suggested Live Flow

1. Show `scripts/qa_cli.py status`
2. Show `scripts/qa_cli.py ask "Which parameter type can you declare in a FastAPI path operation to set response headers?"`
3. Show `scripts/qa_cli.py ask --style explanatory "How do you set custom response headers in FastAPI, and why does using a Response parameter work?"`
4. Show `scripts/qa_cli.py eval --threshold 0.0`
5. Open `artifacts/real_qa/reports/evaluation_report.md`
6. Open `artifacts/real_qa/reports/error_analysis.md`

## Honest Framing

- this is a serious QA system, not a novelty LLM product
- the main strength is end-to-end retrieval QA engineering with grounded explanatory synthesis
- the next scaling step would be larger supervised training and a larger benchmark
- current snapshot is already reproducible and benchmarked, not just a notebook demo
