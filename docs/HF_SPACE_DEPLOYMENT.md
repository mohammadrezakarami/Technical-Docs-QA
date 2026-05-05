# Hugging Face Space Deployment

This project is now structured so it can be deployed as a Docker-based Hugging Face Space.

## Recommended Deployment Shape

Use a **Docker Space** and keep:

- application code
- `artifacts/real_qa/index/`
- `artifacts/real_qa/processed/`
- `artifacts/real_qa/reports/`

inside the Space repository.

Do **not** include:

- `artifacts/real_qa/raw/`
- `artifacts/real_qa/models/`
- local virtual environments

The Space can download model weights from Hugging Face at build time or first runtime use. The index and processed corpus should be versioned with the app because the QA pipeline requires them to answer questions.

## Why This Layout

- `processed/` and `index/` are small enough to version directly.
- local model snapshots are much heavier and unnecessary because the pipeline already falls back to public model names when local snapshots are absent.
- this keeps the repository lighter while still making the app boot with the exact QA corpus and retrieval index you built.

## Space README Template

Create a Space and use a `README.md` at the root of the Space repo with:

```md
---
title: Technical Docs QA
emoji: 📚
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Technical Docs QA

Retrieval-based question answering over official technical documentation with hybrid retrieval, reranking, extractive reading, and explanatory answer synthesis.
```

## Files Needed in the Space Repo

- `Dockerfile`
- `requirements.txt`
- `scripts/`
- `src/`
- `data/eval/real_qa_eval.json`
- `artifacts/real_qa/index/`
- `artifacts/real_qa/processed/`
- `artifacts/real_qa/reports/`
- `README.md` with the Hugging Face YAML block

## Launch Behavior

The container exposes port `7860`, which matches the Hugging Face Spaces Docker recommendation. The app serves:

- `/` browser interface
- `/api/health`
- `/api/status`
- `/api/ask`

## Important Note

If you want faster cold starts across Space restarts, attach persistent storage and set Hugging Face cache variables in the Space settings. This is optional, but useful because models otherwise may need to be downloaded again after rebuilds.
