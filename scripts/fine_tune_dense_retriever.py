from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.sentence_transformer import losses
from torch.utils.data import DataLoader

from src.real_qa.settings import BuildConfig


def load_pairs(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rebuild_dense_index(cfg: BuildConfig, model: SentenceTransformer) -> None:
    chunks_path = cfg.processed_dir / "processed_chunks.parquet"
    dense_ids_path = cfg.index_dir / "dense_chunk_ids.json"
    faiss_path = cfg.index_dir / "dense_faiss_index.faiss"
    chunks = pd.read_parquet(chunks_path).fillna("")
    embeddings = model.encode(
        chunks["dense_text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(faiss_path))
    dense_ids_path.write_text(json.dumps(chunks["chunk_id"].astype(str).tolist(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    torch.set_num_threads(1)
    cfg = BuildConfig(project_root=PROJECT_ROOT)
    training_path = PROJECT_ROOT / "data" / "training" / "dense_retriever_pairs.jsonl"
    if not training_path.exists():
        raise FileNotFoundError(f"Training pairs not found: {training_path}")

    rows = load_pairs(training_path)
    examples = [InputExample(texts=[row["query"], row["positive"]]) for row in rows]
    model = SentenceTransformer(str(cfg.dense_model_dir if cfg.dense_model_dir.exists() else cfg.dense_model_name))
    train_loader = DataLoader(examples, shuffle=True, batch_size=min(16, max(4, int(math.sqrt(len(examples))))))
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=1,
        warmup_steps=max(1, len(train_loader) // 10),
        show_progress_bar=True,
    )
    model.save(str(cfg.dense_model_dir))
    rebuild_dense_index(cfg, model)
    print(json.dumps({"status": "ok", "trained_pairs": len(rows), "dense_model_dir": str(cfg.dense_model_dir)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
