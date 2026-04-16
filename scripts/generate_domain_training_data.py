from __future__ import annotations

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


SEED = 7


def build_examples() -> list[dict]:
    random.seed(SEED)
    chunks_path = PROJECT_ROOT / "artifacts" / "real_qa" / "processed" / "processed_chunks.parquet"
    eval_path = PROJECT_ROOT / "data" / "eval" / "real_qa_eval.json"
    chunks = pd.read_parquet(chunks_path).fillna("")
    eval_set = json.loads(eval_path.read_text(encoding="utf-8"))
    rows = []

    for item in eval_set:
        if not item.get("answerable", True):
            continue
        source_urls = set(item.get("gold_source_urls", []))
        matched = chunks[chunks["source_url"].isin(source_urls)].head(3)
        for _, row in matched.iterrows():
            rows.append(
                {
                    "query": item["question"],
                    "positive": row["dense_text"],
                    "source_url": row["source_url"],
                    "kind": "gold_eval",
                }
            )

    for _, row in chunks.sample(min(120, len(chunks)), random_state=SEED).iterrows():
        title = str(row["title"]).strip()
        section = str(row["section_title"]).strip()
        source = str(row["source_name"]).strip()
        templates = [
            f"What does {title} explain in {source}?",
            f"What is covered in the section {section}?",
            f"How is {section} described in {source}?",
        ]
        for query in templates[:2]:
            rows.append(
                {
                    "query": query,
                    "positive": row["dense_text"],
                    "source_url": row["source_url"],
                    "kind": "synthetic_title_section",
                }
            )

    deduped = []
    seen = set()
    for row in rows:
        key = (row["query"], row["source_url"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def main() -> None:
    out_dir = PROJECT_ROOT / "data" / "training"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dense_retriever_pairs.jsonl"
    rows = build_examples()
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"training_pair_count": len(rows), "output_path": str(out_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
