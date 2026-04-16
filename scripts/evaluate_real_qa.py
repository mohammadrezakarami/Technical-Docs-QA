from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import warnings
from pathlib import Path
from statistics import mean

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message=r"The CrossEncoder\.predict `num_workers` argument is deprecated.*",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.real_qa.pipeline import RealQAPipeline
from src.real_qa.settings import BuildConfig


def normalize_answer(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = " ".join(text.split())
    return text


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = min(pred_tokens.count(token), gold_tokens.count(token))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_exact_match(prediction: str, answers: list[str]) -> float:
    norm_pred = normalize_answer(prediction)
    return float(any(norm_pred == normalize_answer(answer) for answer in answers))


def best_f1(prediction: str, answers: list[str]) -> float:
    return max((token_f1(prediction, answer) for answer in answers), default=0.0)


def required_term_coverage(prediction: str, required_terms: list[str]) -> float:
    if not required_terms:
        return 0.0
    normalized_prediction = normalize_answer(prediction)
    covered = sum(1 for term in required_terms if normalize_answer(term) in normalized_prediction)
    return covered / len(required_terms)


def load_eval_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_example(pipeline: RealQAPipeline, item: dict, threshold: float) -> dict:
    question_type = item.get("question_type", "extractive")
    answer_style = item.get("answer_style", "auto")
    result = pipeline.answer(item["question"], threshold=threshold, style=answer_style)
    predicted_ids = [entry["chunk_id"] for entry in result.get("evidence", [])]
    predicted_urls = [entry.get("source_url", "") for entry in result.get("evidence", [])]
    gold_ids = set(item.get("gold_chunk_ids", []))
    gold_urls = set(item.get("gold_source_urls", []))
    predicted_answer = result.get("final_answer", "")
    answerable = bool(item.get("answerable", True))

    example = {
        "id": item["id"],
        "question": item["question"],
        "answerable": answerable,
        "question_type": question_type,
        "prediction": predicted_answer,
        "answer_type": result.get("answer_type"),
        "confidence": float(result.get("confidence", 0.0)),
        "evidence": result.get("evidence", []),
        "support_sentences": result.get("support_sentences", []),
        "exact_match": 0.0,
        "f1": 0.0,
        "required_term_coverage": 0.0,
        "hit_at_1": 0.0,
        "hit_at_3": 0.0,
        "mrr": 0.0,
        "no_answer_correct": 0.0,
    }

    if gold_ids or gold_urls:
        def is_gold(rank_index: int) -> bool:
            chunk_match = rank_index < len(predicted_ids) and predicted_ids[rank_index] in gold_ids
            url_match = rank_index < len(predicted_urls) and predicted_urls[rank_index] in gold_urls
            return chunk_match or url_match

        example["hit_at_1"] = float(any(is_gold(rank_index) for rank_index in range(min(1, len(predicted_ids)))))
        example["hit_at_3"] = float(any(is_gold(rank_index) for rank_index in range(min(3, len(predicted_ids)))))
        reciprocal_ranks = [
            1.0 / rank
            for rank, (chunk_id, source_url) in enumerate(zip(predicted_ids, predicted_urls), start=1)
            if chunk_id in gold_ids or source_url in gold_urls
        ]
        example["mrr"] = reciprocal_ranks[0] if reciprocal_ranks else 0.0
    else:
        example["hit_at_1"] = 1.0 if not predicted_ids else 0.0
        example["hit_at_3"] = 1.0 if not predicted_ids else 0.0
        example["mrr"] = 1.0 if not predicted_ids else 0.0

    if answerable:
        answers = item.get("answers", [])
        example["exact_match"] = best_exact_match(predicted_answer, answers)
        example["f1"] = best_f1(predicted_answer, answers)
        example["required_term_coverage"] = required_term_coverage(predicted_answer, item.get("required_terms", []))
    else:
        example["no_answer_correct"] = float(result.get("answer_type") == "no_answer")
        example["exact_match"] = example["no_answer_correct"]
        example["f1"] = example["no_answer_correct"]

    return example


def summarize_examples(examples: list[dict], threshold: float) -> dict:
    answerable_examples = [item for item in examples if item["answerable"]]
    explanatory_examples = [item for item in answerable_examples if item.get("question_type") == "explanatory"]
    negative_examples = [item for item in examples if not item["answerable"]]
    summary = {
        "threshold": threshold,
        "example_count": len(examples),
        "answerable_count": len(answerable_examples),
        "explanatory_count": len(explanatory_examples),
        "negative_count": len(negative_examples),
        "qa_exact_match": mean(item["exact_match"] for item in answerable_examples) if answerable_examples else 0.0,
        "qa_f1": mean(item["f1"] for item in answerable_examples) if answerable_examples else 0.0,
        "explanatory_f1": mean(item["f1"] for item in explanatory_examples) if explanatory_examples else 0.0,
        "explanatory_term_coverage": mean(item["required_term_coverage"] for item in explanatory_examples) if explanatory_examples else 0.0,
        "retrieval_hit_at_1": mean(item["hit_at_1"] for item in answerable_examples) if answerable_examples else 0.0,
        "retrieval_hit_at_3": mean(item["hit_at_3"] for item in answerable_examples) if answerable_examples else 0.0,
        "retrieval_mrr": mean(item["mrr"] for item in answerable_examples) if answerable_examples else 0.0,
        "no_answer_accuracy": mean(item["no_answer_correct"] for item in negative_examples) if negative_examples else 0.0,
        "overall_score": mean(item["f1"] for item in examples) if examples else 0.0,
    }
    return summary


def build_markdown_report(output: dict) -> str:
    summary = output["best_summary"]
    lines = [
        "# Real QA Evaluation Report",
        "",
        "## Summary",
        "",
        f"- Selected threshold: `{output['selected_threshold']}`",
        f"- Best threshold by QA F1: `{output['best_threshold_by_qa_f1']}`",
        f"- Example count: `{summary['example_count']}`",
        f"- Answerable questions: `{summary['answerable_count']}`",
        f"- Explanatory questions: `{summary['explanatory_count']}`",
        f"- Negative questions: `{summary['negative_count']}`",
        f"- QA Exact Match: `{summary['qa_exact_match']:.4f}`",
        f"- QA F1: `{summary['qa_f1']:.4f}`",
        f"- Explanatory F1: `{summary['explanatory_f1']:.4f}`",
        f"- Explanatory term coverage: `{summary['explanatory_term_coverage']:.4f}`",
        f"- Retrieval Hit@1: `{summary['retrieval_hit_at_1']:.4f}`",
        f"- Retrieval Hit@3: `{summary['retrieval_hit_at_3']:.4f}`",
        f"- Retrieval MRR: `{summary['retrieval_mrr']:.4f}`",
        f"- No-answer accuracy: `{summary['no_answer_accuracy']:.4f}`",
        f"- Overall score: `{summary['overall_score']:.4f}`",
        "",
        "## Threshold Sweep",
        "",
        "| Threshold | QA EM | QA F1 | Expl. F1 | Expl. Coverage | Hit@1 | Hit@3 | MRR | No-answer |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in output["all_summaries"]:
        lines.append(
            f"| `{item['threshold']}` | `{item['qa_exact_match']:.4f}` | `{item['qa_f1']:.4f}` | "
            f"`{item['explanatory_f1']:.4f}` | `{item['explanatory_term_coverage']:.4f}` | "
            f"`{item['retrieval_hit_at_1']:.4f}` | `{item['retrieval_hit_at_3']:.4f}` | "
            f"`{item['retrieval_mrr']:.4f}` | `{item['no_answer_accuracy']:.4f}` |"
        )

    failed = [
        item for item in output["examples"]
        if item["answerable"] and (item["f1"] < 1.0 or item["hit_at_1"] < 1.0)
    ]
    lines.extend(["", "## Representative Gaps", ""])
    if not failed:
        lines.append("- No failed answerable examples in the current benchmark.")
    else:
        for item in failed[:5]:
            top_evidence = item["evidence"][0] if item["evidence"] else {}
            lines.extend([
                f"### {item['id']}",
                f"- Question: {item['question']}",
                f"- Prediction: `{item['prediction']}`",
                f"- Question type: `{item['question_type']}`",
                f"- Exact Match: `{item['exact_match']:.4f}`",
                f"- F1: `{item['f1']:.4f}`",
                f"- Required term coverage: `{item['required_term_coverage']:.4f}`",
                f"- Hit@1: `{item['hit_at_1']:.4f}`",
                f"- Top evidence chunk: `{top_evidence.get('chunk_id', '')}`",
                f"- Top source: `{top_evidence.get('title', '')}`",
                f"- Top URL: {top_evidence.get('source_url', '')}",
                "",
            ])
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Evaluate the real QA pipeline with QA and retrieval metrics.")
    parser.add_argument(
        "--eval-set",
        default=str(PROJECT_ROOT / "data" / "eval" / "real_qa_eval.json"),
        help="Path to the evaluation JSON file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Reader confidence threshold used to decide answer vs no_answer.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate several thresholds and report the best threshold by QA F1.",
    )
    args = parser.parse_args()

    cfg = BuildConfig(project_root=PROJECT_ROOT)
    pipeline = RealQAPipeline(cfg)
    eval_set = load_eval_set(Path(args.eval_set))
    thresholds = [args.threshold]
    if args.sweep:
        thresholds = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]

    reports = []
    for threshold in thresholds:
        examples = [evaluate_example(pipeline, item, threshold) for item in eval_set]
        summary = summarize_examples(examples, threshold)
        reports.append({"summary": summary, "examples": examples})

    best_report = max(reports, key=lambda item: item["summary"]["qa_f1"])
    output = {
        "selected_threshold": args.threshold,
        "best_threshold_by_qa_f1": best_report["summary"]["threshold"],
        "best_summary": best_report["summary"],
        "all_summaries": [report["summary"] for report in reports],
        "examples": best_report["examples"],
    }

    reports_dir = cfg.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "evaluation_report.json"
    markdown_report_path = reports_dir / "evaluation_report.md"
    report_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown_report_path.write_text(build_markdown_report(output), encoding="utf-8")
    print(json.dumps(output["best_summary"], indent=2, ensure_ascii=False))
    print(f"\nSaved evaluation report to: {report_path}")
    print(f"Saved markdown report to: {markdown_report_path}")


if __name__ == "__main__":
    main()
