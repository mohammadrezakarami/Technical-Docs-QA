from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def classify(example: dict) -> str:
    if not example.get("answerable", True) and example.get("no_answer_correct", 0.0) < 1.0:
        return "false_positive"
    if example.get("answerable", True) and example.get("hit_at_1", 0.0) < 1.0:
        return "retrieval_miss"
    if example.get("answerable", True) and example.get("f1", 0.0) < 1.0:
        return "reader_or_span_issue"
    return "ok"


def main() -> None:
    report_path = PROJECT_ROOT / "artifacts" / "real_qa" / "reports" / "evaluation_report.json"
    output_path = PROJECT_ROOT / "artifacts" / "real_qa" / "reports" / "error_analysis.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    examples = report["examples"]
    grouped: dict[str, list[dict]] = {"retrieval_miss": [], "reader_or_span_issue": [], "false_positive": [], "ok": []}
    for example in examples:
        grouped[classify(example)].append(example)

    lines = [
        "# Error Analysis",
        "",
        "## Counts",
        "",
        f"- Retrieval misses: `{len(grouped['retrieval_miss'])}`",
        f"- Reader/span issues: `{len(grouped['reader_or_span_issue'])}`",
        f"- False positives: `{len(grouped['false_positive'])}`",
        "",
    ]
    for section in ["retrieval_miss", "reader_or_span_issue", "false_positive"]:
        lines.extend([f"## {section}", ""])
        if not grouped[section]:
            lines.append("- None")
            lines.append("")
            continue
        for example in grouped[section][:10]:
            top = example["evidence"][0] if example.get("evidence") else {}
            lines.extend([
                f"### {example['id']}",
                f"- Question: {example['question']}",
                f"- Prediction: `{example.get('prediction', '')}`",
                f"- Hit@1: `{example.get('hit_at_1', 0.0):.4f}`",
                f"- F1: `{example.get('f1', 0.0):.4f}`",
                f"- Top URL: {top.get('source_url', '')}",
                "",
            ])
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
