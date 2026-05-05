from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_python_script(script_name: str, *args: str) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), *args]
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return int(completed.returncode)


def print_status() -> None:
    reports_dir = PROJECT_ROOT / "artifacts" / "real_qa" / "reports"
    manifest_path = reports_dir / "manifest.json"
    eval_path = reports_dir / "evaluation_report.json"
    payload = {
        "project_root": str(PROJECT_ROOT),
        "manifest_exists": manifest_path.exists(),
        "evaluation_exists": eval_path.exists(),
        "artifacts_dir": str(PROJECT_ROOT / "artifacts" / "real_qa"),
    }
    if manifest_path.exists():
        payload["manifest_path"] = str(manifest_path)
    if eval_path.exists():
        payload["evaluation_path"] = str(eval_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified CLI for the real QA project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build", help="Build the real QA artifacts and local model snapshots.")
    subparsers.add_parser("generate-train", help="Generate domain-specific training pairs for retriever tuning.")
    subparsers.add_parser("train", help="Fine-tune the dense retriever on project-specific training data.")
    subparsers.add_parser("serve", help="Run the FastAPI server and browser interface.")

    ask_parser = subparsers.add_parser("ask", help="Ask a question to the real QA system.")
    ask_parser.add_argument("question", help="Question to ask.")
    ask_parser.add_argument("--threshold", type=float, default=0.01, help="Answer confidence threshold.")
    ask_parser.add_argument(
        "--style",
        choices=["auto", "extractive", "explanatory"],
        default="auto",
        help="Answering style.",
    )

    eval_parser = subparsers.add_parser("eval", help="Run evaluation on the gold benchmark set.")
    eval_parser.add_argument("--threshold", type=float, default=0.01, help="Answer confidence threshold.")
    eval_parser.add_argument("--sweep", action="store_true", help="Run a threshold sweep.")

    subparsers.add_parser("status", help="Print build/evaluation status.")
    subparsers.add_parser("analyze", help="Generate an error analysis report from the latest evaluation.")

    args = parser.parse_args()

    if args.command == "build":
        raise SystemExit(run_python_script("build_real_qa_artifacts.py"))
    if args.command == "generate-train":
        raise SystemExit(run_python_script("generate_domain_training_data.py"))
    if args.command == "train":
        raise SystemExit(run_python_script("fine_tune_dense_retriever.py"))
    if args.command == "serve":
        raise SystemExit(run_python_script("run_real_qa_api.py"))
    if args.command == "ask":
        raise SystemExit(
            run_python_script(
                "run_real_qa.py",
                "--threshold",
                str(args.threshold),
                "--style",
                str(args.style),
                args.question,
            )
        )
    if args.command == "eval":
        cmd_args = ["--threshold", str(args.threshold)]
        if args.sweep:
            cmd_args.append("--sweep")
        raise SystemExit(run_python_script("evaluate_real_qa.py", *cmd_args))
    if args.command == "status":
        print_status()
        return
    if args.command == "analyze":
        raise SystemExit(run_python_script("analyze_real_qa_errors.py"))


if __name__ == "__main__":
    main()
