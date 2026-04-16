from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

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


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Run the real terminal QA pipeline.")
    parser.add_argument("question", help="Question to ask the QA system.")
    parser.add_argument("--threshold", type=float, default=0.01, help="Confidence threshold for answer vs no_answer.")
    parser.add_argument(
        "--style",
        choices=["auto", "extractive", "explanatory"],
        default="auto",
        help="Answering style. Use explanatory for grounded multi-chunk synthesis.",
    )
    args = parser.parse_args()

    cfg = BuildConfig(project_root=PROJECT_ROOT)
    pipeline = RealQAPipeline(cfg)
    result = pipeline.answer(args.question, threshold=args.threshold, style=args.style)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
