from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.real_qa.build import build_real_artifacts
from src.real_qa.settings import BuildConfig


def main() -> None:
    cfg = BuildConfig(project_root=PROJECT_ROOT)
    manifest = build_real_artifacts(cfg)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
