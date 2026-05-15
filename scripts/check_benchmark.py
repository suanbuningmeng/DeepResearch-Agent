from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from deepresearch_agent.evaluation import load_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a benchmark JSONL file.")
    parser.add_argument("--benchmark", required=True)
    args = parser.parse_args()
    suite = load_benchmark(args.benchmark)
    print(f"OK: {suite.name} contains {len(suite.examples)} examples.")


if __name__ == "__main__":
    main()
