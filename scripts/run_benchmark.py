from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepresearch_agent.evaluation import EvaluationConfig, EvaluationRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DeepResearch-Agent benchmark.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", default="eval_outputs")
    parser.add_argument("--backend", default="mock")
    parser.add_argument("--mode", default="dag", choices=["linear", "dag"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--enable-context-compression", action="store_true", default=False)
    parser.add_argument("--enable-conflict-detection", action="store_true", default=False)
    parser.add_argument("--enable-red-blue", action="store_true", default=False)
    parser.add_argument("--memory-backend", default="memory", choices=["memory", "sqlite"])
    parser.add_argument("--max-concurrency", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(
        EvaluationRunner().run(
            EvaluationConfig(
                benchmark_path=args.benchmark,
                output_dir=args.output_dir,
                backend=args.backend,
                mode=args.mode,
                limit=args.limit,
                domain=args.domain,
                enable_context_compression=args.enable_context_compression,
                enable_conflict_detection=args.enable_conflict_detection,
                enable_red_blue=args.enable_red_blue,
                memory_backend=args.memory_backend,
                max_concurrency=args.max_concurrency,
            )
        )
    )
    print(summary.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
