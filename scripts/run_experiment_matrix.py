from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepresearch_agent.evaluation import EvaluationConfig, EvaluationRunner


MATRIX = {
    "baseline_mock": {},
    "compression_mock": {"enable_context_compression": True},
    "compression_conflict_mock": {"enable_context_compression": True, "enable_conflict_detection": True},
    "full_pipeline_mock": {"enable_context_compression": True, "enable_conflict_detection": True, "enable_red_blue": True},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed mock experiment matrix.")
    parser.add_argument("--benchmark", default="benchmarks/researchbench.jsonl")
    parser.add_argument("--output-dir", default="eval_outputs")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    root = Path(args.output_dir) / f"matrix_{int(time.time())}"

    async def run_all() -> None:
        for name, overrides in MATRIX.items():
            summary = await EvaluationRunner().run(
                EvaluationConfig(
                    benchmark_path=args.benchmark,
                    output_dir=str(root / name),
                    backend="mock",
                    limit=args.limit,
                    **overrides,
                )
            )
            print(f"{name}: rule={summary.average_rule_overall}, judge={summary.average_judge_overall}")

    asyncio.run(run_all())


if __name__ == "__main__":
    main()
