import asyncio
from pathlib import Path

from deepresearch_agent.evaluation.runner import EvaluationConfig, EvaluationRunner


def test_evaluation_runner_mock_limit_two_outputs_files(tmp_path: Path) -> None:
    async def run() -> None:
        summary = await EvaluationRunner().run(
            EvaluationConfig(
                benchmark_path="benchmarks/researchbench.jsonl",
                output_dir=str(tmp_path),
                backend="mock",
                limit=2,
                max_concurrency=1,
            )
        )

        assert summary.case_count == 2
        assert summary.success_count == 2
        run_dirs = list(tmp_path.iterdir())
        assert run_dirs
        run_dir = run_dirs[0]
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "results.jsonl").exists()
        assert (run_dir / "results.csv").exists()
        assert (run_dir / "summary.md").exists()

    asyncio.run(run())


def test_evaluation_runner_single_failure_does_not_stop(tmp_path: Path) -> None:
    async def run() -> None:
        summary = await EvaluationRunner().run(
            EvaluationConfig(
                benchmark_path="benchmarks/researchbench.jsonl",
                output_dir=str(tmp_path),
                backend="unknown",
                limit=2,
            )
        )

        assert summary.case_count == 2
        assert summary.success_count == 0
        assert summary.error_count == 2

    asyncio.run(run())
