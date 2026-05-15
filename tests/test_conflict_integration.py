from pathlib import Path
import sys
import asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_demo import run_demo


def test_run_demo_with_conflict_detection_writes_conflict_stats(tmp_path: Path) -> None:
    async def run() -> None:
        trace = await run_demo(
            question="test long-context evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
            enable_conflict_detection=True,
        )

        assert "conflict_stats" in trace
        assert trace["conflict_stats"]["enabled"] is True
        assert trace["conflict_stats"]["input_evidence_count"] >= trace["conflict_stats"]["output_evidence_count"]
        assert (tmp_path / "demo_trace.json").exists()

    asyncio.run(run())


def test_run_demo_records_disabled_conflict_stats(tmp_path: Path) -> None:
    async def run() -> None:
        trace = await run_demo(
            question="test long-context evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
            enable_conflict_detection=False,
        )

        assert trace["conflict_stats"]["enabled"] is False

    asyncio.run(run())
