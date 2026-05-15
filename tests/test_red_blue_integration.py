import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_demo import run_demo


def test_run_demo_with_red_blue_writes_stats(tmp_path: Path) -> None:
    async def run() -> None:
        trace = await run_demo(
            question="test long-context evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
            enable_red_blue=True,
        )

        assert "red_blue_stats" in trace
        assert trace["red_blue_stats"]["enabled"] is True
        assert trace["red_blue_stats"]["rounds_completed"] >= 1

    asyncio.run(run())


def test_run_demo_records_disabled_red_blue_stats(tmp_path: Path) -> None:
    async def run() -> None:
        trace = await run_demo(
            question="test long-context evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
            enable_red_blue=False,
        )

        assert trace["red_blue_stats"]["enabled"] is False

    asyncio.run(run())
