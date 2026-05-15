import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_demo import run_demo


def test_run_demo_with_mock_search_writes_search_stats(tmp_path: Path) -> None:
    async def run() -> None:
        trace = await run_demo(
            question="What are the main challenges of long-context LLM evaluation?",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
            enable_web_search=True,
            search_provider="mock",
        )

        assert trace["search_stats"]["enabled"] is True
        assert trace["search_stats"]["query_count"] > 0
        assert trace["search_stats"]["validated_citation_count"] > 0

    asyncio.run(run())


def test_run_demo_records_disabled_search_stats(tmp_path: Path) -> None:
    async def run() -> None:
        trace = await run_demo(
            question="test",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
            enable_web_search=False,
        )

        assert trace["search_stats"]["enabled"] is False

    asyncio.run(run())
