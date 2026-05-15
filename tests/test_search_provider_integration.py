import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepresearch_agent.search.mock_provider import MockSearchProvider
from deepresearch_agent.search.schemas import SearchResult
from scripts import run_demo as run_demo_module


def test_run_demo_mock_provider_still_works(tmp_path: Path) -> None:
    async def run() -> None:
        trace = await run_demo_module.run_demo(
            question="long context LLM evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
            enable_web_search=True,
            search_provider="mock",
        )

        assert trace["search_stats"]["provider"] == "mock"

    asyncio.run(run())


class FakeTavilyProvider(MockSearchProvider):
    provider_name = "tavily"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                id="fake_tavily_1",
                query_id="q",
                title="Long context evaluation source",
                url="https://fake.tavily/source",
                snippet="Long context evaluation includes retrieval, coherence, and citation quality.",
                rank=1,
                provider="tavily",
                domain="fake.tavily",
            )
        ]


def test_run_demo_fake_tavily_provider_via_factory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_demo_module, "create_search_provider", lambda **kwargs: FakeTavilyProvider())

    async def run() -> None:
        trace = await run_demo_module.run_demo(
            question="long context LLM evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
            enable_web_search=True,
            search_provider="tavily",
        )

        assert trace["search_stats"]["provider"] == "tavily"

    asyncio.run(run())


class ErrorProvider(MockSearchProvider):
    provider_name = "tavily"

    def search(self, query: str, top_k: int = 5):
        raise RuntimeError("provider failed without secret")


def test_provider_errors_are_recorded_without_breaking(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_demo_module, "create_search_provider", lambda **kwargs: ErrorProvider())

    async def run() -> None:
        trace = await run_demo_module.run_demo(
            question="long context LLM evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
            enable_web_search=True,
            search_provider="tavily",
        )

        assert trace["search_stats"]["provider_errors"]
        assert trace["search_stats"]["fallback_used"] is True

    asyncio.run(run())
