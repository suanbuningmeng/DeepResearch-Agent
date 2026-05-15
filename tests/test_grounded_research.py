import asyncio

from deepresearch_agent.llm.mock_client import MockLLM
from deepresearch_agent.schemas import TaskNode
from deepresearch_agent.search import CitationValidator, GroundedResearchBuilder, MockSearchProvider, WebFetcher
from deepresearch_agent.search.provider import BaseSearchProvider
from deepresearch_agent.search.schemas import SearchResult


def test_grounded_research_generates_validated_evidence() -> None:
    async def run() -> None:
        task = TaskNode(
            id="task_1",
            name="Identify key challenges",
            description="Analyze long-context LLM evaluation.",
            agent_type="researcher",
            input={"question": "What are long-context LLM evaluation challenges?"},
        )
        builder = GroundedResearchBuilder(MockSearchProvider(), WebFetcher(), CitationValidator())

        evidences, stats = await builder.build_evidence_for_task(
            question="What are long-context LLM evaluation challenges?",
            task=task,
            llm=MockLLM(),
        )

        assert evidences
        assert evidences[0].metadata["grounded"] is True
        assert "citation_validation_status" in evidences[0].metadata
        assert evidences[0].source_url.startswith("https://research.example.org/")
        assert stats.enabled is True
        assert stats.validated_citation_count == len(evidences)

    asyncio.run(run())


class PaperLikeProvider(BaseSearchProvider):
    provider_name = "paper"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return [
            SearchResult(
                id="arxiv_2307.03172v1",
                query_id=query,
                title="Lost in the Middle",
                url="http://arxiv.org/abs/2307.03172v1",
                snippet="Long context evaluation must test whether models use information in the middle of long contexts.",
                rank=1,
                provider="paper",
                domain="arxiv.org",
                metadata={
                    "paper_id": "2307.03172v1",
                    "pdf_url": "http://arxiv.org/pdf/2307.03172v1",
                    "authors": ["Nelson F. Liu"],
                    "source_type": "paper",
                },
            )
        ]


def test_grounded_research_preserves_paper_source_metadata() -> None:
    async def run() -> None:
        task = TaskNode(
            id="task_1",
            name="Identify benchmark challenges",
            description="Analyze long-context LLM evaluation.",
            agent_type="researcher",
            input={"question": "What are long-context LLM evaluation challenges?"},
        )
        builder = GroundedResearchBuilder(PaperLikeProvider(), WebFetcher(), CitationValidator())

        evidences, stats = await builder.build_evidence_for_task(
            question="What are long-context LLM evaluation challenges?",
            task=task,
            llm=MockLLM(),
        )

        assert stats.provider == "paper"
        assert evidences[0].metadata["source_type"] == "paper"
        assert evidences[0].metadata["source_metadata"]["paper_id"] == "2307.03172v1"

    asyncio.run(run())


class EmptyProvider(BaseSearchProvider):
    provider_name = "arxiv"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        return []


def test_grounded_research_empty_results_return_no_fake_grounded_evidence() -> None:
    async def run() -> None:
        task = TaskNode(
            id="task_1",
            name="Identify benchmark challenges",
            description="Analyze long-context LLM evaluation.",
            agent_type="researcher",
            input={"question": "What are long-context LLM evaluation challenges?"},
        )
        builder = GroundedResearchBuilder(EmptyProvider(), WebFetcher(), CitationValidator())

        evidences, stats = await builder.build_evidence_for_task(
            question="What are long-context LLM evaluation challenges?",
            task=task,
            llm=MockLLM(),
        )

        assert evidences == []
        assert stats.provider == "arxiv"
        assert stats.result_count == 0
        assert stats.validated_citation_count == 0
        assert stats.fallback_used is True

    asyncio.run(run())


def test_grounded_fallback_uses_short_clean_snippets() -> None:
    task = TaskNode(
        id="task_1",
        name="Identify benchmark challenges",
        description="Analyze long-context LLM evaluation.",
        agent_type="researcher",
    )
    result = SearchResult(
        id="arxiv_1",
        query_id="long context",
        title="Long Context Benchmark",
        url="http://arxiv.org/abs/1",
        snippet=("Long context evaluation needs benchmark tasks that test retrieval, reasoning, and faithfulness. " * 10)
        + " Authors: A, B | Published: 2025-01-01 | Category: cs.CL",
        rank=1,
        provider="arxiv",
        domain="arxiv.org",
        metadata={"source_type": "paper"},
    )
    builder = GroundedResearchBuilder(PaperLikeProvider(), WebFetcher(), CitationValidator())

    evidences = builder._fallback_evidences(task, [result])

    assert len(evidences[0].content) <= 421
    assert "Authors:" not in evidences[0].content
    assert evidences[0].confidence < 0.6
