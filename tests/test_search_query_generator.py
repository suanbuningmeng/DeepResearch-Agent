import asyncio
import json

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import TaskNode
from deepresearch_agent.search.query_generator import SearchQueryGenerator


class StaticLLM(BaseLLM):
    def __init__(self, text: str) -> None:
        self.text = text

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return self.text


def task() -> TaskNode:
    return TaskNode(id="task_1", name="Long context evaluation", description="Evaluate long context LLMs.", agent_type="researcher")


def semantic_task() -> TaskNode:
    return TaskNode(
        id="task_1",
        name="Semantic communication methods",
        description="Study task-oriented semantic communication with LLMs.",
        agent_type="researcher",
    )


def test_query_generator_parses_valid_json() -> None:
    async def run() -> None:
        generator = SearchQueryGenerator(StaticLLM(json.dumps({"queries": [{"query": "LongBench long context LLM evaluation", "intent": "benchmark"}]})))

        queries = await generator.generate_queries("q", task())

        assert queries[0].query == "LongBench long context LLM evaluation"
        assert generator.last_stats["parse_success"] is True

    asyncio.run(run())


def test_query_generator_fallback_on_non_json() -> None:
    async def run() -> None:
        generator = SearchQueryGenerator(StaticLLM("not json"))

        queries = await generator.generate_queries("What are long context challenges?", task())

        assert queries
        assert all(query.query for query in queries)
        assert generator.last_stats["fallback_used"] is True

    asyncio.run(run())


def test_query_generator_fallback_uses_paper_phrases_for_semantic_communication() -> None:
    async def run() -> None:
        generator = SearchQueryGenerator(StaticLLM("not json"))

        queries = await generator.generate_queries(
            "What are recent methods for task-oriented semantic communication with LLMs?",
            semantic_task(),
        )

        assert queries
        assert "semantic communication" in queries[0].query
        assert "large language model" in queries[0].query
        assert all("long-context" not in query.query.lower() for query in queries)

    asyncio.run(run())
