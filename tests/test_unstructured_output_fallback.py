import asyncio

from deepresearch_agent.agents import ResearcherAgent
from deepresearch_agent.llm import BaseLLM
from deepresearch_agent.schemas import TaskNode, TaskState


class BulletLLM(BaseLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return """
- Retrieval challenge: Models can locate facts but fail synthesis across long inputs.
- Benchmark issue: Synthetic probes do not always reflect realistic research tasks.
"""


def test_researcher_fallback_parses_bullet_output() -> None:
    async def run() -> None:
        task = TaskNode(
            id="task_1",
            name="Identify key challenges",
            description="Analyze challenges.",
            agent_type="researcher",
            state=TaskState.READY,
        )

        evidences = await ResearcherAgent(BulletLLM()).research(task)

        assert len(evidences) == 2
        assert evidences[0].source_url == "model://unstructured-output"
        assert evidences[0].confidence == 0.55
        assert evidences[0].metadata["fallback_parse"] is True

    asyncio.run(run())
