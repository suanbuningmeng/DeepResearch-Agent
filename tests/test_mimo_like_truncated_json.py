from __future__ import annotations

import asyncio

from deepresearch_agent.agents import ResearcherAgent
from deepresearch_agent.llm import BaseLLM
from deepresearch_agent.schemas import TaskNode, TaskState


class TruncatedJsonLLM(BaseLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        prompt_type = str(kwargs.get("prompt_type") or "")
        if prompt_type == "researcher":
            return """
{
  "evidences": [
    {
      "title": "Complete salvaged evidence",
      "content": "This complete object should be salvaged from a truncated JSON array.",
      "source_url": null,
      "confidence": 0.74
    },
    {
      "title": "Incomplete evidence",
      "content":
"""
        return "not json"


def test_researcher_salvages_mimo_like_truncated_json() -> None:
    async def run() -> None:
        task = TaskNode(
            id="task_1",
            name="Task",
            description="Task description.",
            agent_type="researcher",
            state=TaskState.READY,
        )

        evidences = await ResearcherAgent(TruncatedJsonLLM()).research(task)

        assert len(evidences) >= 1
        assert evidences[0].title == "Complete salvaged evidence"
        assert evidences[0].content
        assert evidences[0].metadata["partial_json_salvaged"] is True
        assert evidences[0].metadata["schema_coercion_success"] is True

    asyncio.run(run())
