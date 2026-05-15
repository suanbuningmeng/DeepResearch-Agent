from __future__ import annotations

import asyncio
import json

from deepresearch_agent.agents import ResearcherAgent
from deepresearch_agent.llm import BaseLLM
from deepresearch_agent.schemas import TaskNode, TaskState


class ResearcherRepairLLM(BaseLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        prompt_type = str(kwargs.get("prompt_type") or "")
        if prompt_type == "researcher":
            return "Evidence one says repair is needed, but this is not JSON."
        if prompt_type == "researcher_output_task_1_json_repair":
            return json.dumps(
                {
                    "evidences": [
                        {
                            "title": "Repaired evidence",
                            "content": "The repaired evidence has real content.",
                            "source_url": None,
                            "confidence": 0.8,
                        }
                    ]
                }
            )
        return "{}"


class EmptyFallbackLLM(BaseLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return "   "


def _task() -> TaskNode:
    return TaskNode(
        id="task_1",
        name="Task",
        description="Task description fallback content.",
        agent_type="researcher",
        state=TaskState.READY,
    )


def test_researcher_uses_repair_retry_for_evidence_json() -> None:
    async def run() -> None:
        evidences = await ResearcherAgent(ResearcherRepairLLM()).research(_task())

        assert len(evidences) == 1
        assert evidences[0].title == "Repaired evidence"
        assert evidences[0].content == "The repaired evidence has real content."
        assert evidences[0].metadata["json_parse_success"] is False
        assert evidences[0].metadata["repair_attempted"] is True
        assert evidences[0].metadata["repair_success"] is True
        assert evidences[0].metadata["fallback_parse"] is False

    asyncio.run(run())


def test_researcher_fallback_does_not_emit_empty_content() -> None:
    async def run() -> None:
        evidences = await ResearcherAgent(EmptyFallbackLLM()).research(_task())

        assert evidences
        assert all(evidence.content for evidence in evidences)
        assert evidences[0].content == "Task description fallback content."
        assert evidences[0].metadata["fallback_parse"] is True

    asyncio.run(run())
