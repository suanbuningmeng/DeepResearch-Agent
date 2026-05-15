from __future__ import annotations

import asyncio

from deepresearch_agent.agents import ResearcherAgent
from deepresearch_agent.agents.researcher import coerce_evidence_data
from deepresearch_agent.llm import BaseLLM
from deepresearch_agent.schemas import TaskNode, TaskState


class BlankLLM(BaseLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return ""


def _task() -> TaskNode:
    return TaskNode(
        id="task_1",
        name="Research task",
        description="Analyze long-context evaluation.",
        agent_type="researcher",
        state=TaskState.READY,
    )


def test_researcher_coerces_findings_key() -> None:
    evidences = coerce_evidence_data(
        {
            "findings": [
                {
                    "title": "Finding",
                    "summary": "Long-context evaluation needs retrieval and synthesis tests.",
                    "source": "https://example.org/source",
                    "confidence_score": "0.8",
                }
            ]
        },
        _task(),
        raw_output="",
    )

    assert evidences[0].title == "Finding"
    assert evidences[0].content == "Long-context evaluation needs retrieval and synthesis tests."
    assert evidences[0].source_url == "https://example.org/source"
    assert evidences[0].confidence == 0.8


def test_researcher_coerces_claim_and_source_variants() -> None:
    evidences = coerce_evidence_data(
        [
            {
                "claim": "Position bias affects long-context evaluation.",
                "url": "https://example.org/position",
            }
        ],
        _task(),
        raw_output="",
    )

    assert evidences[0].content == "Position bias affects long-context evaluation."
    assert evidences[0].title.startswith("Position bias")
    assert evidences[0].source_url == "https://example.org/position"


def test_researcher_prefixes_duplicate_llm_ids_and_sanitizes_unverified_numbers() -> None:
    evidences = coerce_evidence_data(
        {
            "evidences": [
                {
                    "id": "evidence_1",
                    "title": "Unverified case study",
                    "content": "The system achieved a 70% bandwidth reduction and 55% latency reduction.",
                    "source_url": None,
                    "confidence": 0.9,
                }
            ]
        },
        _task(),
        raw_output="",
    )

    assert evidences[0].id == "task_1_evidence_1"
    assert evidences[0].confidence == 0.5
    assert "70%" not in evidences[0].content
    assert "55%" not in evidences[0].content
    assert evidences[0].metadata["citation_validation_status"] == "unverified"


def test_researcher_fallback_content_is_not_empty() -> None:
    async def run() -> None:
        evidences = await ResearcherAgent(BlankLLM()).research(_task())

        assert evidences
        assert all(evidence.content for evidence in evidences)

    asyncio.run(run())
