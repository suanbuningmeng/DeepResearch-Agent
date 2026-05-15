import asyncio
import json

from deepresearch_agent.agents.red import RedAgent
from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.red_blue.schemas import RedIssueType
from deepresearch_agent.schemas import Evidence


class StaticLLM(BaseLLM):
    def __init__(self, text: str) -> None:
        self.text = text

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return self.text


def evidence(source_url: str | None = "mock://source") -> Evidence:
    return Evidence(
        id="e1",
        task_id="task_1",
        title="Evidence",
        content="Evidence content.",
        source_url=source_url,
        confidence=0.8,
    )


def test_red_agent_parses_valid_json_critique() -> None:
    async def run() -> None:
        agent = RedAgent(
            StaticLLM(
                json.dumps(
                    {
                        "issues": [
                            {
                                "id": "issue_1",
                                "issue_type": "coverage",
                                "severity": "high",
                                "location": "Key Findings",
                                "description": "Coverage is thin.",
                                "evidence_ids": ["e1"],
                                "suggested_action": "ADD",
                                "rationale": "More evidence should be used.",
                            }
                        ],
                        "summary": "One issue.",
                    }
                )
            )
        )

        critique = await agent.critique("q", "report", [evidence()], round_id=1)

        assert critique.high_severity_count == 1
        assert critique.issues[0].issue_type == RedIssueType.COVERAGE
        assert agent.last_stats["red_parse_success"] is True

    asyncio.run(run())


def test_red_agent_fallback_on_non_json() -> None:
    async def run() -> None:
        agent = RedAgent(StaticLLM("not json"))

        critique = await agent.critique("q", "report", [evidence()], judge_score={"coverage": 75}, round_id=1)

        assert critique.issues
        assert agent.last_stats["red_fallback_used"] is True

    asyncio.run(run())


def test_red_agent_fallback_low_citation_quality_issue() -> None:
    async def run() -> None:
        agent = RedAgent(StaticLLM("not json"))

        critique = await agent.critique(
            "q",
            "report",
            [evidence("https://example.com/source")],
            judge_score={"citation_quality": 70},
            round_id=1,
        )

        assert any(issue.issue_type == RedIssueType.CITATION_QUALITY for issue in critique.issues)

    asyncio.run(run())
