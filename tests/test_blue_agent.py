import asyncio
import json

from deepresearch_agent.agents.blue import BlueAgent
from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.red_blue.schemas import (
    RedCritique,
    RedIssue,
    RedIssueSeverity,
    RedIssueType,
    RepairActionType,
)
from deepresearch_agent.schemas import Evidence


class StaticLLM(BaseLLM):
    def __init__(self, text: str) -> None:
        self.text = text

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return self.text


def critique() -> RedCritique:
    return RedCritique(
        round_id=1,
        issues=[
            RedIssue(
                id="issue_1",
                issue_type=RedIssueType.CITATION_QUALITY,
                severity=RedIssueSeverity.HIGH,
                location="Evidence Summary",
                description="Citation needs verification.",
                evidence_ids=["e1"],
                suggested_action=RepairActionType.VERIFY,
                rationale="Citation quality is low.",
            )
        ],
        high_severity_count=1,
        summary="Needs repair.",
    )


def evidence() -> Evidence:
    return Evidence(
        id="e1",
        task_id="task_1",
        title="Evidence",
        content="Evidence content.",
        source_url="mock://source",
        confidence=0.8,
    )


def test_blue_agent_parses_valid_json_repair() -> None:
    async def run() -> None:
        agent = BlueAgent(
            StaticLLM(
                json.dumps(
                    {
                        "revised_report": "# Report\n\n## Abstract\nA.\n\n## Key Findings\n- K.\n\n## Evidence Summary\nE.\n\n## Limitations\nL.\n\n## Conclusion\nC.",
                        "patches": [
                            {
                                "id": "patch_1",
                                "action_type": "MODIFY",
                                "target_location": "Conclusion",
                                "before_text": "old",
                                "after_text": "new",
                                "evidence_ids": ["e1"],
                                "rationale": "Improve caution.",
                                "applied": True,
                            }
                        ],
                        "summary": "Repaired.",
                    }
                )
            )
        )

        result = await agent.repair("q", "report", [evidence()], critique())

        assert result.patches[0].applied is True
        assert result.revised_report.endswith("\n")
        assert agent.last_stats["blue_parse_success"] is True

    asyncio.run(run())


def test_blue_agent_fallback_appends_cautionary_note() -> None:
    async def run() -> None:
        agent = BlueAgent(StaticLLM("not json"))

        result = await agent.repair("q", "# Report\n\n## Conclusion\nDone.", [evidence()], critique())

        assert "## Cautionary Note" in result.revised_report
        assert result.patches[0].action_type == RepairActionType.VERIFY
        assert result.patches[0].applied is True
        assert agent.last_stats["blue_fallback_used"] is True

    asyncio.run(run())
