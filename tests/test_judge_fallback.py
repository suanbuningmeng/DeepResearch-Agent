from __future__ import annotations

import asyncio

from deepresearch_agent.agents import JudgeAgent
from deepresearch_agent.llm import BaseLLM


class StaticTextLLM(BaseLLM):
    def __init__(self, text: str) -> None:
        self.text = text

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return self.text


def test_judge_agent_parses_standard_json() -> None:
    async def run() -> None:
        judge = JudgeAgent(
            StaticTextLLM(
                '{"factuality": 88, "coverage": 85, "reasoning_depth": 84, "citation_quality": 80, "clarity": 90, "overall": 86, "comments": "ok"}'
            )
        )

        score = await judge.judge("Question?", "Report", [])

        assert score.overall == 86
        assert judge.last_stats["json_parse_success"] is True
        assert judge.last_stats["fallback_used"] is False

    asyncio.run(run())


def test_judge_agent_parses_fenced_json() -> None:
    async def run() -> None:
        judge = JudgeAgent(
            StaticTextLLM(
                """```json
{"factuality": 81, "coverage": 82, "reasoning_depth": 83, "citation_quality": 84, "clarity": 85, "overall": 83, "comments": "ok"}
```"""
            )
        )

        score = await judge.judge("Question?", "Report", [])

        assert score.overall == 83
        assert judge.last_stats["json_parse_success"] is True

    asyncio.run(run())


def test_judge_agent_extracts_scores_from_natural_language() -> None:
    async def run() -> None:
        judge = JudgeAgent(
            StaticTextLLM(
                "Factuality: 85\nCoverage score is 80\nReasoning depth: 78\nCitation quality: 70\nOverall: 79"
            )
        )

        score = await judge.judge("Question?", "Report", [])

        assert score.factuality == 85
        assert score.coverage == 80
        assert score.overall == 79
        assert judge.last_stats["json_parse_success"] is False
        assert judge.last_stats["extracted_from_text"] is True
        assert judge.last_stats["fallback_used"] is False

    asyncio.run(run())


def test_judge_agent_uses_fallback_for_unusable_output() -> None:
    async def run() -> None:
        judge = JudgeAgent(StaticTextLLM("I cannot provide a structured score, but the report looks acceptable."))

        score = await judge.judge("Question?", "Tiny report", [])

        assert score.overall > 0
        assert "Fallback judge score was used" in score.comments
        assert judge.last_stats["json_parse_success"] is False
        assert judge.last_stats["extracted_from_text"] is False
        assert judge.last_stats["fallback_used"] is True

    asyncio.run(run())
