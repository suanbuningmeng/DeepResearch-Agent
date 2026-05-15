from __future__ import annotations

import asyncio
import json

from deepresearch_agent.agents import JudgeAgent
from deepresearch_agent.llm import BaseLLM


class JudgeRepairLLM(BaseLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        prompt_type = str(kwargs.get("prompt_type") or "")
        if prompt_type == "judge":
            return "The report is good but I will not provide numeric labels."
        if prompt_type == "judge_json_repair":
            return json.dumps(
                {
                    "factuality": 82,
                    "coverage": 80,
                    "reasoning_depth": 78,
                    "citation_quality": 72,
                    "clarity": 88,
                    "overall": 81,
                    "comments": "Repaired judge score.",
                }
            )
        return "{}"


class BrokenJudgeRepairLLM(BaseLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return "not json and no rubric scores"


def test_judge_uses_repair_retry_before_fallback() -> None:
    async def run() -> None:
        judge = JudgeAgent(JudgeRepairLLM())

        score = await judge.judge("Question?", "Report", [])

        assert score.overall == 81
        assert judge.last_stats["json_parse_success"] is False
        assert judge.last_stats["extracted_from_text"] is False
        assert judge.last_stats["repair_attempted"] is True
        assert judge.last_stats["repair_success"] is True
        assert judge.last_stats["fallback_used"] is False

    asyncio.run(run())


def test_judge_falls_back_when_repair_fails() -> None:
    async def run() -> None:
        judge = JudgeAgent(BrokenJudgeRepairLLM())

        score = await judge.judge("Question?", "Report", [])

        assert score.overall > 0
        assert judge.last_stats["repair_attempted"] is True
        assert judge.last_stats["repair_success"] is False
        assert judge.last_stats["fallback_used"] is True

    asyncio.run(run())
