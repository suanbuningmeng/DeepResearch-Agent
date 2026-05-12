from __future__ import annotations

import json

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import Evidence, JudgeScore


class JudgeAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def judge(
        self,
        question: str,
        report: str,
        evidences: list[Evidence],
    ) -> JudgeScore:
        prompt = (
            "You are a judge agent. Score this report from 0 to 100.\n"
            f"Question: {question}\n"
            f"Report:\n{report}\n"
            f"Evidence count: {len(evidences)}\n"
            "Return JSON with factuality, coverage, reasoning_depth, citation_quality, clarity, overall, and comments."
        )
        raw = await self.llm.agenerate(prompt, prompt_type="judge")
        return JudgeScore.model_validate(json.loads(raw))
