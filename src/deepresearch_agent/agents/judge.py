from __future__ import annotations

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import Evidence, JudgeScore
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


class JudgeAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def judge(
        self,
        question: str,
        report: str,
        evidences: list[Evidence],
    ) -> JudgeScore:
        """Score a report against the question and evidence using the configured LLM."""
        prompt = (
            "You are a judge agent. Score this report from 0 to 100.\n"
            f"Question: {question}\n"
            f"Report:\n{report}\n"
            f"Evidence count: {len(evidences)}\n"
            "Return only JSON with factuality, coverage, reasoning_depth, citation_quality, clarity, overall, and comments."
        )
        raw = await self.llm.agenerate(prompt, prompt_type="judge")
        data = safe_json_loads(strip_thinking(raw))
        return JudgeScore.model_validate(
            {
                "factuality": _coerce_score(data.get("factuality", 0)),
                "coverage": _coerce_score(data.get("coverage", 0)),
                "reasoning_depth": _coerce_score(data.get("reasoning_depth", 0)),
                "citation_quality": _coerce_score(data.get("citation_quality", 0)),
                "clarity": _coerce_score(data.get("clarity", 0)),
                "overall": _coerce_score(data.get("overall", 0)),
                "comments": str(data.get("comments") or ""),
            }
        )


def _coerce_score(value: object) -> int:
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))
