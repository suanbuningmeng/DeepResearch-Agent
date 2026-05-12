from __future__ import annotations

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import Evidence, TaskNode
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


class ResearcherAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def research(self, task: TaskNode) -> list[Evidence]:
        """Generate mock evidence items for a single research task."""
        prompt = (
            "You are a researcher agent. Generate concise mock evidence for this subtask.\n"
            f"Task ID: {task.id}\n"
            f"Task name: {task.name}\n"
            f"Task description: {task.description}\n"
            "Return only JSON with an evidences array. Each evidence must include title, content, source_url, and confidence."
        )
        raw = await self.llm.agenerate(prompt, prompt_type="researcher")
        data = safe_json_loads(strip_thinking(raw))

        return [
            Evidence(
                id=f"{task.id}_evidence_{index}",
                task_id=task.id,
                title=str(item.get("title") or f"Evidence {index} for {task.name}"),
                content=str(item.get("content") or item.get("summary") or ""),
                source_url=_optional_str(item.get("source_url") or item.get("url")),
                confidence=_coerce_confidence(item.get("confidence", 0.7)),
                metadata={"task_name": task.name},
            )
            for index, item in enumerate(data["evidences"], start=1)
        ]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.7
    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))
