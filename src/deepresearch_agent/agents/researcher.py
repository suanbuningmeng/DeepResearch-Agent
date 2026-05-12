from __future__ import annotations

import json

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import Evidence, TaskNode


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
            "Return JSON with an evidences array."
        )
        raw = await self.llm.agenerate(prompt, prompt_type="researcher")
        data = json.loads(raw)

        return [
            Evidence(
                id=f"{task.id}_evidence_{index}",
                task_id=task.id,
                title=item["title"],
                content=item["content"],
                source_url=item.get("source_url"),
                confidence=item["confidence"],
                metadata={"task_name": task.name},
            )
            for index, item in enumerate(data["evidences"], start=1)
        ]
