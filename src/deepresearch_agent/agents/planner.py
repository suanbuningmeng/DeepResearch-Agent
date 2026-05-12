from __future__ import annotations

import json

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import TaskNode, TaskState


class PlannerAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def plan(self, question: str) -> list[TaskNode]:
        """Split a user question into structured research subtasks."""
        prompt = (
            "You are a planner agent. Split the research question into 3-5 subtasks.\n"
            f"Question: {question}\n"
            "Return JSON with a subtasks array."
        )
        raw = await self.llm.agenerate(prompt, prompt_type="planner")
        data = json.loads(raw)

        return [
            TaskNode(
                id=item["id"],
                name=item["name"],
                description=item["description"],
                agent_type="researcher",
                dependencies=[],
                state=TaskState.READY,
                input={"question": question},
            )
            for item in data["subtasks"]
        ]
