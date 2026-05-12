from __future__ import annotations

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import TaskNode, TaskState
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


class PlannerAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def plan(self, question: str) -> list[TaskNode]:
        """Split a user question into structured research subtasks."""
        prompt = (
            "You are a planner agent. Split the research question into 3-5 subtasks.\n"
            f"Question: {question}\n"
            "Return only JSON with a subtasks array. Each subtask must include id, name, and description."
        )
        raw = await self.llm.agenerate(prompt, prompt_type="planner")
        data = safe_json_loads(strip_thinking(raw))

        return [
            TaskNode(
                id=_normalize_task_id(item.get("id", index)),
                name=str(item.get("name") or f"Research subtask {index}"),
                description=str(item.get("description") or item.get("question") or item.get("task") or ""),
                agent_type="researcher",
                dependencies=[],
                state=TaskState.READY,
                input={"question": question},
            )
            for index, item in enumerate(data["subtasks"], start=1)
        ]


def _normalize_task_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return "task_unknown"
    if text.startswith("task_"):
        return text
    return f"task_{text}"
