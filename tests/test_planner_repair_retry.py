from __future__ import annotations

import asyncio
import json

from deepresearch_agent.agents import PlannerAgent
from deepresearch_agent.llm import BaseLLM


class PlannerRepairLLM(BaseLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        prompt_type = str(kwargs.get("prompt_type") or "")
        if prompt_type == "planner":
            return "I will create tasks, but not in JSON."
        if prompt_type == "planner_json_repair":
            return json.dumps(
                {
                    "subtasks": [
                        {
                            "id": "task_1",
                            "name": "Repair task",
                            "description": "Repaired task description.",
                            "agent_type": "researcher",
                            "dependencies": [],
                        }
                    ]
                }
            )
        return "{}"


def test_planner_uses_repair_retry_before_fallback() -> None:
    async def run() -> None:
        planner = PlannerAgent(PlannerRepairLLM())

        tasks = await planner.plan("Question?")

        assert len(tasks) >= 3
        assert tasks[0].name == "Repair task"
        assert planner.last_stats["json_parse_success"] is False
        assert planner.last_stats["repair_attempted"] is True
        assert planner.last_stats["repair_success"] is True
        assert planner.last_stats["partial_repair_used"] is True
        assert planner.last_stats["fallback_used"] is False

    asyncio.run(run())
