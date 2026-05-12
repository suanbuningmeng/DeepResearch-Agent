from __future__ import annotations

from datetime import datetime, timezone

from deepresearch_agent.agents import ReplannerAgent
from deepresearch_agent.engine.dag import TaskDAG
from deepresearch_agent.schemas import ReplanEvent, TaskNode


class ReplanManager:
    """Coordinate dynamic replanning and insertion of replacement tasks."""

    def __init__(self, replanner_agent: ReplannerAgent) -> None:
        self.replanner_agent = replanner_agent

    async def create_replan_event(
        self,
        dag: TaskDAG,
        failed_tasks: list[TaskNode],
        reason: str,
        round_number: int,
    ) -> ReplanEvent:
        new_tasks = await self.replanner_agent.replan(
            failed_tasks=failed_tasks,
            existing_tasks=dag.list_tasks(),
            reason=reason,
        )
        for task in new_tasks:
            dag.add_task(task)

        return ReplanEvent(
            id=f"replan_{round_number}",
            reason=reason,
            failed_task_ids=[task.id for task in failed_tasks],
            new_task_ids=[task.id for task in new_tasks],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
