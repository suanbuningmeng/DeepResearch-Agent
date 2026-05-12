from __future__ import annotations

from deepresearch_agent.schemas import TaskNode, TaskState


class ReplannerAgent:
    """Create replacement research tasks for failed or degraded tasks."""

    async def replan(
        self,
        failed_tasks: list[TaskNode],
        existing_tasks: list[TaskNode],
        reason: str,
    ) -> list[TaskNode]:
        existing_ids = {task.id for task in existing_tasks}
        new_tasks: list[TaskNode] = []

        for failed_task in failed_tasks:
            candidates = [
                (
                    f"{failed_task.id}_replan_1",
                    f"Find alternative evidence for {failed_task.name}",
                    f"Find alternative mock evidence for the failed task: {failed_task.description}",
                ),
                (
                    f"{failed_task.id}_replan_2",
                    f"Summarize fallback findings for {failed_task.name}",
                    f"Summarize fallback findings after replan reason: {reason}",
                ),
            ]
            for task_id, name, description in candidates:
                if task_id in existing_ids:
                    continue
                existing_ids.add(task_id)
                new_tasks.append(
                    TaskNode(
                        id=task_id,
                        name=name,
                        description=description,
                        agent_type="researcher",
                        dependencies=list(failed_task.dependencies),
                        state=TaskState.PENDING,
                        input=dict(failed_task.input),
                        max_retries=failed_task.max_retries,
                        timeout_seconds=failed_task.timeout_seconds,
                    )
                )

        return new_tasks
