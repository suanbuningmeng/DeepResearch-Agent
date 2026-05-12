from __future__ import annotations

from deepresearch_agent.schemas import TaskNode, TaskState


FINISHED_STATES = {
    TaskState.SUCCEEDED,
    TaskState.FAILED,
    TaskState.TIMEOUT,
    TaskState.DEGRADED,
    TaskState.CANCELLED,
}

DEPENDENCY_DONE_STATES = {
    TaskState.SUCCEEDED,
    TaskState.DEGRADED,
}


class TaskDAG:
    """In-memory DAG of research tasks and their dependencies."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskNode] = {}

    def add_task(self, task: TaskNode) -> None:
        if task.id in self._tasks:
            raise ValueError(f"Task already exists: {task.id}")
        self._tasks[task.id] = task

    def get_task(self, task_id: str) -> TaskNode:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"Unknown task id: {task_id}") from exc

    def list_tasks(self) -> list[TaskNode]:
        return list(self._tasks.values())

    def get_ready_tasks(self) -> list[TaskNode]:
        return [
            task
            for task in self._tasks.values()
            if task.state == TaskState.READY
            or (
                task.state == TaskState.PENDING
                and all(
                    self.get_task(dependency_id).state in DEPENDENCY_DONE_STATES
                    for dependency_id in task.dependencies
                )
            )
        ]

    def mark_task_state(self, task_id: str, state: TaskState) -> None:
        self.get_task(task_id).state = state

    def all_finished(self) -> bool:
        return all(task.state in FINISHED_STATES for task in self._tasks.values())

    def has_failed(self) -> bool:
        return any(task.state in {TaskState.FAILED, TaskState.TIMEOUT} for task in self._tasks.values())
