from __future__ import annotations

from deepresearch_agent.schemas import TaskNode, TaskState


class TaskStateMachine:
    """Validate and apply legal task state transitions."""

    _ALLOWED_TRANSITIONS: set[tuple[TaskState, TaskState]] = {
        (TaskState.PENDING, TaskState.READY),
        (TaskState.READY, TaskState.RUNNING),
        (TaskState.RUNNING, TaskState.SUCCEEDED),
        (TaskState.RUNNING, TaskState.FAILED),
        (TaskState.RUNNING, TaskState.TIMEOUT),
        (TaskState.RUNNING, TaskState.CANCELLED),
        (TaskState.FAILED, TaskState.READY),
        (TaskState.FAILED, TaskState.DEGRADED),
        (TaskState.TIMEOUT, TaskState.DEGRADED),
        (TaskState.READY, TaskState.CANCELLED),
        (TaskState.PENDING, TaskState.CANCELLED),
    }

    def can_transition(self, from_state: TaskState, to_state: TaskState) -> bool:
        return (from_state, to_state) in self._ALLOWED_TRANSITIONS

    def transition(self, task: TaskNode, to_state: TaskState) -> TaskNode:
        from_state = task.state
        if not self.can_transition(from_state, to_state):
            raise ValueError(f"Illegal task state transition: {from_state} -> {to_state}")
        task.state = to_state
        return task
