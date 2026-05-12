import pytest

from deepresearch_agent.engine import TaskStateMachine
from deepresearch_agent.schemas import TaskNode, TaskState


def make_task(state: TaskState = TaskState.PENDING) -> TaskNode:
    return TaskNode(
        id="task_test",
        name="Test task",
        description="A task used to test state transitions.",
        agent_type="researcher",
        state=state,
    )


def test_state_machine_allows_legal_transitions() -> None:
    machine = TaskStateMachine()
    task = make_task()

    machine.transition(task, TaskState.READY)
    machine.transition(task, TaskState.RUNNING)
    machine.transition(task, TaskState.SUCCEEDED)

    assert task.state == TaskState.SUCCEEDED


def test_state_machine_allows_retry_and_degraded_paths() -> None:
    machine = TaskStateMachine()
    task = make_task(TaskState.RUNNING)

    machine.transition(task, TaskState.FAILED)
    machine.transition(task, TaskState.READY)
    machine.transition(task, TaskState.RUNNING)
    machine.transition(task, TaskState.TIMEOUT)
    machine.transition(task, TaskState.DEGRADED)

    assert task.state == TaskState.DEGRADED


def test_state_machine_rejects_illegal_transition() -> None:
    machine = TaskStateMachine()
    task = make_task()

    with pytest.raises(ValueError):
        machine.transition(task, TaskState.RUNNING)
