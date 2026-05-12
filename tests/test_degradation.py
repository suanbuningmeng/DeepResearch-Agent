from deepresearch_agent.engine import DegradationManager
from deepresearch_agent.schemas import TaskNode, TaskState


def test_degradation_manager_creates_fallback_evidence_and_record() -> None:
    task = TaskNode(
        id="task_9",
        name="Recover missing evidence",
        description="Test degraded fallback creation.",
        agent_type="researcher",
        state=TaskState.TIMEOUT,
    )
    manager = DegradationManager()

    evidence = manager.create_fallback_evidence(task, reason="timeout")
    record = manager.create_record(task, reason="timeout", fallback_evidence=evidence)

    assert evidence.id == "task_9_degraded_evidence"
    assert evidence.title == "Degraded evidence for Recover missing evidence"
    assert evidence.source_url == "mock://degraded/task_9"
    assert evidence.confidence == 0.5
    assert record.task_id == "task_9"
    assert record.original_state == TaskState.TIMEOUT
    assert record.fallback_evidence_id == evidence.id
