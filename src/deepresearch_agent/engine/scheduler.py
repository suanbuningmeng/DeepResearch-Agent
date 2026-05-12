from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from deepresearch_agent.agents import ResearcherAgent
from deepresearch_agent.engine.dag import TaskDAG
from deepresearch_agent.engine.state_machine import TaskStateMachine
from deepresearch_agent.engine.timeout import run_with_timeout
from deepresearch_agent.memory import MemoryStore
from deepresearch_agent.schemas import Evidence, TaskNode, TaskState


class DAGTaskScheduler:
    """Execute ready research tasks from a TaskDAG with bounded asyncio concurrency."""

    def __init__(
        self,
        researcher_agent: ResearcherAgent,
        memory_store: MemoryStore,
        max_concurrency: int = 3,
        task_timeout_seconds: int = 30,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        if task_timeout_seconds < 1:
            raise ValueError("task_timeout_seconds must be at least 1.")

        self.researcher_agent = researcher_agent
        self.memory_store = memory_store
        self.max_concurrency = max_concurrency
        self.task_timeout_seconds = task_timeout_seconds
        self.state_machine = TaskStateMachine()
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def run(self, dag: TaskDAG) -> dict[str, Any]:
        trace: dict[str, Any] = {
            "max_concurrency": self.max_concurrency,
            "task_state_transitions": [],
            "task_start_time": {},
            "task_end_time": {},
            "task_duration_seconds": {},
            "retry_count": {},
            "degraded_tasks": [],
            "final_task_states": {},
            "execution_steps": [],
        }

        while not dag.all_finished():
            ready_tasks = dag.get_ready_tasks()
            if not ready_tasks:
                self._cancel_unreachable_pending_tasks(dag, trace)
                break

            for task in ready_tasks:
                if task.state == TaskState.PENDING:
                    self._transition(task, TaskState.READY, trace)

            await asyncio.gather(*(self._run_task(task, trace) for task in ready_tasks))

        trace["final_task_states"] = {
            task.id: task.state.value for task in dag.list_tasks()
        }
        trace["retry_count"] = {
            task.id: task.retry_count for task in dag.list_tasks()
        }
        return trace

    async def _run_task(self, task: TaskNode, trace: dict[str, Any]) -> None:
        async with self._semaphore:
            self._transition(task, TaskState.RUNNING, trace)
            trace["task_start_time"][task.id] = _utc_now()
            started_at = perf_counter()

            try:
                timeout_seconds = min(task.timeout_seconds, self.task_timeout_seconds)
                evidences = await run_with_timeout(
                    self.researcher_agent.research(task),
                    timeout_seconds,
                )
            except TimeoutError:
                task.error = "Task timed out."
                self._finish_task_timing(task, started_at, trace)
                self._transition(task, TaskState.TIMEOUT, trace)
                self._apply_degraded_fallback(task, trace, reason="timeout")
                return
            except Exception as exc:
                task.error = str(exc)
                self._finish_task_timing(task, started_at, trace)
                self._transition(task, TaskState.FAILED, trace)
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    trace["retry_count"][task.id] = task.retry_count
                    self._transition(task, TaskState.READY, trace)
                    trace["execution_steps"].append(
                        {
                            "step": "research_retry_scheduled",
                            "task_id": task.id,
                            "retry_count": task.retry_count,
                        }
                    )
                    return
                self._apply_degraded_fallback(task, trace, reason="failure")
                return

            self.memory_store.add_evidences(evidences)
            task.output = {"evidence_ids": [evidence.id for evidence in evidences]}
            self._finish_task_timing(task, started_at, trace)
            self._transition(task, TaskState.SUCCEEDED, trace)
            trace["execution_steps"].append(
                {
                    "step": "research",
                    "task_id": task.id,
                    "state": task.state.value,
                    "evidence_count": len(evidences),
                    "retry_count": task.retry_count,
                }
            )

    def _transition(
        self,
        task: TaskNode,
        to_state: TaskState,
        trace: dict[str, Any],
    ) -> None:
        from_state = task.state
        self.state_machine.transition(task, to_state)
        trace["task_state_transitions"].append(
            {
                "task_id": task.id,
                "from": from_state.value,
                "to": to_state.value,
                "timestamp": _utc_now(),
                "retry_count": task.retry_count,
            }
        )

    def _apply_degraded_fallback(
        self,
        task: TaskNode,
        trace: dict[str, Any],
        reason: str,
    ) -> None:
        evidence = Evidence(
            id=f"{task.id}_degraded_evidence",
            task_id=task.id,
            title=f"Degraded evidence for {task.name}",
            content=(
                "This evidence was generated by degraded fallback because the "
                "original research task failed or timed out."
            ),
            source_url=f"mock://degraded/{task.id}",
            confidence=0.5,
            metadata={"degraded": True, "reason": reason},
        )
        self.memory_store.add_evidence(evidence)
        task.output = {"evidence_ids": [evidence.id], "degraded": True}
        self._transition(task, TaskState.DEGRADED, trace)
        trace["degraded_tasks"].append(task.id)
        trace["execution_steps"].append(
            {
                "step": "research_degraded",
                "task_id": task.id,
                "state": task.state.value,
                "reason": reason,
                "evidence_count": 1,
                "retry_count": task.retry_count,
            }
        )

    def _finish_task_timing(
        self,
        task: TaskNode,
        started_at: float,
        trace: dict[str, Any],
    ) -> None:
        trace["task_end_time"][task.id] = _utc_now()
        trace["task_duration_seconds"][task.id] = round(perf_counter() - started_at, 6)

    def _cancel_unreachable_pending_tasks(
        self,
        dag: TaskDAG,
        trace: dict[str, Any],
    ) -> None:
        for task in dag.list_tasks():
            if task.state in {TaskState.PENDING, TaskState.READY}:
                self._transition(task, TaskState.CANCELLED, trace)
                trace["execution_steps"].append(
                    {
                        "step": "research_cancelled",
                        "task_id": task.id,
                        "state": task.state.value,
                    }
                )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
