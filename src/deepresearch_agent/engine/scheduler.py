from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

from deepresearch_agent.agents import ReplannerAgent, ResearcherAgent
from deepresearch_agent.engine.dag import TaskDAG
from deepresearch_agent.engine.degradation import DegradationManager
from deepresearch_agent.engine.evidence_policy import EvidencePolicy
from deepresearch_agent.engine.replanner import ReplanManager
from deepresearch_agent.engine.state_machine import TaskStateMachine
from deepresearch_agent.engine.timeout import run_with_timeout
from deepresearch_agent.memory import MemoryStore
from deepresearch_agent.schemas import SchedulerConfig, TaskNode, TaskState


class DAGTaskScheduler:
    """Execute DAG research tasks with bounded concurrency, fallback, and replan support."""

    def __init__(
        self,
        researcher_agent: ResearcherAgent,
        memory_store: MemoryStore,
        max_concurrency: int = 3,
        task_timeout_seconds: int = 30,
        config: SchedulerConfig | None = None,
        replanner_agent: ReplannerAgent | None = None,
        evidence_policy: EvidencePolicy | None = None,
        degradation_manager: DegradationManager | None = None,
        failure_scenario: str = "none",
    ) -> None:
        self.config = config or SchedulerConfig(
            max_concurrency=max_concurrency,
            task_timeout_seconds=task_timeout_seconds,
        )
        if self.config.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        if self.config.task_timeout_seconds < 1:
            raise ValueError("task_timeout_seconds must be at least 1.")
        if self.config.global_timeout_seconds < 1:
            raise ValueError("global_timeout_seconds must be at least 1.")

        self.researcher_agent = researcher_agent
        self.memory_store = memory_store
        self.max_concurrency = self.config.max_concurrency
        self.task_timeout_seconds = self.config.task_timeout_seconds
        self.failure_scenario = failure_scenario
        self.state_machine = TaskStateMachine()
        self.evidence_policy = evidence_policy or EvidencePolicy(self.config.min_total_evidences)
        self.degradation_manager = degradation_manager or DegradationManager()
        self.replan_manager = ReplanManager(replanner_agent or ReplannerAgent())
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._replanned_source_task_ids: set[str] = set()
        self._forced_compose = False

    async def run(self, dag: TaskDAG) -> dict[str, Any]:
        trace = self._new_trace()
        started_at = perf_counter()

        while not dag.all_finished():
            remaining_seconds = self._remaining_global_seconds(started_at)
            if remaining_seconds <= 0:
                self._forced_compose = True
                self._cancel_unfinished_tasks(dag, trace, reason="global_timeout")
                break

            ready_tasks = dag.get_ready_tasks()
            if not ready_tasks:
                self._cancel_unfinished_tasks(dag, trace, reason="no_ready_tasks")
                break

            for task in ready_tasks:
                if task.state == TaskState.PENDING:
                    self._transition(task, TaskState.READY, trace)

            batch = [
                asyncio.create_task(self._run_task(task, trace))
                for task in ready_tasks
            ]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*batch),
                    timeout=remaining_seconds,
                )
            except TimeoutError:
                self._forced_compose = True
                for running_task in batch:
                    running_task.cancel()
                await asyncio.gather(*batch, return_exceptions=True)
                self._cancel_unfinished_tasks(dag, trace, reason="global_timeout")
                break

            await self._maybe_replan(dag, trace)

        self._finalize_trace(dag, trace)
        return trace

    async def _run_task(self, task: TaskNode, trace: dict[str, Any]) -> None:
        async with self._semaphore:
            self._transition(task, TaskState.RUNNING, trace)
            trace["task_start_time"][task.id] = _utc_now()
            started_at = perf_counter()

            try:
                timeout_seconds = self.config.task_timeout_seconds
                task.timeout_seconds = timeout_seconds
                evidences = await run_with_timeout(
                    self._research_task(task),
                    timeout_seconds,
                )
            except asyncio.CancelledError:
                self._finish_task_timing(task, started_at, trace)
                if task.state == TaskState.RUNNING:
                    self._transition(task, TaskState.CANCELLED, trace)
                    trace["cancelled_tasks"].append(task.id)
                raise
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

    async def _maybe_replan(self, dag: TaskDAG, trace: dict[str, Any]) -> None:
        if not self.config.enable_replan:
            self._update_evidence_sufficiency(trace)
            return
        if trace["replan_rounds"] >= self.config.max_replan_rounds:
            self._update_evidence_sufficiency(trace)
            return

        failed_tasks = [
            task
            for task in dag.list_tasks()
            if task.state in {TaskState.FAILED, TaskState.DEGRADED}
            and task.id not in self._replanned_source_task_ids
        ]
        replan_source_tasks = failed_tasks
        degraded_or_failed_count = len(failed_tasks)
        evidences = self.memory_store.list_evidences()
        insufficiency_reason = self.evidence_policy.reason_if_insufficient(evidences)
        critical_failed_tasks = [
            task for task in failed_tasks if bool(task.input.get("critical"))
        ]

        reason: str | None = None
        if degraded_or_failed_count >= self.config.batch_failure_threshold:
            reason = (
                f"{degraded_or_failed_count} tasks failed or degraded, meeting "
                f"batch threshold {self.config.batch_failure_threshold}."
            )
            trace["batch_failure_triggered"] = True
        elif insufficiency_reason and dag.all_finished():
            reason = insufficiency_reason
            replan_source_tasks = [
                task
                for task in dag.list_tasks()
                if task.id not in self._replanned_source_task_ids
            ][:1]
        elif critical_failed_tasks:
            reason = "A critical task failed or degraded."

        if reason is None or not replan_source_tasks:
            self._update_evidence_sufficiency(trace)
            return

        trace["replan_rounds"] += 1
        event = await self.replan_manager.create_replan_event(
            dag=dag,
            failed_tasks=replan_source_tasks,
            reason=reason,
            round_number=trace["replan_rounds"],
        )
        self._replanned_source_task_ids.update(event.failed_task_ids)
        trace["replan_events"].append(event.model_dump(mode="json"))
        trace["execution_steps"].append(
            {
                "step": "replan",
                "state": "SUCCEEDED",
                "reason": reason,
                "failed_task_ids": event.failed_task_ids,
                "new_task_ids": event.new_task_ids,
            }
        )
        self._update_evidence_sufficiency(trace)

    async def _research_task(self, task: TaskNode):
        await self._apply_failure_injection(task)
        return await self.researcher_agent.research(task)

    async def _apply_failure_injection(self, task: TaskNode) -> None:
        if "_replan_" in task.id:
            return
        if self.failure_scenario == "fail_one" and task.id == "task_2":
            raise RuntimeError("Injected deterministic failure for fail_one.")
        if self.failure_scenario == "fail_many" and task.id in {"task_2", "task_3"}:
            raise RuntimeError("Injected deterministic failure for fail_many.")
        if self.failure_scenario == "timeout_one" and task.id == "task_3":
            await asyncio.sleep(self.config.task_timeout_seconds + 1)
        if self.failure_scenario == "global_timeout" and task.id in {"task_1", "task_2", "task_3", "task_4"}:
            await asyncio.sleep(self.config.global_timeout_seconds + 5)

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
        evidence = self.degradation_manager.create_fallback_evidence(task, reason)
        record = self.degradation_manager.create_record(task, reason, evidence)
        self.memory_store.add_evidence(evidence)
        task.output = {"evidence_ids": [evidence.id], "degraded": True}
        self._transition(task, TaskState.DEGRADED, trace)
        trace["degraded_tasks"].append(task.id)
        trace["degradation_records"].append(record.model_dump(mode="json"))
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

    def _cancel_unfinished_tasks(
        self,
        dag: TaskDAG,
        trace: dict[str, Any],
        reason: str,
    ) -> None:
        for task in dag.list_tasks():
            if task.state in {TaskState.PENDING, TaskState.READY, TaskState.RUNNING}:
                self._transition(task, TaskState.CANCELLED, trace)
                if task.id not in trace["cancelled_tasks"]:
                    trace["cancelled_tasks"].append(task.id)
                trace["execution_steps"].append(
                    {
                        "step": "research_cancelled",
                        "task_id": task.id,
                        "state": task.state.value,
                        "reason": reason,
                    }
                )

    def _new_trace(self) -> dict[str, Any]:
        return {
            "max_concurrency": self.config.max_concurrency,
            "global_timeout_seconds": self.config.global_timeout_seconds,
            "forced_compose": False,
            "cancelled_tasks": [],
            "replan_events": [],
            "replan_rounds": 0,
            "degradation_records": [],
            "batch_failure_triggered": False,
            "evidence_sufficiency": {
                "is_sufficient": False,
                "reason": "No evidence has been collected yet.",
                "evidence_count": 0,
                "min_total_evidences": self.config.min_total_evidences,
            },
            "task_state_transitions": [],
            "task_start_time": {},
            "task_end_time": {},
            "task_duration_seconds": {},
            "retry_count": {},
            "degraded_tasks": [],
            "final_task_states": {},
            "execution_steps": [],
        }

    def _update_evidence_sufficiency(self, trace: dict[str, Any]) -> None:
        evidences = self.memory_store.list_evidences()
        reason = self.evidence_policy.reason_if_insufficient(evidences)
        trace["evidence_sufficiency"] = {
            "is_sufficient": reason is None,
            "reason": reason,
            "evidence_count": len(evidences),
            "min_total_evidences": self.config.min_total_evidences,
        }

    def _finalize_trace(self, dag: TaskDAG, trace: dict[str, Any]) -> None:
        trace["forced_compose"] = self._forced_compose
        trace["final_task_states"] = {
            task.id: task.state.value for task in dag.list_tasks()
        }
        trace["retry_count"] = {
            task.id: task.retry_count for task in dag.list_tasks()
        }
        self._update_evidence_sufficiency(trace)

    def _remaining_global_seconds(self, started_at: float) -> float:
        return self.config.global_timeout_seconds - (perf_counter() - started_at)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
