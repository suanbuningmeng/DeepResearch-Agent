import asyncio

from deepresearch_agent.agents import ResearcherAgent
from deepresearch_agent.engine import DAGTaskScheduler, TaskDAG
from deepresearch_agent.llm import MockLLM
from deepresearch_agent.memory import MemoryStore
from deepresearch_agent.schemas import SchedulerConfig, TaskNode, TaskState


def make_dag() -> TaskDAG:
    dag = TaskDAG()
    for task_id, name in [
        ("task_1", "Identify key challenges"),
        ("task_2", "Review evaluation benchmarks"),
        ("task_3", "Compare recent methods"),
        ("task_4", "Assess limitations"),
    ]:
        dag.add_task(
            TaskNode(
                id=task_id,
                name=name,
                description=f"Research subtask for {name}.",
                agent_type="researcher",
                state=TaskState.PENDING,
            )
        )
    return dag


def test_fail_one_produces_degraded_task_or_retry_trace() -> None:
    async def run() -> None:
        memory = MemoryStore()
        scheduler = DAGTaskScheduler(
            researcher_agent=ResearcherAgent(MockLLM()),
            memory_store=memory,
            config=SchedulerConfig(enable_replan=False, task_timeout_seconds=2),
            failure_scenario="fail_one",
        )

        trace = await scheduler.run(make_dag())

        assert "task_2" in trace["degraded_tasks"]
        assert trace["retry_count"]["task_2"] == 1
        assert trace["final_task_states"]["task_2"] == "DEGRADED"

    asyncio.run(run())


def test_fail_many_triggers_batch_replan() -> None:
    async def run() -> None:
        memory = MemoryStore()
        scheduler = DAGTaskScheduler(
            researcher_agent=ResearcherAgent(MockLLM()),
            memory_store=memory,
            config=SchedulerConfig(task_timeout_seconds=2, batch_failure_threshold=2),
            failure_scenario="fail_many",
        )

        trace = await scheduler.run(make_dag())

        assert trace["batch_failure_triggered"] is True
        assert trace["replan_events"]
        assert trace["replan_rounds"] == 1
        assert "task_2_replan_1" in trace["final_task_states"]
        assert "task_3_replan_2" in trace["final_task_states"]

    asyncio.run(run())
