import asyncio

from deepresearch_agent.agents import ResearcherAgent
from deepresearch_agent.engine import DAGTaskScheduler, TaskDAG
from deepresearch_agent.llm import MockLLM
from deepresearch_agent.memory import MemoryStore
from deepresearch_agent.schemas import TaskNode, TaskState


def make_task(task_id: str, name: str) -> TaskNode:
    return TaskNode(
        id=task_id,
        name=name,
        description=f"Research subtask for {name}.",
        agent_type="researcher",
        state=TaskState.PENDING,
    )


def test_dag_scheduler_runs_independent_research_tasks() -> None:
    async def run() -> None:
        dag = TaskDAG()
        dag.add_task(make_task("task_1", "Identify key challenges"))
        dag.add_task(make_task("task_2", "Review evaluation benchmarks"))
        dag.add_task(make_task("task_3", "Compare recent methods"))

        memory = MemoryStore()
        scheduler = DAGTaskScheduler(
            researcher_agent=ResearcherAgent(MockLLM()),
            memory_store=memory,
            max_concurrency=2,
            task_timeout_seconds=5,
        )

        trace = await scheduler.run(dag)
        final_states = {task.state for task in dag.list_tasks()}

        assert final_states <= {TaskState.SUCCEEDED, TaskState.DEGRADED}
        assert len(memory.list_evidences()) >= 3
        assert "task_state_transitions" in trace
        assert "final_task_states" in trace
        assert trace["task_state_transitions"]
        assert set(trace["final_task_states"]) == {"task_1", "task_2", "task_3"}

    asyncio.run(run())
