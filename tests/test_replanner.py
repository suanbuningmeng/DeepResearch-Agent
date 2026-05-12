import asyncio

from deepresearch_agent.agents import ReplannerAgent
from deepresearch_agent.schemas import TaskNode, TaskState


def test_replanner_agent_creates_replacement_tasks() -> None:
    async def run() -> None:
        failed_task = TaskNode(
            id="task_2",
            name="Review evaluation benchmarks",
            description="Summarize benchmark designs.",
            agent_type="researcher",
            state=TaskState.DEGRADED,
        )

        new_tasks = await ReplannerAgent().replan(
            failed_tasks=[failed_task],
            existing_tasks=[failed_task],
            reason="batch failure",
        )

        assert [task.id for task in new_tasks] == ["task_2_replan_1", "task_2_replan_2"]
        assert all(task.agent_type == "researcher" for task in new_tasks)
        assert all(task.state == TaskState.PENDING for task in new_tasks)

    asyncio.run(run())
