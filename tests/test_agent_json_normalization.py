import asyncio
import json

from deepresearch_agent.agents import JudgeAgent, PlannerAgent, ResearcherAgent
from deepresearch_agent.llm import BaseLLM
from deepresearch_agent.schemas import TaskNode, TaskState


class StaticLLM(BaseLLM):
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return json.dumps(self.payload)


def test_planner_normalizes_numeric_task_ids() -> None:
    async def run() -> None:
        planner = PlannerAgent(
            StaticLLM(
                {
                    "subtasks": [
                        {"id": 1, "name": "A", "description": "First task"},
                        {"id": "task_2", "name": "B", "description": "Second task"},
                    ]
                }
            )
        )

        tasks = await planner.plan("Question?")

        assert [task.id for task in tasks] == ["task_1", "task_2"]

    asyncio.run(run())


def test_researcher_coerces_confidence_and_optional_source_url() -> None:
    async def run() -> None:
        researcher = ResearcherAgent(
            StaticLLM(
                {
                    "evidences": [
                        {
                            "title": "Evidence",
                            "content": "Content",
                            "source_url": None,
                            "confidence": "88",
                        }
                    ]
                }
            )
        )
        task = TaskNode(
            id="task_1",
            name="Task",
            description="Description",
            agent_type="researcher",
            state=TaskState.READY,
        )

        evidences = await researcher.research(task)

        assert evidences[0].source_url is None
        assert evidences[0].confidence == 0.88

    asyncio.run(run())


def test_judge_coerces_string_scores() -> None:
    async def run() -> None:
        judge = JudgeAgent(
            StaticLLM(
                {
                    "factuality": "88",
                    "coverage": "85",
                    "reasoning_depth": "86",
                    "citation_quality": "82",
                    "clarity": "90",
                    "overall": "86",
                    "comments": "ok",
                }
            )
        )

        score = await judge.judge("Question?", "Report", [])

        assert score.overall == 86

    asyncio.run(run())
