import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepresearch_agent.agents import PlannerAgent
from deepresearch_agent.llm import BaseLLM
from deepresearch_agent.ui.trace_view import extract_trace_summary
from scripts.run_demo import run_demo


class StaticTextLLM(BaseLLM):
    def __init__(self, text: str) -> None:
        self.text = text

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        return self.text


def test_planner_handles_valid_json() -> None:
    async def run() -> None:
        planner = PlannerAgent(
            StaticTextLLM(
                json.dumps(
                    {
                        "subtasks": [
                            {
                                "id": "task_1",
                                "name": "A",
                                "description": "Do A",
                                "agent_type": "researcher",
                                "dependencies": [],
                            }
                        ]
                    }
                )
            )
        )

        tasks = await planner.plan("Question?")

        assert tasks[0].id == "task_1"
        assert planner.last_stats["json_parse_success"] is True
        assert planner.last_stats["fallback_used"] is False

    asyncio.run(run())


def test_planner_normalizes_int_ids_and_dependencies() -> None:
    async def run() -> None:
        planner = PlannerAgent(
            StaticTextLLM(
                json.dumps(
                    {
                        "subtasks": [
                            {
                                "id": 1,
                                "name": "A",
                                "description": "Do A",
                                "dependencies": [0, "task_x"],
                            }
                        ]
                    }
                )
            )
        )

        tasks = await planner.plan("Question?")

        assert tasks[0].id == "task_1"
        assert tasks[0].agent_type == "researcher"
        assert tasks[0].dependencies == ["0", "task_x"]

    asyncio.run(run())


def test_planner_parallelizes_research_survey_dependencies() -> None:
    async def run() -> None:
        planner = PlannerAgent(
            StaticTextLLM(
                json.dumps(
                    {
                        "subtasks": [
                            {
                                "id": "task_1",
                                "name": "Find methods",
                                "description": "Find recent methods.",
                                "dependencies": [],
                            },
                            {
                                "id": "task_2",
                                "name": "Analyze applications",
                                "description": "Analyze task-oriented applications.",
                                "dependencies": ["task_1"],
                            },
                            {
                                "id": "task_3",
                                "name": "Review metrics",
                                "description": "Review evaluation metrics.",
                                "dependencies": ["task_1", "task_2"],
                            },
                        ]
                    }
                )
            )
        )

        tasks = await planner.plan("What are recent methods for task-oriented semantic communication with LLMs?")

        assert len(tasks) == 3
        assert all(task.dependencies == [] for task in tasks)

    asyncio.run(run())


def test_planner_fallback_generates_four_default_tasks() -> None:
    async def run() -> None:
        planner = PlannerAgent(StaticTextLLM("I will discuss this in prose, not JSON."))

        tasks = await planner.plan("What are the biggest challenges of semantic communication?")

        assert len(tasks) == 4
        assert [task.id for task in tasks] == ["task_1", "task_2", "task_3", "task_4"]
        assert tasks[0].name == "Identify key challenges"
        assert all(task.agent_type == "researcher" for task in tasks)
        assert all(task.dependencies == [] for task in tasks)
        assert planner.last_stats["fallback_used"] is True
        assert planner.last_stats["error"]

    asyncio.run(run())


def test_run_demo_trace_records_planner_stats(tmp_path: Path) -> None:
    async def run() -> None:
        trace = await run_demo(
            question="test planner stats",
            backend="mock",
            output_dir=tmp_path / "outputs",
            mode="dag",
        )

        assert trace["planner_stats"]["json_parse_success"] is True
        assert trace["planner_stats"]["fallback_used"] is False
        assert trace["planner_stats"]["subtask_count"] >= 3

    asyncio.run(run())


def test_trace_summary_handles_missing_planner_stats() -> None:
    summary = extract_trace_summary({"backend": "mock"})

    assert "planner_stats" in summary
    assert summary["planner_stats"] is None
