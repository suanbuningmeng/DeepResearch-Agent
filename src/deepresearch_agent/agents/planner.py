from __future__ import annotations

from pathlib import Path
from typing import Any

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import TaskNode, TaskState
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking
from deepresearch_agent.utils.structured_output import repair_json_with_llm, try_parse_json_lenient


PLANNER_SCHEMA_HINT = """
{
  "subtasks": [
    {
      "id": "task_1",
      "name": "string",
      "description": "string",
      "agent_type": "researcher",
      "dependencies": []
    }
  ]
}
"""


class PlannerAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.last_stats: dict[str, Any] = {
            "json_parse_success": False,
            "repair_attempted": False,
            "repair_success": False,
            "schema_coercion_success": False,
            "partial_repair_used": False,
            "fallback_used": False,
            "subtask_count": 0,
            "error": None,
        }

    async def plan(self, question: str) -> list[TaskNode]:
        """Split a user question into structured research subtasks."""
        prompt = (
            "You are a planner agent. Split the research question into 3-5 subtasks.\n"
            f"Question: {question}\n"
            "Every subtask must directly address the user's question.\n"
            "Do not create generic, unrelated, or placeholder tasks.\n"
            "Never output names or descriptions such as default_task, unknown, No description provided, TBD, or N/A.\n"
            "Each subtask must have a specific name and a concrete description.\n"
            "For literature surveys, method reviews, benchmark analysis, and other research evidence collection tasks, make subtasks independent by default.\n"
            "Use dependencies: [] for researcher subtasks unless a task truly cannot be performed without a previous task's output.\n"
            "Do not create a serial chain like task_2 depends on task_1, task_3 depends on task_2, task_4 depends on task_3 for normal research surveys.\n"
            "If the question is about long-context LLM evaluation, focus on benchmarks, metrics, context retention, multi-hop reasoning, lost-in-the-middle effects, faithfulness, latency, cost, scalability, retrieval, synthesis, and citation quality.\n"
            "Return valid JSON only.\n"
            "Do not output markdown.\n"
            "Do not include markdown fences.\n"
            "Do not include explanations.\n"
            "Do not include <think> blocks.\n"
            "Return 3-5 subtasks.\n"
            "Use this schema exactly:\n"
            "{\n"
            '  "subtasks": [\n'
            "    {\n"
            '      "id": "task_1",\n'
            '      "name": "string",\n'
            '      "description": "string",\n'
            '      "agent_type": "researcher",\n'
            '      "dependencies": []\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        raw = await self.llm.agenerate(prompt, prompt_type="planner")
        _write_debug_planner_output(raw)
        cleaned = strip_thinking(raw)
        try:
            data = safe_json_loads(cleaned)
            tasks, partial_repair_used = coerce_planner_data(data, question, fill_missing=False)
            self.last_stats = {
                "json_parse_success": True,
                "repair_attempted": False,
                "repair_success": False,
                "schema_coercion_success": False,
                "partial_repair_used": partial_repair_used,
                "fallback_used": False,
                "subtask_count": len(tasks),
                "error": None,
            }
            return tasks
        except (KeyError, TypeError, ValueError) as exc:
            parse_error = str(exc)
            lenient_data = try_parse_json_lenient(cleaned)
            if lenient_data is not None:
                try:
                    tasks, partial_repair_used = coerce_planner_data(lenient_data, question)
                    self.last_stats = {
                        "json_parse_success": False,
                        "repair_attempted": False,
                        "repair_success": False,
                        "schema_coercion_success": True,
                        "partial_repair_used": partial_repair_used,
                        "fallback_used": False,
                        "subtask_count": len(tasks),
                        "error": None,
                    }
                    return tasks
                except (KeyError, TypeError, ValueError) as lenient_exc:
                    parse_error = str(lenient_exc)

            repaired, repair_stats = await repair_json_with_llm(
                self.llm,
                raw_output=raw,
                schema_hint=PLANNER_SCHEMA_HINT,
                task_name="planner",
            )
            if repaired is not None:
                try:
                    tasks, partial_repair_used = coerce_planner_data(repaired, question)
                    self.last_stats = {
                        "json_parse_success": False,
                        "repair_attempted": bool(repair_stats["repair_attempted"]),
                        "repair_success": True,
                        "schema_coercion_success": True,
                        "partial_repair_used": partial_repair_used,
                        "fallback_used": False,
                        "subtask_count": len(tasks),
                        "error": None,
                    }
                    return tasks
                except (KeyError, TypeError, ValueError) as repair_exc:
                    parse_error = str(repair_exc)

            tasks = _default_fallback_tasks(question)
            self.last_stats = {
                "json_parse_success": False,
                "repair_attempted": bool(repair_stats.get("repair_attempted", False)),
                "repair_success": False,
                "schema_coercion_success": False,
                "partial_repair_used": False,
                "fallback_used": True,
                "subtask_count": len(tasks),
                "error": (repair_stats.get("repair_error") or parse_error)[:200],
            }
            return tasks


def coerce_planner_data(data: Any, question: str, fill_missing: bool = True) -> tuple[list[TaskNode], bool]:
    """Coerce common planner JSON variants into TaskNode objects."""
    items: Any = None
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("subtasks", "tasks", "plan", "research_tasks"):
            if isinstance(data.get(key), list):
                items = data[key]
                break
    if items is None:
        raise ValueError("Planner returned no subtasks.")

    tasks = _normalize_subtask_items(items, question)
    if _should_parallelize_research_tasks(question):
        tasks = _clear_research_dependencies(tasks)
    partial_repair_used = False
    if fill_missing and len(tasks) < 3:
        partial_repair_used = True
        existing_ids = {task.id for task in tasks}
        for fallback in _default_fallback_tasks(question):
            if fallback.id in existing_ids:
                continue
            tasks.append(fallback)
            existing_ids.add(fallback.id)
            if len(tasks) >= 4:
                break
    return tasks, partial_repair_used


def _normalize_subtask_items(items: object, question: str) -> list[TaskNode]:
    if not isinstance(items, list):
        raise ValueError("Planner JSON field 'subtasks' must be a list.")
    tasks: list[TaskNode] = []
    for index, item in enumerate(items[:5], start=1):
        if not isinstance(item, dict):
            item = {}
        description_value = item.get("description") or item.get("details") or item.get("objective") or item.get("question") or item.get("task")
        name_value = item.get("name") or item.get("title") or item.get("task_name")
        description = str(description_value or name_value or question).strip()
        name = str(name_value or description[:80] or f"Research subtask {index}").strip()
        dependencies = item.get("dependencies") or []
        if not isinstance(dependencies, list):
            dependencies = []
        tasks.append(
            TaskNode(
                id=_normalize_task_id(item.get("id", item.get("task_id", index)), index),
                name=name,
                description=description or name,
                agent_type=str(item.get("agent_type") or "researcher"),
                dependencies=[str(dependency) for dependency in dependencies],
                state=TaskState.READY,
                input={"question": question},
            )
        )
    if not tasks:
        raise ValueError("Planner returned no subtasks.")
    return tasks


def _should_parallelize_research_tasks(question: str) -> bool:
    text = question.lower()
    indicators = (
        "what are",
        "recent",
        "methods",
        "challenges",
        "survey",
        "review",
        "compare",
        "evaluation",
        "benchmark",
        "paper",
        "research",
        "llm",
        "large language model",
        "semantic communication",
        "ai",
    )
    return any(indicator in text for indicator in indicators)


def _clear_research_dependencies(tasks: list[TaskNode]) -> list[TaskNode]:
    """Research evidence collection usually benefits from parallel shards."""
    parallel_tasks: list[TaskNode] = []
    for task in tasks:
        if task.agent_type == "researcher" and task.dependencies:
            parallel_tasks.append(task.model_copy(update={"dependencies": []}))
        else:
            parallel_tasks.append(task)
    return parallel_tasks


def _default_fallback_tasks(question: str) -> list[TaskNode]:
    defaults = [
        (
            "Identify key challenges",
            "Identify the main challenges related to the user question.",
        ),
        (
            "Survey recent methods",
            "Survey recent methods, frameworks, or approaches related to the user question.",
        ),
        (
            "Analyze effectiveness and limitations",
            "Analyze the effectiveness, limitations, and trade-offs of the identified methods.",
        ),
        (
            "Summarize trends and future directions",
            "Summarize emerging trends and future research directions.",
        ),
    ]
    return [
        TaskNode(
            id=f"task_{index}",
            name=name,
            description=description,
            agent_type="researcher",
            dependencies=[],
            state=TaskState.READY,
            input={"question": question, "planner_fallback": True},
        )
        for index, (name, description) in enumerate(defaults, start=1)
    ]


def _normalize_task_id(value: object, fallback_index: int) -> str:
    text = str(value).strip()
    if not text:
        return f"task_{fallback_index}"
    if text.startswith("task_"):
        return text
    return f"task_{text}"


def _write_debug_planner_output(raw: str) -> None:
    output_path = Path("outputs") / "debug_last_planner_output.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(raw, encoding="utf-8")
