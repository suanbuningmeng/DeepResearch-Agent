from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from deepresearch_agent.agents import JudgeAgent, PlannerAgent, ResearcherAgent, WriterAgent
from deepresearch_agent.engine import DAGTaskScheduler, TaskDAG
from deepresearch_agent.llm import MockLLM
from deepresearch_agent.memory import MemoryStore
from deepresearch_agent.report import format_report_with_score
from deepresearch_agent.schemas import ResearchReport, TaskNode, TaskState


DEFAULT_QUESTION = "What are the main challenges and recent methods for long-context LLM evaluation?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepResearch-Agent MVP demo.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Research question to investigate.")
    parser.add_argument("--backend", default="mock", choices=["mock"], help="LLM backend to use.")
    parser.add_argument("--mode", default="dag", choices=["linear", "dag"], help="Research execution mode.")
    parser.add_argument("--max-concurrency", type=int, default=3, help="Maximum concurrent DAG research tasks.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for demo outputs.")
    return parser


async def run_demo(
    question: str,
    backend: str,
    output_dir: Path,
    mode: str = "dag",
    max_concurrency: int = 3,
) -> dict[str, Any]:
    if backend != "mock":
        raise ValueError("Only the mock backend is implemented in this MVP.")

    llm = MockLLM()
    memory = MemoryStore()
    planner = PlannerAgent(llm)
    researcher = ResearcherAgent(llm)
    writer = WriterAgent(llm)
    judge = JudgeAgent(llm)

    execution_steps: list[dict[str, Any]] = []

    subtasks = await planner.plan(question)
    execution_steps.append({"step": "plan", "state": "SUCCEEDED", "subtask_count": len(subtasks)})

    scheduler_trace: dict[str, Any] = {}
    if mode == "linear":
        for task in subtasks:
            task.state = TaskState.RUNNING
            evidences = await researcher.research(task)
            memory.add_evidences(evidences)
            task.output = {"evidence_ids": [evidence.id for evidence in evidences]}
            task.state = TaskState.SUCCEEDED
            execution_steps.append(
                {
                    "step": "research",
                    "task_id": task.id,
                    "state": task.state.value,
                    "evidence_count": len(evidences),
                }
            )
    else:
        dag = _build_task_dag(subtasks)
        scheduler = DAGTaskScheduler(
            researcher_agent=researcher,
            memory_store=memory,
            max_concurrency=max_concurrency,
            task_timeout_seconds=30,
        )
        scheduler_trace = await scheduler.run(dag)
        execution_steps.append(
            {
                "step": "dag_research",
                "state": "SUCCEEDED",
                "max_concurrency": max_concurrency,
                "task_count": len(subtasks),
            }
        )
        execution_steps.extend(scheduler_trace.get("execution_steps", []))

    evidences = memory.list_evidences()
    markdown = await writer.write(question, evidences)
    execution_steps.append({"step": "write", "state": "SUCCEEDED", "evidence_count": len(evidences)})

    score = await judge.judge(question, markdown, evidences)
    execution_steps.append({"step": "judge", "state": "SUCCEEDED", "overall": score.overall})

    report = ResearchReport(question=question, markdown=markdown, evidences=evidences, score=score)
    final_markdown = format_report_with_score(report.markdown, score)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "demo_report.md"
    trace_path = output_dir / "demo_trace.json"

    report_path.write_text(final_markdown, encoding="utf-8")
    trace = {
        "question": question,
        "backend": backend,
        "execution_mode": mode,
        "planned_subtasks": [task.model_dump(mode="json") for task in subtasks],
        "collected_evidences": [evidence.model_dump(mode="json") for evidence in evidences],
        "final_judge_score": score.model_dump(mode="json"),
        "execution_steps": execution_steps,
        "outputs": {
            "report": str(report_path),
            "trace": str(trace_path),
        },
    }
    if mode == "dag":
        trace.update(
            {
                "max_concurrency": scheduler_trace["max_concurrency"],
                "task_state_transitions": scheduler_trace["task_state_transitions"],
                "task_start_time": scheduler_trace["task_start_time"],
                "task_end_time": scheduler_trace["task_end_time"],
                "task_duration_seconds": scheduler_trace["task_duration_seconds"],
                "retry_count": scheduler_trace["retry_count"],
                "degraded_tasks": scheduler_trace["degraded_tasks"],
                "final_task_states": scheduler_trace["final_task_states"],
            }
        )
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return trace


def _build_task_dag(tasks: list[TaskNode]) -> TaskDAG:
    dag = TaskDAG()
    for task in tasks:
        task.state = TaskState.PENDING
        dag.add_task(task)
    return dag


def main() -> None:
    args = build_parser().parse_args()
    trace = asyncio.run(
        run_demo(
            args.question,
            args.backend,
            Path(args.output_dir),
            mode=args.mode,
            max_concurrency=args.max_concurrency,
        )
    )
    print(f"Wrote {trace['outputs']['report']}")
    print(f"Wrote {trace['outputs']['trace']}")


if __name__ == "__main__":
    main()
