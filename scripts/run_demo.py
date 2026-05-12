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
from deepresearch_agent.agents.writer import select_top_evidences_per_task
from deepresearch_agent.engine import DAGTaskScheduler, TaskDAG
from deepresearch_agent.llm import create_llm
from deepresearch_agent.memory import MemoryStore
from deepresearch_agent.report import ensure_report_completeness, format_report_with_score
from deepresearch_agent.schemas import ResearchReport, SchedulerConfig, TaskNode, TaskState


DEFAULT_QUESTION = "What are the main challenges and recent methods for long-context LLM evaluation?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepResearch-Agent MVP demo.")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="Research question to investigate.")
    parser.add_argument(
        "--backend",
        default="mock",
        choices=["mock", "openai-compatible", "deepseek", "vllm"],
        help="LLM backend to use.",
    )
    parser.add_argument("--api-base", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key-env", default=None, help="Environment variable that stores the API key.")
    parser.add_argument("--model", default=None, help="Model name for real LLM backends.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for real LLM backends.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for real LLM backends.")
    parser.add_argument("--request-timeout", type=int, default=180, help="Read timeout for real LLM requests.")
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None)
    thinking_group.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.add_argument("--mode", default="dag", choices=["linear", "dag"], help="Research execution mode.")
    parser.add_argument("--max-concurrency", type=int, default=3, help="Maximum concurrent DAG research tasks.")
    parser.add_argument("--global-timeout-seconds", type=int, default=None, help="Global timeout for DAG mode.")
    parser.add_argument("--writer-top-k-per-task", type=int, default=2, help="Evidence items per task used by WriterAgent.")
    parser.add_argument("--enable-replan", dest="enable_replan", action="store_true", default=True)
    parser.add_argument("--disable-replan", dest="enable_replan", action="store_false")
    parser.add_argument(
        "--failure-scenario",
        default="none",
        choices=["none", "fail_one", "fail_many", "timeout_one", "global_timeout"],
        help="Deterministic mock failure scenario for DAG mode.",
    )
    parser.add_argument("--output-dir", default="outputs", help="Directory for demo outputs.")
    return parser


async def run_demo(
    question: str,
    backend: str,
    output_dir: Path,
    mode: str = "dag",
    max_concurrency: int = 3,
    global_timeout_seconds: int | None = None,
    enable_replan: bool = True,
    failure_scenario: str = "none",
    api_base: str | None = None,
    api_key_env: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    request_timeout: int = 180,
    enable_thinking: bool | None = None,
    writer_top_k_per_task: int = 2,
) -> dict[str, Any]:
    llm = create_llm(
        backend,
        api_base=api_base,
        api_key_env=api_key_env,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        enable_thinking=enable_thinking,
    )
    memory = MemoryStore()
    planner = PlannerAgent(llm)
    researcher = ResearcherAgent(llm)
    writer = WriterAgent(llm)
    judge = JudgeAgent(llm)

    execution_steps: list[dict[str, Any]] = []
    effective_global_timeout_seconds = (
        global_timeout_seconds
        if global_timeout_seconds is not None
        else (300 if backend != "mock" else 120)
    )

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
        task_timeout_seconds = 1 if failure_scenario == "timeout_one" else 30
        config = SchedulerConfig(
            max_concurrency=max_concurrency,
            task_timeout_seconds=task_timeout_seconds,
            global_timeout_seconds=effective_global_timeout_seconds,
            enable_replan=enable_replan,
        )
        scheduler = DAGTaskScheduler(
            researcher_agent=researcher,
            memory_store=memory,
            config=config,
            failure_scenario=failure_scenario,
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
    writer_evidences = select_top_evidences_per_task(evidences, writer_top_k_per_task)
    markdown = await writer.write(question, evidences, top_k_per_task=writer_top_k_per_task)
    if scheduler_trace.get("forced_compose"):
        markdown = _append_partial_report_notice(markdown)
        markdown, report_quality_check = ensure_report_completeness(markdown)
    else:
        report_quality_check = writer.last_quality_check or {}
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
        "writer_top_k_per_task": writer_top_k_per_task,
        "used_evidence_count_for_writer": len(writer_evidences),
        "report_quality_check": report_quality_check,
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
                "global_timeout_seconds": scheduler_trace["global_timeout_seconds"],
                "forced_compose": scheduler_trace["forced_compose"],
                "cancelled_tasks": scheduler_trace["cancelled_tasks"],
                "replan_events": scheduler_trace["replan_events"],
                "replan_rounds": scheduler_trace["replan_rounds"],
                "degradation_records": scheduler_trace["degradation_records"],
                "batch_failure_triggered": scheduler_trace["batch_failure_triggered"],
                "evidence_sufficiency": scheduler_trace["evidence_sufficiency"],
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


def _append_partial_report_notice(markdown: str) -> str:
    notice = """

## Partial Report Notice
Some tasks were not completed due to global timeout.
The report is based on partial evidence.
"""
    return markdown.rstrip() + notice


def main() -> None:
    args = build_parser().parse_args()
    trace = asyncio.run(
        run_demo(
            args.question,
            args.backend,
            Path(args.output_dir),
            mode=args.mode,
            max_concurrency=args.max_concurrency,
            global_timeout_seconds=args.global_timeout_seconds,
            enable_replan=args.enable_replan,
            failure_scenario=args.failure_scenario,
            api_base=args.api_base,
            api_key_env=args.api_key_env,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            enable_thinking=args.enable_thinking,
            writer_top_k_per_task=args.writer_top_k_per_task,
        )
    )
    print(f"Wrote {trace['outputs']['report']}")
    print(f"Wrote {trace['outputs']['trace']}")


if __name__ == "__main__":
    main()
