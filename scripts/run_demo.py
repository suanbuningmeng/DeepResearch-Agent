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

from deepresearch_agent.agents import BlueAgent, JudgeAgent, PlannerAgent, RedAgent, ResearcherAgent, WriterAgent
from deepresearch_agent.agents.writer import select_top_evidences_per_task
from deepresearch_agent.compression import ContextCompressionPipeline
from deepresearch_agent.conflict import EvidenceConflictDetector, EvidenceConflictResolver
from deepresearch_agent.engine import DAGTaskScheduler, TaskDAG
from deepresearch_agent.llm import create_llm
from deepresearch_agent.memory import HashingEmbeddingProvider
from deepresearch_agent.memory import MemoryStore, SQLiteMemoryStore
from deepresearch_agent.memory.source_quality import SourceQuality, classify_source_url
from deepresearch_agent.red_blue.controller import RedBlueController
from deepresearch_agent.red_blue.schemas import RedBlueStats
from deepresearch_agent.report import ensure_report_completeness, format_report_with_score
from deepresearch_agent.search import CitationValidator, SearchStats, WebFetcher
from deepresearch_agent.search.factory import create_search_provider, default_api_key_env
from deepresearch_agent.schemas import ConflictStats, MemoryStats, ResearchReport, SchedulerConfig, TaskNode, TaskState
from deepresearch_agent.ui.config import DemoRunConfig


DEFAULT_QUESTION = "What are the main challenges and recent methods for long-context LLM evaluation?"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the DeepResearch-Agent MVP demo.")
    parser.add_argument(
        "--preset",
        default="custom",
        choices=["custom", "resume"],
        help="Use a stable, resume-friendly demo configuration. Explicit CLI flags still override preset defaults.",
    )
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
    parser.add_argument("--memory-backend", default="memory", choices=["memory", "sqlite"], help="Evidence memory backend.")
    parser.add_argument("--memory-db-path", default="data/memory.sqlite", help="SQLite memory database path.")
    parser.add_argument("--vector-index-path", default="data/vector_index.npz", help="Numpy vector index path.")
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument("--enable-memory-retrieval", dest="enable_memory_retrieval", action="store_true", default=True)
    memory_group.add_argument("--disable-memory-retrieval", dest="enable_memory_retrieval", action="store_false")
    parser.add_argument("--memory-search-top-k", type=int, default=10, help="Top-K evidence items retrieved from memory.")
    parser.add_argument("--clear-memory", action="store_true", help="Clear persistent memory before running.")
    compression_group = parser.add_mutually_exclusive_group()
    compression_group.add_argument("--enable-context-compression", dest="enable_context_compression", action="store_true", default=False)
    compression_group.add_argument("--disable-context-compression", dest="enable_context_compression", action="store_false")
    parser.add_argument("--compression-l1-top-n", type=int, default=20, help="L1 embedding compression recall size.")
    parser.add_argument("--compression-l2-top-k", type=int, default=8, help="L2 TextRank compression final size.")
    conflict_group = parser.add_mutually_exclusive_group()
    conflict_group.add_argument("--enable-conflict-detection", dest="enable_conflict_detection", action="store_true", default=False)
    conflict_group.add_argument("--disable-conflict-detection", dest="enable_conflict_detection", action="store_false")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.92, help="Near-duplicate evidence similarity threshold.")
    parser.add_argument("--semantic-opposition-threshold", type=float, default=0.65, help="Semantic-opposition evidence similarity threshold.")
    red_blue_group = parser.add_mutually_exclusive_group()
    red_blue_group.add_argument("--enable-red-blue", dest="enable_red_blue", action="store_true", default=False)
    red_blue_group.add_argument("--disable-red-blue", dest="enable_red_blue", action="store_false")
    parser.add_argument("--red-blue-max-rounds", type=int, default=2, help="Maximum Red-Blue repair rounds.")
    parser.add_argument("--red-blue-min-score-delta", type=int, default=1, help="Minimum score delta for Red-Blue convergence.")
    severity_group = parser.add_mutually_exclusive_group()
    severity_group.add_argument("--red-blue-stop-on-no-high-severity", dest="red_blue_stop_on_no_high_severity", action="store_true", default=True)
    severity_group.add_argument("--red-blue-allow-low-severity-rounds", dest="red_blue_stop_on_no_high_severity", action="store_false")
    search_group = parser.add_mutually_exclusive_group()
    search_group.add_argument("--enable-web-search", dest="enable_web_search", action="store_true", default=False)
    search_group.add_argument("--disable-web-search", dest="enable_web_search", action="store_false")
    parser.add_argument("--search-provider", default="mock", choices=["mock", "arxiv", "paper", "tavily", "exa", "serper", "brave", "web"], help="Search provider to use when web search is enabled.")
    parser.add_argument("--search-top-k", type=int, default=3, help="Search results per query.")
    parser.add_argument("--max-search-queries", type=int, default=3, help="Search queries per research task.")
    parser.add_argument("--allow-network-fetch", action="store_true", default=False, help="Allow WebFetcher to request source URLs.")
    parser.add_argument("--search-timeout", type=int, default=20, help="Search provider request timeout.")
    parser.add_argument("--search-api-key-env", default=None, help="Environment variable that stores the search API key.")
    parser.add_argument("--search-api-base", default="", help="Generic web search API base URL.")
    parser.add_argument("--search-provider-name", default="mock", help="Provider name written to search stats.")
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
    memory_backend: str = "memory",
    memory_db_path: str = "data/memory.sqlite",
    vector_index_path: str = "data/vector_index.npz",
    enable_memory_retrieval: bool = True,
    memory_search_top_k: int = 10,
    clear_memory: bool = False,
    enable_context_compression: bool = False,
    compression_l1_top_n: int = 20,
    compression_l2_top_k: int = 8,
    enable_conflict_detection: bool = False,
    near_duplicate_threshold: float = 0.92,
    semantic_opposition_threshold: float = 0.65,
    enable_red_blue: bool = False,
    red_blue_max_rounds: int = 2,
    red_blue_min_score_delta: int = 1,
    red_blue_stop_on_no_high_severity: bool = True,
    enable_web_search: bool = False,
    search_provider: str = "mock",
    search_top_k: int = 3,
    max_search_queries: int = 3,
    allow_network_fetch: bool = False,
    search_timeout: int = 20,
    search_api_key_env: str | None = None,
    search_api_base: str = "",
    search_provider_name: str = "mock",
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
    memory = _create_memory_store(memory_backend, memory_db_path, vector_index_path)
    if clear_memory:
        memory.clear()
    planner = PlannerAgent(llm)
    active_search_provider = _create_search_provider(
        provider=search_provider,
        api_key_env=search_api_key_env,
        api_base=search_api_base,
        provider_name=search_provider_name,
        timeout=search_timeout,
    )
    researcher = ResearcherAgent(
        llm,
        enable_web_search=enable_web_search,
        search_provider=active_search_provider,
        fetcher=WebFetcher(allow_network=allow_network_fetch),
        citation_validator=CitationValidator(),
        max_search_queries=max_search_queries,
        search_top_k=search_top_k,
    )
    writer = WriterAgent(llm)
    judge = JudgeAgent(llm)
    red_agent = RedAgent(llm)
    blue_agent = BlueAgent(llm)

    execution_steps: list[dict[str, Any]] = []
    effective_global_timeout_seconds = (
        global_timeout_seconds
        if global_timeout_seconds is not None
        else (300 if backend != "mock" else 120)
    )

    subtasks = await planner.plan(question)
    planner_stats = planner.last_stats
    execution_steps.append(
        {
            "step": "plan",
            "state": "SUCCEEDED",
            "subtask_count": len(subtasks),
            "fallback_used": planner_stats.get("fallback_used", False),
        }
    )

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
        task_timeout_seconds = 1 if failure_scenario == "timeout_one" else (120 if enable_web_search else 60)
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
    writer_source_evidences = _select_writer_source_evidences(
        memory=memory,
        question=question,
        fallback_evidences=evidences,
        memory_backend=memory_backend,
        enable_memory_retrieval=enable_memory_retrieval,
        memory_search_top_k=memory_search_top_k,
    )
    embedding_provider = HashingEmbeddingProvider()
    compression_pipeline = ContextCompressionPipeline(
        embedding_provider=embedding_provider,
        l1_top_n=compression_l1_top_n,
        l2_top_k=compression_l2_top_k,
        enabled=enable_context_compression,
    )
    compressed_evidences, compression_stats = compression_pipeline.compress(question, writer_source_evidences)
    conflict_resolved_evidences, conflict_stats = _resolve_evidence_conflicts(
        evidences=compressed_evidences,
        embedding_provider=embedding_provider,
        enabled=enable_conflict_detection,
        near_duplicate_threshold=near_duplicate_threshold,
        semantic_opposition_threshold=semantic_opposition_threshold,
    )
    writer_evidences = select_top_evidences_per_task(conflict_resolved_evidences, writer_top_k_per_task)
    markdown = await writer.write(question, conflict_resolved_evidences, top_k_per_task=writer_top_k_per_task)
    if scheduler_trace.get("forced_compose"):
        markdown = _append_partial_report_notice(markdown)
        markdown, report_quality_check = ensure_report_completeness(markdown)
    else:
        report_quality_check = writer.last_quality_check or {}
    execution_steps.append({"step": "write", "state": "SUCCEEDED", "evidence_count": len(evidences)})

    memory_stats = _build_memory_stats(
        memory=memory,
        backend=memory_backend,
        db_path=memory_db_path if memory_backend == "sqlite" else None,
        vector_index_path=vector_index_path if memory_backend == "sqlite" else None,
        retrieved_evidence_count=len(writer_source_evidences) if enable_memory_retrieval else 0,
        memory_search_top_k=memory_search_top_k,
    )

    score = await judge.judge(question, markdown, evidences)
    execution_steps.append({"step": "judge", "state": "SUCCEEDED", "overall": score.overall})
    red_blue_stats = RedBlueStats(enabled=False)
    if enable_red_blue:
        controller = RedBlueController(
            red_agent=red_agent,
            blue_agent=blue_agent,
            judge_agent=judge,
            max_rounds=red_blue_max_rounds,
            min_score_delta=red_blue_min_score_delta,
            stop_on_no_high_severity=red_blue_stop_on_no_high_severity,
        )
        markdown, red_blue_stats = await controller.run(
            question=question,
            initial_report=markdown,
            evidences=evidences,
            initial_judge_score=score.model_dump(mode="json"),
            memory_stats=memory_stats.model_dump(mode="json"),
            compression_stats=compression_stats.model_dump(mode="json"),
            conflict_stats=conflict_stats.model_dump(mode="json"),
        )
        score = await judge.judge(question, markdown, evidences)
        execution_steps.append(
            {
                "step": "red_blue_repair",
                "state": "SUCCEEDED",
                "rounds_completed": red_blue_stats.rounds_completed,
                "stopped_reason": red_blue_stats.stopped_reason,
                "final_overall": score.overall,
            }
        )

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
        "planner_stats": planner_stats,
        "planned_subtasks": [task.model_dump(mode="json") for task in subtasks],
        "collected_evidences": [evidence.model_dump(mode="json") for evidence in evidences],
        "writer_top_k_per_task": writer_top_k_per_task,
        "used_evidence_count_for_writer": len(writer_evidences),
        "report_quality_check": report_quality_check,
        "search_stats": _search_stats_or_disabled(researcher.search_stats, enable_web_search).model_dump(mode="json"),
        "memory_stats": memory_stats.model_dump(mode="json"),
        "compression_stats": compression_stats.model_dump(mode="json"),
        "conflict_stats": conflict_stats.model_dump(mode="json"),
        "red_blue_stats": red_blue_stats.model_dump(mode="json"),
        "final_judge_score": score.model_dump(mode="json"),
        "judge_stats": judge.last_stats,
        "llm_call_stats": _llm_call_stats(llm),
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


async def run_demo_with_config(config: DemoRunConfig) -> dict[str, Any]:
    output_dir = config.output_dir / config.run_id if config.run_id else config.output_dir
    return await run_demo(
        question=config.question,
        backend=config.backend,
        output_dir=output_dir,
        mode=config.mode,
        max_concurrency=config.max_concurrency,
        global_timeout_seconds=config.global_timeout_seconds,
        enable_replan=config.enable_replan,
        failure_scenario=config.failure_scenario,
        api_base=config.api_base,
        api_key_env=config.api_key_env,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        request_timeout=config.request_timeout,
        enable_thinking=config.enable_thinking,
        writer_top_k_per_task=config.writer_top_k_per_task,
        memory_backend=config.memory_backend,
        memory_db_path=config.memory_db_path,
        vector_index_path=config.vector_index_path,
        enable_memory_retrieval=config.enable_memory_retrieval,
        memory_search_top_k=config.memory_search_top_k,
        clear_memory=config.clear_memory,
        enable_context_compression=config.enable_context_compression,
        compression_l1_top_n=config.compression_l1_top_n,
        compression_l2_top_k=config.compression_l2_top_k,
        enable_conflict_detection=config.enable_conflict_detection,
        near_duplicate_threshold=config.near_duplicate_threshold,
        semantic_opposition_threshold=config.semantic_opposition_threshold,
        enable_red_blue=config.enable_red_blue,
        red_blue_max_rounds=config.red_blue_max_rounds,
        red_blue_min_score_delta=config.red_blue_min_score_delta,
        red_blue_stop_on_no_high_severity=config.red_blue_stop_on_no_high_severity,
        enable_web_search=config.enable_web_search,
        search_provider=config.search_provider,
        search_top_k=config.search_top_k,
        max_search_queries=config.max_search_queries,
        allow_network_fetch=config.allow_network_fetch,
        search_timeout=config.search_timeout,
        search_api_key_env=config.search_api_key_env,
        search_api_base=config.search_api_base,
        search_provider_name=config.search_provider_name,
    )


def _build_task_dag(tasks: list[TaskNode]) -> TaskDAG:
    dag = TaskDAG()
    for task in tasks:
        task.state = TaskState.PENDING
        dag.add_task(task)
    return dag


def _llm_call_stats(llm: object) -> list[dict[str, Any]]:
    stats = getattr(llm, "call_stats", None)
    if not isinstance(stats, list):
        return []
    return [dict(item) for item in stats if isinstance(item, dict)]


def _append_partial_report_notice(markdown: str) -> str:
    notice = """

## Partial Report Notice
Some tasks were not completed due to global timeout.
The report is based on partial evidence.
"""
    return markdown.rstrip() + notice


def _resolve_evidence_conflicts(
    evidences: list,
    embedding_provider: HashingEmbeddingProvider,
    enabled: bool,
    near_duplicate_threshold: float,
    semantic_opposition_threshold: float,
) -> tuple[list, ConflictStats]:
    if not enabled:
        return list(evidences), ConflictStats(
            enabled=False,
            input_evidence_count=len(evidences),
            output_evidence_count=len(evidences),
        )
    detector = EvidenceConflictDetector(
        embedding_provider=embedding_provider,
        near_duplicate_threshold=near_duplicate_threshold,
        semantic_similarity_threshold=semantic_opposition_threshold,
    )
    conflicts = detector.detect(evidences)
    resolver = EvidenceConflictResolver()
    return resolver.resolve(evidences, conflicts)


def _create_search_provider(
    provider: str,
    api_key_env: str | None,
    api_base: str,
    provider_name: str,
    timeout: int,
):
    del provider_name
    return create_search_provider(
        provider_name=provider,
        api_key_env=api_key_env or default_api_key_env(provider),
        api_base=api_base or None,
        timeout=timeout,
    )


def _search_stats_or_disabled(stats: SearchStats, enabled: bool) -> SearchStats:
    if enabled:
        return stats
    return SearchStats(enabled=False)


def _create_memory_store(memory_backend: str, db_path: str, vector_index_path: str):
    if memory_backend == "sqlite":
        return SQLiteMemoryStore(db_path=db_path, vector_index_path=vector_index_path)
    return MemoryStore()


def _select_writer_source_evidences(
    memory,
    question: str,
    fallback_evidences,
    memory_backend: str,
    enable_memory_retrieval: bool,
    memory_search_top_k: int,
):
    if memory_backend != "sqlite" or not enable_memory_retrieval:
        return list(fallback_evidences)
    retrieved = memory.search_evidences(question, top_k=memory_search_top_k)
    if not retrieved:
        return list(fallback_evidences)
    real_url = [
        evidence
        for evidence in retrieved
        if classify_source_url(evidence.source_url) == SourceQuality.REAL_URL
    ]
    other = [
        evidence
        for evidence in retrieved
        if classify_source_url(evidence.source_url) != SourceQuality.REAL_URL
    ]
    return real_url + other


def _build_memory_stats(
    memory,
    backend: str,
    db_path: str | None,
    vector_index_path: str | None,
    retrieved_evidence_count: int,
    memory_search_top_k: int,
) -> MemoryStats:
    evidences = memory.list_evidences()
    source_quality_summary = {quality.value: 0 for quality in SourceQuality}
    if hasattr(memory, "source_quality_summary"):
        source_quality_summary = memory.source_quality_summary()
    else:
        for evidence in evidences:
            source_quality_summary[classify_source_url(evidence.source_url).value] += 1
    return MemoryStats(
        backend=backend,
        db_path=db_path,
        vector_index_path=vector_index_path,
        inserted_evidence_count=int(getattr(memory, "inserted_evidence_count", len(evidences))),
        duplicate_evidence_count=int(getattr(memory, "duplicate_evidence_count", 0)),
        total_evidence_count=len(evidences),
        retrieved_evidence_count=retrieved_evidence_count,
        memory_search_top_k=memory_search_top_k,
        source_quality_summary=source_quality_summary,
    )


def main() -> None:
    raw_args = sys.argv[1:]
    args = build_parser().parse_args(raw_args)
    _apply_preset(args, raw_args)
    trace = asyncio.run(
        run_demo_with_config(
            DemoRunConfig(
                question=args.question,
                backend=args.backend,
                output_dir=Path(args.output_dir),
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
                memory_backend=args.memory_backend,
                memory_db_path=args.memory_db_path,
                vector_index_path=args.vector_index_path,
                enable_memory_retrieval=args.enable_memory_retrieval,
                memory_search_top_k=args.memory_search_top_k,
                clear_memory=args.clear_memory,
                enable_context_compression=args.enable_context_compression,
                compression_l1_top_n=args.compression_l1_top_n,
                compression_l2_top_k=args.compression_l2_top_k,
                enable_conflict_detection=args.enable_conflict_detection,
                near_duplicate_threshold=args.near_duplicate_threshold,
                semantic_opposition_threshold=args.semantic_opposition_threshold,
                enable_red_blue=args.enable_red_blue,
                red_blue_max_rounds=args.red_blue_max_rounds,
                red_blue_min_score_delta=args.red_blue_min_score_delta,
                red_blue_stop_on_no_high_severity=args.red_blue_stop_on_no_high_severity,
                enable_web_search=args.enable_web_search,
                search_provider=args.search_provider,
                search_top_k=args.search_top_k,
                max_search_queries=args.max_search_queries,
                allow_network_fetch=args.allow_network_fetch,
                search_timeout=args.search_timeout,
                search_api_key_env=args.search_api_key_env,
                search_api_base=args.search_api_base,
                search_provider_name=args.search_provider_name,
            )
        )
    )
    print(f"Wrote {trace['outputs']['report']}")
    print(f"Wrote {trace['outputs']['trace']}")


def _apply_preset(args: argparse.Namespace, raw_args: list[str]) -> None:
    if args.preset != "resume":
        return
    provided = _provided_options(raw_args)
    _set_if_not_provided(args, provided, "mode", "dag")
    _set_if_not_provided(args, provided, "max_concurrency", 1)
    _set_if_not_provided(args, provided, "global_timeout_seconds", 720)
    _set_if_not_provided(args, provided, "request_timeout", 240)
    _set_if_not_provided(args, provided, "max_tokens", 2048)
    _set_if_not_provided(args, provided, "enable_thinking", False)
    _set_if_not_provided(args, provided, "enable_web_search", True)
    _set_if_not_provided(args, provided, "search_provider", "arxiv")
    _set_if_not_provided(args, provided, "max_search_queries", 1)
    _set_if_not_provided(args, provided, "search_top_k", 1)
    _set_if_not_provided(args, provided, "enable_context_compression", False)
    _set_if_not_provided(args, provided, "enable_conflict_detection", False)
    _set_if_not_provided(args, provided, "enable_red_blue", False)


def _provided_options(raw_args: list[str]) -> set[str]:
    options: set[str] = set()
    aliases = {
        "enable-thinking": "enable_thinking",
        "disable-thinking": "enable_thinking",
        "enable-web-search": "enable_web_search",
        "disable-web-search": "enable_web_search",
        "enable-context-compression": "enable_context_compression",
        "disable-context-compression": "enable_context_compression",
        "enable-conflict-detection": "enable_conflict_detection",
        "disable-conflict-detection": "enable_conflict_detection",
        "enable-red-blue": "enable_red_blue",
        "disable-red-blue": "enable_red_blue",
    }
    for value in raw_args:
        if not value.startswith("--"):
            continue
        name = value[2:].split("=", 1)[0].replace("-", "_")
        options.add(aliases.get(name.replace("_", "-"), name))
    return options


def _set_if_not_provided(args: argparse.Namespace, provided: set[str], name: str, value: object) -> None:
    if name not in provided:
        setattr(args, name, value)


if __name__ == "__main__":
    main()
