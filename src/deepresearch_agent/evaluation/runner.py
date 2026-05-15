from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from deepresearch_agent.evaluation.benchmark import BenchmarkSuite, filter_benchmark, load_benchmark
from deepresearch_agent.evaluation.exporter import export_results_csv, export_results_jsonl, export_summary_markdown
from deepresearch_agent.evaluation.judge_metrics import JudgeMetricResult, extract_judge_metrics
from deepresearch_agent.evaluation.rule_metrics import RuleMetricResult, compute_rule_metrics
from deepresearch_agent.evaluation.statistics import bootstrap_ci
from deepresearch_agent.ui.config import DemoRunConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_demo import run_demo_with_config


class EvaluationConfig(BaseModel):
    benchmark_path: str
    output_dir: str = "eval_outputs"
    backend: str = "mock"
    mode: str = "dag"
    limit: int | None = None
    domain: str | None = None
    enable_context_compression: bool = False
    enable_conflict_detection: bool = False
    enable_red_blue: bool = False
    memory_backend: str = "memory"
    clear_memory: bool = True
    max_concurrency: int = 1
    red_blue_max_rounds: int = 2


class EvaluationCaseResult(BaseModel):
    example_id: str
    domain: str
    question: str
    report_path: str
    trace_path: str
    rule_metrics: RuleMetricResult | None = None
    judge_metrics: JudgeMetricResult | None = None
    error: str | None = None


class EvaluationSummary(BaseModel):
    benchmark_name: str
    case_count: int
    success_count: int
    error_count: int
    average_rule_overall: float | None
    average_judge_overall: float | None
    bootstrap_ci_rule_overall: tuple[float, float] | None
    bootstrap_ci_judge_overall: tuple[float, float] | None
    domain_breakdown: dict
    config: dict


class EvaluationRunner:
    async def run(self, config: EvaluationConfig) -> EvaluationSummary:
        suite = filter_benchmark(load_benchmark(config.benchmark_path), domain=config.domain, limit=config.limit)
        run_dir = Path(config.output_dir) / _run_id(suite, config)
        run_dir.mkdir(parents=True, exist_ok=True)
        results: list[EvaluationCaseResult] = []

        for example in suite.examples:
            case_dir = run_dir / example.id
            case_dir.mkdir(parents=True, exist_ok=True)
            report_path = case_dir / "report.md"
            trace_path = case_dir / "trace.json"
            try:
                trace = await run_demo_with_config(
                    DemoRunConfig(
                        question=example.question,
                        backend=config.backend,
                        mode=config.mode,
                        output_dir=case_dir,
                        max_concurrency=config.max_concurrency,
                        memory_backend=config.memory_backend,
                        memory_db_path=str(case_dir / "memory.sqlite"),
                        vector_index_path=str(case_dir / "vector_index.npz"),
                        clear_memory=config.clear_memory,
                        enable_context_compression=config.enable_context_compression,
                        enable_conflict_detection=config.enable_conflict_detection,
                        enable_red_blue=config.enable_red_blue,
                        red_blue_max_rounds=config.red_blue_max_rounds,
                    )
                )
                demo_report_path = case_dir / "demo_report.md"
                demo_trace_path = case_dir / "demo_trace.json"
                if demo_report_path.exists():
                    report_path.write_text(demo_report_path.read_text(encoding="utf-8"), encoding="utf-8")
                if demo_trace_path.exists():
                    trace_path.write_text(demo_trace_path.read_text(encoding="utf-8"), encoding="utf-8")
                report = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
                rule_metrics = compute_rule_metrics(example, report, trace)
                judge_metrics = extract_judge_metrics(trace)
                result = EvaluationCaseResult(
                    example_id=example.id,
                    domain=example.domain,
                    question=example.question,
                    report_path=str(report_path),
                    trace_path=str(trace_path),
                    rule_metrics=rule_metrics,
                    judge_metrics=judge_metrics,
                )
            except Exception as exc:
                result = EvaluationCaseResult(
                    example_id=example.id,
                    domain=example.domain,
                    question=example.question,
                    report_path=str(report_path),
                    trace_path=str(trace_path),
                    error=str(exc)[:500],
                )
            (case_dir / "case_result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
            results.append(result)

        summary = _build_summary(suite, results, config)
        (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        export_results_jsonl(results, str(run_dir / "results.jsonl"))
        export_results_csv(results, str(run_dir / "results.csv"))
        export_summary_markdown(summary, str(run_dir / "summary.md"))
        return summary


def _build_summary(
    suite: BenchmarkSuite,
    results: list[EvaluationCaseResult],
    config: EvaluationConfig,
) -> EvaluationSummary:
    successes = [result for result in results if result.error is None]
    rule_values = [result.rule_metrics.rule_overall for result in successes if result.rule_metrics is not None]
    judge_values = [
        result.judge_metrics.overall
        for result in successes
        if result.judge_metrics is not None and result.judge_metrics.overall is not None
    ]
    return EvaluationSummary(
        benchmark_name=suite.name,
        case_count=len(results),
        success_count=len(successes),
        error_count=len(results) - len(successes),
        average_rule_overall=_average(rule_values),
        average_judge_overall=_average(judge_values),
        bootstrap_ci_rule_overall=bootstrap_ci(rule_values) if rule_values else None,
        bootstrap_ci_judge_overall=bootstrap_ci([float(value) for value in judge_values]) if judge_values else None,
        domain_breakdown=_domain_breakdown(results),
        config=_safe_config(config),
    )


def _domain_breakdown(results: list[EvaluationCaseResult]) -> dict:
    breakdown: dict[str, dict[str, Any]] = {}
    for result in results:
        item = breakdown.setdefault(result.domain, {"case_count": 0, "success_count": 0, "average_rule_overall": None})
        item["case_count"] += 1
        if result.error is None:
            item["success_count"] += 1
    for domain, item in breakdown.items():
        values = [
            result.rule_metrics.rule_overall
            for result in results
            if result.domain == domain and result.error is None and result.rule_metrics is not None
        ]
        item["average_rule_overall"] = _average(values)
    return breakdown


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 6)


def _safe_config(config: EvaluationConfig) -> dict:
    return config.model_dump()


def _run_id(suite: BenchmarkSuite, config: EvaluationConfig) -> str:
    flags = []
    if config.enable_context_compression:
        flags.append("compression")
    if config.enable_conflict_detection:
        flags.append("conflict")
    if config.enable_red_blue:
        flags.append("redblue")
    suffix = "_".join(flags) or "baseline"
    return f"{suite.name}_{config.backend}_{suffix}_{int(time.time())}"
