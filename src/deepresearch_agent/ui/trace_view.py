from __future__ import annotations

from typing import Any


def extract_trace_summary(trace: dict[str, Any]) -> dict[str, Any]:
    score = trace.get("final_judge_score") or {}
    return {
        "backend": trace.get("backend"),
        "execution_mode": trace.get("execution_mode"),
        "overall": score.get("overall"),
        "forced_compose": trace.get("forced_compose", False),
        "cancelled_tasks": trace.get("cancelled_tasks", []),
        "degraded_tasks": trace.get("degraded_tasks", []),
        "replan_rounds": trace.get("replan_rounds", 0),
        "planner_stats": trace.get("planner_stats"),
        "evidence_sufficiency": trace.get("evidence_sufficiency"),
        "report_quality_check": trace.get("report_quality_check"),
        "outputs": trace.get("outputs", {}),
    }


def task_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in trace.get("planned_subtasks", []):
        output = task.get("output") or {}
        evidence_ids = output.get("evidence_ids") or []
        rows.append(
            {
                "id": task.get("id"),
                "name": task.get("name"),
                "state": task.get("state"),
                "retry_count": task.get("retry_count", 0),
                "evidence_count": len(evidence_ids),
                "timeout_seconds": task.get("timeout_seconds"),
            }
        )
    return rows


def execution_step_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    return list(trace.get("execution_steps", []))


def get_memory_stats(trace: dict[str, Any]) -> dict[str, Any] | None:
    stats = trace.get("memory_stats")
    return stats if isinstance(stats, dict) else None


def get_compression_stats(trace: dict[str, Any]) -> dict[str, Any] | None:
    stats = trace.get("compression_stats")
    return stats if isinstance(stats, dict) else None


def get_conflict_stats(trace: dict[str, Any]) -> dict[str, Any] | None:
    stats = trace.get("conflict_stats")
    return stats if isinstance(stats, dict) else None


def conflict_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    stats = get_conflict_stats(trace) or {}
    conflicts = stats.get("conflicts") or []
    rows: list[dict[str, Any]] = []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            continue
        rows.append(
            {
                "id": conflict.get("id"),
                "left_evidence_id": conflict.get("left_evidence_id"),
                "right_evidence_id": conflict.get("right_evidence_id"),
                "conflict_type": conflict.get("conflict_type"),
                "severity": conflict.get("severity"),
                "resolution_action": conflict.get("resolution_action"),
                "reason": conflict.get("reason"),
            }
        )
    return rows


def get_red_blue_stats(trace: dict[str, Any]) -> dict[str, Any] | None:
    stats = trace.get("red_blue_stats")
    return stats if isinstance(stats, dict) else None


def get_search_stats(trace: dict[str, Any]) -> dict[str, Any] | None:
    stats = trace.get("search_stats")
    return stats if isinstance(stats, dict) else None


def search_stats_summary(trace: dict[str, Any]) -> dict[str, Any] | None:
    stats = get_search_stats(trace)
    if stats is None:
        return None
    return {
        "enabled": stats.get("enabled"),
        "provider": stats.get("provider"),
        "query_count": stats.get("query_count"),
        "result_count": stats.get("result_count"),
        "fetched_document_count": stats.get("fetched_document_count"),
        "validated_citation_count": stats.get("validated_citation_count"),
        "supported_count": stats.get("supported_count"),
        "partially_supported_count": stats.get("partially_supported_count"),
        "unsupported_count": stats.get("unsupported_count"),
        "unreachable_count": stats.get("unreachable_count"),
        "no_source_count": stats.get("no_source_count"),
        "provider_errors": stats.get("provider_errors", []),
        "api_base_host": stats.get("api_base_host"),
        "search_timeout": stats.get("search_timeout"),
        "search_provider_mode": stats.get("search_provider_mode"),
    }


def red_blue_round_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    stats = get_red_blue_stats(trace) or {}
    rounds = stats.get("rounds") or []
    rows: list[dict[str, Any]] = []
    for round_trace in rounds:
        if not isinstance(round_trace, dict):
            continue
        rows.append(
            {
                "round_id": round_trace.get("round_id"),
                "red_issue_count": round_trace.get("red_issue_count"),
                "high_severity_count": round_trace.get("high_severity_count"),
                "patches_applied_count": round_trace.get("patches_applied_count"),
                "judge_score_before": round_trace.get("judge_score_before"),
                "judge_score_after": round_trace.get("judge_score_after"),
                "score_delta": round_trace.get("score_delta"),
                "stopped_reason": round_trace.get("stopped_reason"),
                "red_fallback_used": round_trace.get("red_fallback_used"),
                "blue_fallback_used": round_trace.get("blue_fallback_used"),
            }
        )
    return rows


def evaluation_summary_fields(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark_name": summary.get("benchmark_name"),
        "case_count": summary.get("case_count"),
        "success_count": summary.get("success_count"),
        "error_count": summary.get("error_count"),
        "average_rule_overall": summary.get("average_rule_overall"),
        "average_judge_overall": summary.get("average_judge_overall"),
        "bootstrap_ci_rule_overall": summary.get("bootstrap_ci_rule_overall"),
        "bootstrap_ci_judge_overall": summary.get("bootstrap_ci_judge_overall"),
    }
