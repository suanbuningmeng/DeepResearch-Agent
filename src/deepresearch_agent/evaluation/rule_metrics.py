from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from deepresearch_agent.evaluation.benchmark import BenchmarkExample


class RuleMetricResult(BaseModel):
    example_id: str
    key_point_coverage: float
    citation_coverage: float
    source_quality_score: float
    hallucination_risk: float
    report_completeness: float
    conflict_penalty: float
    compression_ratio: float | None = None
    red_blue_improvement: float | None = None
    rule_overall: float


ABSOLUTE_TERMS = {"always", "never", "definitely", "prove", "guaranteed", "must", "all", "none"}


def compute_rule_metrics(example: BenchmarkExample, report: str, trace: dict[str, Any]) -> RuleMetricResult:
    key_point_coverage = _key_point_coverage(example.expected_key_points, report)
    citation_coverage = _citation_coverage(report)
    source_quality_score = _source_quality_score(trace)
    hallucination_risk = _hallucination_risk(report, citation_coverage, trace)
    report_completeness = _report_completeness(trace)
    conflict_penalty = _conflict_penalty(trace)
    compression_ratio = _optional_float((trace.get("compression_stats") or {}).get("compression_ratio"))
    red_blue_improvement = _optional_float((trace.get("red_blue_stats") or {}).get("score_delta"))
    rule_overall = 100 * (
        0.30 * key_point_coverage
        + 0.20 * citation_coverage
        + 0.15 * source_quality_score
        + 0.15 * report_completeness
        + 0.10 * (1 - hallucination_risk)
        + 0.10 * (1 - conflict_penalty)
    )
    return RuleMetricResult(
        example_id=example.id,
        key_point_coverage=round(key_point_coverage, 6),
        citation_coverage=round(citation_coverage, 6),
        source_quality_score=round(source_quality_score, 6),
        hallucination_risk=round(hallucination_risk, 6),
        report_completeness=round(report_completeness, 6),
        conflict_penalty=round(conflict_penalty, 6),
        compression_ratio=compression_ratio,
        red_blue_improvement=red_blue_improvement,
        rule_overall=round(max(0.0, min(100.0, rule_overall)), 6),
    )


def _key_point_coverage(key_points: list[str], report: str) -> float:
    if not key_points:
        return 1.0
    normalized_report = _normalize(report)
    matched = 0
    for key_point in key_points:
        normalized_key = _normalize(key_point)
        if normalized_key in normalized_report:
            matched += 1
            continue
        key_tokens = set(normalized_key.split())
        report_tokens = set(normalized_report.split())
        if key_tokens and len(key_tokens & report_tokens) / len(key_tokens) >= 0.5:
            matched += 1
    return matched / len(key_points)


def _citation_coverage(report: str) -> float:
    evidence_section = _extract_section(report, "Evidence Summary") or report
    bullets = [line.strip() for line in evidence_section.splitlines() if re.match(r"^[-*]\s+", line.strip())]
    if not bullets:
        return 0.0
    cited = [
        bullet
        for bullet in bullets
        if re.search(r"\[[^\]]*(?:evidence|task_|e\d+)[^\]]*\]", bullet, flags=re.IGNORECASE)
        or "http://" in bullet
        or "https://" in bullet
        or "mock://" in bullet
    ]
    return len(cited) / len(bullets)


def _source_quality_score(trace: dict[str, Any]) -> float:
    search_stats = trace.get("search_stats") or {}
    if search_stats.get("enabled") and int(search_stats.get("validated_citation_count") or 0) > 0:
        total = int(search_stats.get("validated_citation_count") or 0)
        score = (
            int(search_stats.get("supported_count") or 0) * 1.0
            + int(search_stats.get("partially_supported_count") or 0) * 0.55
            + int(search_stats.get("unsupported_count") or 0) * 0.15
            + int(search_stats.get("unreachable_count") or 0) * 0.1
            + int(search_stats.get("no_source_count") or 0) * 0.0
        )
        return score / total
    summary = ((trace.get("memory_stats") or {}).get("source_quality_summary") or {})
    total = sum(int(count) for count in summary.values()) if summary else 0
    if total == 0:
        return 0.5
    weighted = (
        int(summary.get("real_url", 0)) * 1.0
        + int(summary.get("null_source", 0)) * 0.55
        + int(summary.get("example_url", 0)) * 0.2
        + int(summary.get("mock_source", 0)) * 0.4
        + int(summary.get("model_generated", 0)) * 0.35
        + int(summary.get("unknown", 0)) * 0.5
    )
    return weighted / total


def _hallucination_risk(report: str, citation_coverage: float, trace: dict[str, Any]) -> float:
    tokens = set(_normalize(report).split())
    absolute_hits = len(tokens & ABSOLUTE_TERMS)
    if absolute_hits == 0:
        base = 0.0
    else:
        base = 0.25 * absolute_hits + (1.0 - citation_coverage) * 0.5
    search_stats = trace.get("search_stats") or {}
    if search_stats.get("enabled"):
        unsupported = int(search_stats.get("unsupported_count") or 0) + int(search_stats.get("no_source_count") or 0)
        validated = max(1, int(search_stats.get("validated_citation_count") or 0))
        base += 0.4 * (unsupported / validated)
    return min(1.0, base)


def _report_completeness(trace: dict[str, Any]) -> float:
    quality = trace.get("report_quality_check") or {}
    if quality.get("is_complete") is True:
        return 1.0
    missing = len(quality.get("missing_sections") or [])
    return max(0.0, 1.0 - 0.2 * missing - (0.2 if quality.get("report_incomplete") else 0.0))


def _conflict_penalty(trace: dict[str, Any]) -> float:
    stats = trace.get("conflict_stats") or {}
    if not stats.get("enabled"):
        return 0.0
    marked = len(stats.get("marked_conflict_evidence_ids") or [])
    unresolved = int(stats.get("conflict_count") or 0) - len(stats.get("dropped_evidence_ids") or [])
    return min(1.0, 0.1 * marked + 0.05 * max(0, unresolved))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s-]", " ", text.lower())).strip()


def _extract_section(markdown: str, section: str) -> str:
    lines = markdown.splitlines()
    start = None
    for index, line in enumerate(lines):
        if re.match(rf"^\s*#+\s+{re.escape(section)}\b", line, flags=re.IGNORECASE):
            start = index + 1
            continue
        if start is not None and re.match(r"^\s*#+\s+", line):
            return "\n".join(lines[start:index])
    if start is None:
        return ""
    return "\n".join(lines[start:])


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
