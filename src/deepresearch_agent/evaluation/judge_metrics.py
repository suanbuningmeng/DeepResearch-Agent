from __future__ import annotations

from pydantic import BaseModel


class JudgeMetricResult(BaseModel):
    factuality: int | None = None
    coverage: int | None = None
    reasoning_depth: int | None = None
    citation_quality: int | None = None
    clarity: int | None = None
    overall: int | None = None


def extract_judge_metrics(trace: dict) -> JudgeMetricResult:
    score = trace.get("final_judge_score") or {}
    result = JudgeMetricResult(
        factuality=_optional_int(score.get("factuality")),
        coverage=_optional_int(score.get("coverage")),
        reasoning_depth=_optional_int(score.get("reasoning_depth")),
        citation_quality=_optional_int(score.get("citation_quality")),
        clarity=_optional_int(score.get("clarity")),
        overall=_optional_int(score.get("overall")),
    )
    red_blue_stats = trace.get("red_blue_stats") or {}
    if red_blue_stats.get("enabled") and red_blue_stats.get("final_overall_score") is not None:
        result.overall = _optional_int(red_blue_stats.get("final_overall_score"))
    return result


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return None
