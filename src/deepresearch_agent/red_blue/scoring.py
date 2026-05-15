from __future__ import annotations

from typing import Any

from deepresearch_agent.schemas import JudgeScore


def overall_score(score: dict[str, Any] | JudgeScore | None) -> int | None:
    if score is None:
        return None
    if isinstance(score, JudgeScore):
        return score.overall
    try:
        value = score.get("overall")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return max(0, min(100, int(float(value))))
    except (TypeError, ValueError):
        return None
