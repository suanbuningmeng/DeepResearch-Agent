from __future__ import annotations

from collections import Counter

from deepresearch_agent.red_blue.schemas import RedCritique, RedIssueSeverity


def recount_critique(critique: RedCritique) -> RedCritique:
    counts = Counter(issue.severity for issue in critique.issues)
    critique.high_severity_count = counts[RedIssueSeverity.HIGH]
    critique.medium_severity_count = counts[RedIssueSeverity.MEDIUM]
    critique.low_severity_count = counts[RedIssueSeverity.LOW]
    return critique
