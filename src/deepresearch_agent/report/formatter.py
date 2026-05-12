from __future__ import annotations

from deepresearch_agent.schemas import JudgeScore


def format_report_with_score(markdown: str, score: JudgeScore) -> str:
    score_block = f"""

---

## Judge Score
- Factuality: {score.factuality}
- Coverage: {score.coverage}
- Reasoning Depth: {score.reasoning_depth}
- Citation Quality: {score.citation_quality}
- Clarity: {score.clarity}
- Overall: {score.overall}

{score.comments}
"""
    return markdown.rstrip() + score_block
