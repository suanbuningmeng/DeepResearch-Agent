from __future__ import annotations

from deepresearch_agent.agents.judge import extract_scores_from_text, fallback_judge_score, normalize_judge_score
from deepresearch_agent.schemas import Evidence


def test_normalize_judge_score_handles_string_numbers() -> None:
    score = normalize_judge_score(
        {
            "factuality": "88",
            "coverage": "85/100",
            "reasoning_depth": 82.5,
            "citation_quality": "70",
            "clarity": "91",
            "overall": "86",
            "comments": "ok",
        }
    )

    assert score["factuality"] == 88
    assert score["coverage"] == 85
    assert score["reasoning_depth"] == 82
    assert score["overall"] == 86
    assert score["comments"] == "ok"


def test_normalize_judge_score_handles_nested_scores_and_bounds() -> None:
    score = normalize_judge_score(
        {
            "scores": {
                "factuality": -10,
                "coverage": 120,
                "reasoning_depth": "75",
            },
            "comments": "nested",
        }
    )

    assert score["factuality"] == 0
    assert score["coverage"] == 100
    assert score["reasoning_depth"] == 75
    assert score["citation_quality"] == 58
    assert score["comments"] == "nested"


def test_extract_scores_from_text_handles_common_variants() -> None:
    text = """
    Factuality = 85/100
    Coverage score is 80
    Reasoning depth: 78
    I would rate citation quality as 70.
    Clarity: 90
    Overall: 82
    """

    score = extract_scores_from_text(text)

    assert score is not None
    assert score["factuality"] == 85
    assert score["coverage"] == 80
    assert score["reasoning_depth"] == 78
    assert score["citation_quality"] == 70
    assert score["overall"] == 82


def test_fallback_judge_score_returns_complete_fields() -> None:
    evidence = Evidence(
        id="e1",
        task_id="t1",
        title="Supported source",
        content="A supported claim.",
        source_url="https://example.org/source",
        confidence=0.9,
        metadata={"citation_validation_status": "supported"},
    )

    score = fallback_judge_score(
        question="Question?",
        report="# Report\n\n## Abstract\nText\n\n## Key Findings\nText\n\n## Evidence Summary\nText\n\n## Limitations\nText\n\n## Conclusion\nText",
        evidences=[evidence],
        error="parse failed",
    )

    for field in ["factuality", "coverage", "reasoning_depth", "citation_quality", "clarity", "overall", "comments"]:
        assert field in score
    assert 0 <= score["overall"] <= 100
    assert "Fallback judge score was used" in score["comments"]
