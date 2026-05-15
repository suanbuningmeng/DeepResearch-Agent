from __future__ import annotations

from deepresearch_agent.agents.judge import coerce_judge_data


def test_judge_coerces_nested_scores() -> None:
    score = coerce_judge_data(
        {
            "scores": {
                "factual_accuracy": "82",
                "coverage": 80,
                "reasoning": 78,
                "citations": 72,
                "readability": 88,
            },
            "comments": "nested",
        }
    )

    assert score is not None
    assert score["factuality"] == 82
    assert score["reasoning_depth"] == 78
    assert score["citation_quality"] == 72
    assert score["clarity"] == 88
    assert score["overall"] == 80


def test_judge_coerces_metric_score_list() -> None:
    score = coerce_judge_data(
        [
            {"metric": "factuality", "score": 80},
            {"metric": "coverage", "score": 75},
            {"metric": "reasoning_depth", "score": 70},
            {"metric": "final_score", "score": 76},
        ]
    )

    assert score is not None
    assert score["factuality"] == 80
    assert score["coverage"] == 75
    assert score["reasoning_depth"] == 70
    assert score["overall"] == 76
