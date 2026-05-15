from deepresearch_agent.conflict.heuristics import (
    detect_numeric_direction_conflict,
    has_antonym_conflict,
)


def test_has_antonym_conflict_detects_improve_vs_degrade() -> None:
    detected, reason = has_antonym_conflict(
        "Method A can improve accuracy on long-context evaluation.",
        "Method A may degrade accuracy on long-context evaluation.",
    )

    assert detected is True
    assert reason is not None
    assert "improve" in reason
    assert "degrade" in reason


def test_detect_numeric_direction_conflict_for_accuracy_directions() -> None:
    detected, reason = detect_numeric_direction_conflict(
        "Method A increases accuracy by 10 percent.",
        "Method A decreases accuracy by 10 percent.",
    )

    assert detected is True
    assert reason is not None
    assert "accuracy" in reason


def test_unrelated_text_does_not_report_conflict() -> None:
    detected, reason = has_antonym_conflict(
        "Needle tests measure retrieval in long prompts.",
        "SQLite stores evidence records for later retrieval.",
    )

    assert detected is False
    assert reason is None
