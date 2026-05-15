from deepresearch_agent.ui.trace_view import get_red_blue_stats, red_blue_round_rows


def test_trace_view_extracts_red_blue_stats() -> None:
    trace = {
        "red_blue_stats": {
            "enabled": True,
            "rounds_completed": 1,
            "rounds": [
                {
                    "round_id": 1,
                    "red_issue_count": 2,
                    "high_severity_count": 1,
                    "patches_applied_count": 1,
                    "judge_score_before": 80,
                    "judge_score_after": 84,
                    "score_delta": 4,
                    "stopped_reason": "round_completed",
                    "red_fallback_used": False,
                    "blue_fallback_used": True,
                }
            ],
        }
    }

    assert get_red_blue_stats(trace)["rounds_completed"] == 1
    assert red_blue_round_rows(trace)[0]["score_delta"] == 4


def test_trace_view_handles_missing_red_blue_stats() -> None:
    trace = {}

    assert get_red_blue_stats(trace) is None
    assert red_blue_round_rows(trace) == []
