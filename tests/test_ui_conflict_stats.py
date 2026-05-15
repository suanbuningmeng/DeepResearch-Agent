from deepresearch_agent.ui.trace_view import conflict_rows, get_conflict_stats


def test_trace_view_extracts_conflict_stats() -> None:
    trace = {
        "conflict_stats": {
            "enabled": True,
            "conflict_count": 1,
            "conflicts": [
                {
                    "id": "conflict_e1_e2_1",
                    "left_evidence_id": "e1",
                    "right_evidence_id": "e2",
                    "conflict_type": "semantic_opposition",
                    "severity": "medium",
                    "resolution_action": "mark_for_writer",
                    "reason": "test",
                }
            ],
        }
    }

    assert get_conflict_stats(trace)["conflict_count"] == 1
    assert conflict_rows(trace)[0]["conflict_type"] == "semantic_opposition"


def test_trace_view_handles_missing_conflict_stats() -> None:
    trace = {}

    assert get_conflict_stats(trace) is None
    assert conflict_rows(trace) == []
