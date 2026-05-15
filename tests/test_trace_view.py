from deepresearch_agent.ui.trace_view import (
    execution_step_rows,
    extract_trace_summary,
    get_compression_stats,
    get_memory_stats,
    task_rows,
)


def test_trace_summary_extracts_key_fields() -> None:
    trace = {
        "backend": "mock",
        "execution_mode": "dag",
        "final_judge_score": {"overall": 86},
        "forced_compose": False,
    }

    summary = extract_trace_summary(trace)

    assert summary["backend"] == "mock"
    assert summary["execution_mode"] == "dag"
    assert summary["overall"] == 86
    assert summary["forced_compose"] is False


def test_trace_view_handles_missing_memory_and_compression_stats() -> None:
    trace = {"planned_subtasks": [], "execution_steps": []}

    assert get_memory_stats(trace) is None
    assert get_compression_stats(trace) is None
    assert task_rows(trace) == []
    assert execution_step_rows(trace) == []
