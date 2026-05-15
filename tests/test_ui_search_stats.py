from deepresearch_agent.ui.trace_view import get_search_stats


def test_trace_view_extracts_search_stats() -> None:
    trace = {"search_stats": {"enabled": True, "query_count": 2}}

    assert get_search_stats(trace)["query_count"] == 2


def test_trace_view_missing_search_stats() -> None:
    assert get_search_stats({}) is None
