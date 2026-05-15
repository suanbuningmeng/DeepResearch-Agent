from deepresearch_agent.search.mock_provider import MockSearchProvider


def test_mock_provider_returns_results() -> None:
    results = MockSearchProvider().search("semantic communication challenges", top_k=2)

    assert len(results) == 2
    assert results[0].url


def test_mock_provider_long_context_relevant_result() -> None:
    results = MockSearchProvider().search("LongBench long context LLM evaluation benchmark", top_k=2)

    assert "long" in results[0].title.lower()
    assert "context" in results[0].snippet.lower()


def test_mock_provider_unknown_query_returns_fallback() -> None:
    results = MockSearchProvider().search("obscure topic xyz", top_k=1)

    assert results[0].title == "General deep research system evaluation"
