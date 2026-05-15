from deepresearch_agent.search.errors import MissingSearchAPIKeyError, SearchProviderError, SearchRateLimitError, SearchRequestError


def test_missing_key_error_does_not_include_key() -> None:
    error = MissingSearchAPIKeyError("TAVILY_API_KEY is required.")

    assert "secret" not in str(error)


def test_rate_limit_error_is_search_provider_error() -> None:
    error = SearchRateLimitError("rate limited")

    assert isinstance(error, SearchProviderError)


def test_request_error_is_search_provider_error() -> None:
    error = SearchRequestError("request failed")

    assert isinstance(error, SearchProviderError)
