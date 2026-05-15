import httpx
import pytest

from deepresearch_agent.search.errors import SearchProviderError, SearchRateLimitError
from deepresearch_agent.search.tavily_provider import TavilySearchProvider


class Response:
    def __init__(self, data=None, status_code=200) -> None:
        self._data = data or {}
        self.status_code = status_code

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=httpx.Request("POST", "https://api.test/search"), response=httpx.Response(self.status_code))


def test_tavily_parses_results(monkeypatch) -> None:
    def fake_post(*args, **kwargs):
        return Response({"results": [{"title": "A", "url": "https://a.test", "content": "Snippet"}]})

    monkeypatch.setattr(httpx, "post", fake_post)

    results = TavilySearchProvider("secret", api_base="https://api.test").search("q")

    assert results[0].provider == "tavily"
    assert results[0].snippet == "Snippet"
    assert results[0].domain == "a.test"


def test_tavily_missing_fields_do_not_crash(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response({"results": [{"url": "https://a.test"}]}))

    results = TavilySearchProvider("secret", api_base="https://api.test").search("q")

    assert results[0].title == "https://a.test"


@pytest.mark.parametrize("status_code", [401, 500])
def test_tavily_http_errors_are_sanitized(monkeypatch, status_code: int) -> None:
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response(status_code=status_code))

    with pytest.raises(SearchProviderError) as exc:
        TavilySearchProvider("secret-key", api_base="https://api.test").search("q")

    assert "secret-key" not in str(exc.value)
    assert str(status_code) in str(exc.value)


def test_tavily_rate_limit_error(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response(status_code=429))

    with pytest.raises(SearchRateLimitError):
        TavilySearchProvider("secret", api_base="https://api.test").search("q")
