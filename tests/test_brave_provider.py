import httpx

from deepresearch_agent.search.brave_provider import BraveSearchProvider


class Response:
    def json(self):
        return {"web": {"results": [{"title": "A", "url": "https://a.test", "description": "Desc"}]}}

    def raise_for_status(self):
        return None


def test_brave_parses_web_results(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response())

    results = BraveSearchProvider("secret", api_base="https://api.test").search("q")

    assert results[0].provider == "brave"
    assert results[0].snippet == "Desc"
    assert results[0].domain == "a.test"
