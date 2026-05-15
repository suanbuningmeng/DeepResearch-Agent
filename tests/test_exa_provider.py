import httpx

from deepresearch_agent.search.exa_provider import ExaSearchProvider


class Response:
    def __init__(self, data) -> None:
        self._data = data

    def json(self):
        return self._data

    def raise_for_status(self):
        return None


def test_exa_parses_highlights(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response({"results": [{"title": "A", "url": "https://a.test", "highlights": ["one", "two"]}]}))

    results = ExaSearchProvider("secret", api_base="https://api.test").search("q")

    assert results[0].provider == "exa"
    assert results[0].snippet == "one two"


def test_exa_empty_results(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response({"results": []}))

    assert ExaSearchProvider("secret", api_base="https://api.test").search("q") == []
