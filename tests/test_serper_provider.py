import httpx

from deepresearch_agent.search.serper_provider import SerperSearchProvider


class Response:
    def json(self):
        return {"organic": [{"title": "A", "link": "https://a.test", "snippet": "Snippet", "position": 3}]}

    def raise_for_status(self):
        return None


def test_serper_parses_organic_results(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: Response())

    results = SerperSearchProvider("secret", api_base="https://api.test").search("q")

    assert results[0].provider == "serper"
    assert results[0].rank == 3
    assert results[0].url == "https://a.test"
