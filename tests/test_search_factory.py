import pytest

from deepresearch_agent.search import ArxivSearchProvider, MockSearchProvider, PaperSearchProvider, create_search_provider
from deepresearch_agent.search.errors import MissingSearchAPIKeyError
from deepresearch_agent.search.tavily_provider import TavilySearchProvider


def test_factory_returns_mock_provider() -> None:
    assert isinstance(create_search_provider("mock"), MockSearchProvider)


def test_factory_returns_arxiv_without_api_key() -> None:
    assert isinstance(create_search_provider("arxiv"), ArxivSearchProvider)


def test_factory_returns_paper_without_api_key() -> None:
    assert isinstance(create_search_provider("paper"), PaperSearchProvider)


def test_factory_missing_key_raises(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(MissingSearchAPIKeyError):
        create_search_provider("tavily")


def test_factory_unknown_provider_raises() -> None:
    with pytest.raises(ValueError):
        create_search_provider("unknown")


def test_factory_reads_env_without_leaking_key(monkeypatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "secret-key")

    provider = create_search_provider("tavily")

    assert isinstance(provider, TavilySearchProvider)
    assert "secret-key" not in repr(provider)
