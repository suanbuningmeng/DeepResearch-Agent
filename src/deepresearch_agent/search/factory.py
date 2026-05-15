from __future__ import annotations

import os

from deepresearch_agent.search.arxiv_provider import ArxivSearchProvider, PaperSearchProvider
from deepresearch_agent.search.brave_provider import BraveSearchProvider
from deepresearch_agent.search.errors import MissingSearchAPIKeyError
from deepresearch_agent.search.exa_provider import ExaSearchProvider
from deepresearch_agent.search.mock_provider import MockSearchProvider
from deepresearch_agent.search.provider import BaseSearchProvider
from deepresearch_agent.search.serper_provider import SerperSearchProvider
from deepresearch_agent.search.tavily_provider import TavilySearchProvider
from deepresearch_agent.search.web_provider import WebSearchProvider


DEFAULT_KEY_ENVS = {
    "tavily": "TAVILY_API_KEY",
    "exa": "EXA_API_KEY",
    "serper": "SERPER_API_KEY",
    "brave": "BRAVE_API_KEY",
    "web": "SEARCH_API_KEY",
    "generic": "SEARCH_API_KEY",
}

KEYLESS_PROVIDERS = {"mock", "arxiv", "paper"}


def create_search_provider(
    provider_name: str,
    api_key: str | None = None,
    api_key_env: str | None = None,
    api_base: str | None = None,
    timeout: int = 20,
) -> BaseSearchProvider:
    provider = provider_name.strip().lower()
    if provider == "mock":
        return MockSearchProvider()
    if provider == "arxiv":
        return ArxivSearchProvider(api_base=api_base or "https://export.arxiv.org/api/query", timeout=timeout)
    if provider == "paper":
        return PaperSearchProvider(api_base=api_base or "https://export.arxiv.org/api/query", timeout=timeout)
    if provider not in {"tavily", "exa", "serper", "brave", "web", "generic"}:
        raise ValueError("Unknown search provider. Expected one of: mock, arxiv, paper, tavily, exa, serper, brave, web.")

    resolved_env = api_key_env or DEFAULT_KEY_ENVS[provider]
    resolved_key = api_key or os.getenv(resolved_env, "")
    if not resolved_key:
        raise MissingSearchAPIKeyError(f"{resolved_env} is required for search provider '{provider}'.")

    if provider == "tavily":
        return TavilySearchProvider(api_key=resolved_key, api_base=api_base or "https://api.tavily.com", timeout=timeout)
    if provider == "exa":
        return ExaSearchProvider(api_key=resolved_key, api_base=api_base or "https://api.exa.ai", timeout=timeout)
    if provider == "serper":
        return SerperSearchProvider(api_key=resolved_key, api_base=api_base or "https://google.serper.dev", timeout=timeout)
    if provider == "brave":
        return BraveSearchProvider(api_key=resolved_key, api_base=api_base or "https://api.search.brave.com/res/v1", timeout=timeout)
    return WebSearchProvider(
        api_key=resolved_key,
        api_base=api_base,
        provider_name="web",
        timeout=timeout,
    )


def default_api_key_env(provider_name: str) -> str:
    if provider_name.strip().lower() in KEYLESS_PROVIDERS:
        return ""
    return DEFAULT_KEY_ENVS.get(provider_name.strip().lower(), "SEARCH_API_KEY")
