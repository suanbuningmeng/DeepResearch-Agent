from __future__ import annotations

import os

import httpx

from deepresearch_agent.search.errors import MissingSearchAPIKeyError
from deepresearch_agent.search.provider import BaseSearchProvider, domain_from_url, handle_http_error
from deepresearch_agent.search.schemas import SearchResult


class WebSearchProvider(BaseSearchProvider):
    def __init__(
        self,
        api_key: str | None = None,
        api_key_env: str = "SEARCH_API_KEY",
        api_base: str | None = None,
        provider_name: str | None = None,
        timeout: int = 20,
    ) -> None:
        self.api_key_env = api_key_env
        self.api_key = api_key or os.getenv(api_key_env, "")
        self.api_base = api_base or os.getenv("SEARCH_API_BASE", "")
        self.provider_name = provider_name or os.getenv("SEARCH_PROVIDER_NAME", "web")
        self.timeout = timeout
        if not self.api_key:
            raise MissingSearchAPIKeyError(f"{api_key_env} is required for web search provider.")
        if not self.api_base:
            raise ValueError("SEARCH_API_BASE or --search-api-base is required for web search provider.")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        try:
            response = httpx.get(
                self.api_base,
                params={"q": query, "num": top_k},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise handle_http_error(self.provider_name, self.api_base, exc) from exc
        items = data.get("results") or data.get("items") or []
        results: list[SearchResult] = []
        for rank, item in enumerate(items[:top_k], start=1):
            url = str(item.get("url") or item.get("link") or "")
            if not url:
                continue
            results.append(
                SearchResult(
                    id=f"{self.provider_name}_{rank}",
                    query_id=query[:80],
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("snippet") or item.get("description") or ""),
                    rank=rank,
                    provider=self.provider_name,
                    domain=domain_from_url(url),
                )
            )
        return results
