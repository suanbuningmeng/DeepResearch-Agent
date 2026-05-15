from __future__ import annotations

import httpx

from deepresearch_agent.search.provider import BaseSearchProvider, domain_from_url, handle_http_error
from deepresearch_agent.search.schemas import SearchResult


class TavilySearchProvider(BaseSearchProvider):
    provider_name = "tavily"

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.tavily.com",
        timeout: int = 20,
        include_raw_content: bool = False,
        search_depth: str = "basic",
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.include_raw_content = include_raw_content
        self.search_depth = search_depth

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        endpoint = f"{self.api_base}/search"
        payload = {
            "query": query,
            "max_results": top_k,
            "search_depth": self.search_depth,
            "include_answer": False,
            "include_raw_content": self.include_raw_content,
        }
        try:
            response = httpx.post(
                endpoint,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise handle_http_error(self.provider_name, endpoint, exc) from exc

        results = data.get("results") or []
        return [
            SearchResult(
                id=f"tavily_{index}",
                query_id=query[:80],
                title=str(item.get("title") or item.get("url") or f"Tavily result {index}"),
                url=str(item.get("url") or ""),
                snippet=str(item.get("content") or item.get("snippet") or ""),
                rank=index,
                provider=self.provider_name,
                domain=domain_from_url(str(item.get("url") or "")),
            )
            for index, item in enumerate(results[:top_k], start=1)
            if isinstance(item, dict) and item.get("url")
        ]
