from __future__ import annotations

import httpx

from deepresearch_agent.search.provider import BaseSearchProvider, domain_from_url, handle_http_error
from deepresearch_agent.search.schemas import SearchResult


class BraveSearchProvider(BaseSearchProvider):
    provider_name = "brave"

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.search.brave.com/res/v1",
        timeout: int = 20,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        endpoint = f"{self.api_base}/web/search"
        try:
            response = httpx.get(
                endpoint,
                headers={"X-Subscription-Token": self.api_key, "Accept": "application/json"},
                params={"q": query, "count": top_k},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise handle_http_error(self.provider_name, endpoint, exc) from exc

        items = ((data.get("web") or {}).get("results") or [])
        return [
            SearchResult(
                id=f"brave_{index}",
                query_id=query[:80],
                title=str(item.get("title") or item.get("url") or f"Brave result {index}"),
                url=str(item.get("url") or ""),
                snippet=str(item.get("description") or item.get("snippet") or ""),
                rank=index,
                provider=self.provider_name,
                domain=domain_from_url(str(item.get("url") or "")),
            )
            for index, item in enumerate(items[:top_k], start=1)
            if isinstance(item, dict) and item.get("url")
        ]
