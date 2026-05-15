from __future__ import annotations

import httpx

from deepresearch_agent.search.provider import BaseSearchProvider, domain_from_url, handle_http_error
from deepresearch_agent.search.schemas import SearchResult


class SerperSearchProvider(BaseSearchProvider):
    provider_name = "serper"

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://google.serper.dev",
        timeout: int = 20,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        endpoint = f"{self.api_base}/search"
        try:
            response = httpx.post(
                endpoint,
                headers={"X-API-KEY": self.api_key, "Content-Type": "application/json"},
                json={"q": query, "num": top_k},
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise handle_http_error(self.provider_name, endpoint, exc) from exc

        organic = data.get("organic") or []
        return [
            SearchResult(
                id=f"serper_{index}",
                query_id=query[:80],
                title=str(item.get("title") or item.get("link") or f"Serper result {index}"),
                url=str(item.get("link") or ""),
                snippet=str(item.get("snippet") or ""),
                rank=int(item.get("position") or index),
                provider=self.provider_name,
                domain=domain_from_url(str(item.get("link") or "")),
            )
            for index, item in enumerate(organic[:top_k], start=1)
            if isinstance(item, dict) and item.get("link")
        ]
