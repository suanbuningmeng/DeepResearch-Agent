from __future__ import annotations

import httpx

from deepresearch_agent.search.provider import BaseSearchProvider, domain_from_url, handle_http_error
from deepresearch_agent.search.schemas import SearchResult


class ExaSearchProvider(BaseSearchProvider):
    provider_name = "exa"

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.exa.ai",
        timeout: int = 20,
        include_highlights: bool = True,
    ) -> None:
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self.include_highlights = include_highlights

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        endpoint = f"{self.api_base}/search"
        payload = {
            "query": query,
            "numResults": top_k,
            "type": "auto",
            "contents": {"highlights": self.include_highlights},
        }
        try:
            response = httpx.post(
                endpoint,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
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
                id=f"exa_{index}",
                query_id=query[:80],
                title=str(item.get("title") or item.get("url") or f"Exa result {index}"),
                url=str(item.get("url") or ""),
                snippet=_snippet(item),
                rank=index,
                provider=self.provider_name,
                domain=domain_from_url(str(item.get("url") or "")),
            )
            for index, item in enumerate(results[:top_k], start=1)
            if isinstance(item, dict) and item.get("url")
        ]


def _snippet(item: dict) -> str:
    highlights = item.get("highlights")
    if isinstance(highlights, list) and highlights:
        return " ".join(str(highlight) for highlight in highlights)
    text = str(item.get("text") or "")
    if text:
        return text[:500]
    return str(item.get("summary") or "")
