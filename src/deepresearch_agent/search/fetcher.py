from __future__ import annotations

import re

import httpx

from deepresearch_agent.search.schemas import FetchedDocument, SearchResult


class WebFetcher:
    def __init__(
        self,
        timeout: int = 15,
        max_chars: int = 4000,
        allow_network: bool = False,
    ) -> None:
        self.timeout = timeout
        self.max_chars = max_chars
        self.allow_network = allow_network

    def fetch(self, result: SearchResult) -> FetchedDocument:
        if not self.allow_network:
            return FetchedDocument(
                url=result.url,
                title=result.title,
                text=result.snippet[: self.max_chars],
                snippet=result.snippet,
                status_code=None,
                fetch_success=True,
            )
        try:
            response = httpx.get(result.url, timeout=self.timeout, follow_redirects=True)
            text = _html_to_text(response.text)[: self.max_chars]
            return FetchedDocument(
                url=result.url,
                title=result.title,
                text=text,
                snippet=result.snippet,
                status_code=response.status_code,
                fetch_success=response.status_code < 400,
                error=None if response.status_code < 400 else f"HTTP {response.status_code}",
            )
        except Exception as exc:
            return FetchedDocument(
                url=result.url,
                title=result.title,
                snippet=result.snippet,
                fetch_success=False,
                error=str(exc)[:200],
            )


def _html_to_text(html: str) -> str:
    cleaned = re.sub(r"<(script|style).*?</\1>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()
