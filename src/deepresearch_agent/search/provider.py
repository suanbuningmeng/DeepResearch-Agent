from __future__ import annotations

from urllib.parse import urlparse

import httpx

from deepresearch_agent.search.errors import SearchProviderError, SearchRateLimitError, SearchRequestError
from deepresearch_agent.search.schemas import SearchResult


class BaseSearchProvider:
    provider_name: str = "base"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError


def domain_from_url(url: str) -> str | None:
    return urlparse(url).netloc or None


def handle_http_error(provider_name: str, endpoint: str, exc: Exception) -> SearchProviderError:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        message = f"{provider_name} search request failed with status_code={status_code} endpoint={endpoint}"
        if status_code == 429:
            return SearchRateLimitError(message)
        return SearchRequestError(message)
    if isinstance(exc, httpx.HTTPError):
        return SearchRequestError(f"{provider_name} search request failed endpoint={endpoint}: {exc.__class__.__name__}")
    if isinstance(exc, ValueError):
        return SearchRequestError(f"{provider_name} search response was invalid JSON endpoint={endpoint}")
    return SearchProviderError(f"{provider_name} search failed endpoint={endpoint}: {exc.__class__.__name__}")
