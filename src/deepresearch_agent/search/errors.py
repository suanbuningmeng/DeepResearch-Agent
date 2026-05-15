from __future__ import annotations


class SearchProviderError(Exception):
    """Base error for search provider failures."""


class MissingSearchAPIKeyError(SearchProviderError):
    """Raised when a real search provider is selected without an API key."""


class SearchRateLimitError(SearchProviderError):
    """Raised when a provider reports rate limiting."""


class SearchRequestError(SearchProviderError):
    """Raised when a provider request fails."""
