from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SearchQuery(BaseModel):
    id: str
    query: str
    task_id: str | None = None
    intent: str | None = None


class SearchResult(BaseModel):
    id: str
    query_id: str
    title: str
    url: str
    snippet: str
    rank: int
    provider: str = "mock"
    domain: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FetchedDocument(BaseModel):
    url: str
    title: str | None = None
    text: str | None = None
    snippet: str | None = None
    status_code: int | None = None
    fetch_success: bool = False
    error: str | None = None


class CitationValidationResult(BaseModel):
    evidence_id: str
    source_url: str | None
    validation_status: str
    url_reachable: bool
    title_match_score: float
    content_overlap_score: float
    support_score: float
    reason: str


class SearchStats(BaseModel):
    enabled: bool
    provider: str = "mock"
    query_count: int = 0
    result_count: int = 0
    fetched_document_count: int = 0
    validated_citation_count: int = 0
    supported_count: int = 0
    partially_supported_count: int = 0
    unsupported_count: int = 0
    unreachable_count: int = 0
    no_source_count: int = 0
    search_queries: list[str] = Field(default_factory=list)
    top_domains: dict[str, int] = Field(default_factory=dict)
    fallback_used: bool = False
    provider_errors: list[str] = Field(default_factory=list)
    api_base_host: str | None = None
    search_timeout: int = 20
    search_provider_mode: str = "mock"
