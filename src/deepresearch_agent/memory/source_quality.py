from __future__ import annotations

from enum import StrEnum


class SourceQuality(StrEnum):
    REAL_URL = "real_url"
    NULL_SOURCE = "null_source"
    EXAMPLE_URL = "example_url"
    MOCK_SOURCE = "mock_source"
    MODEL_GENERATED = "model_generated"
    UNKNOWN = "unknown"


def classify_source_url(source_url: str | None) -> SourceQuality:
    if source_url is None or not str(source_url).strip():
        return SourceQuality.NULL_SOURCE
    normalized = str(source_url).strip().lower()
    if "example.com" in normalized:
        return SourceQuality.EXAMPLE_URL
    if normalized.startswith("mock://"):
        return SourceQuality.MOCK_SOURCE
    if normalized.startswith("model://"):
        return SourceQuality.MODEL_GENERATED
    if normalized.startswith(("http://", "https://")):
        return SourceQuality.REAL_URL
    return SourceQuality.UNKNOWN
