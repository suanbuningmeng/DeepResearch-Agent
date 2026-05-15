from __future__ import annotations

from deepresearch_agent.search.schemas import SearchResult


def rank_search_results(results: list[SearchResult]) -> list[SearchResult]:
    deduped: dict[str, SearchResult] = {}
    for result in sorted(
        results,
        key=lambda item: (
            -float(item.metadata.get("retrieval_relevance_score") or 0.0),
            item.rank,
            item.domain or "",
            item.title,
        ),
    ):
        key = result.url or result.id
        if key not in deduped:
            deduped[key] = result
    return list(deduped.values())
