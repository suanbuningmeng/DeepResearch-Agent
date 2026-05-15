from __future__ import annotations

from collections import Counter
from typing import Any

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import Evidence, TaskNode
from deepresearch_agent.search.citation_validator import CitationValidator
from deepresearch_agent.search.fetcher import WebFetcher
from deepresearch_agent.search.provider import BaseSearchProvider
from deepresearch_agent.search.query_generator import SearchQueryGenerator
from deepresearch_agent.search.schemas import FetchedDocument, SearchResult, SearchStats
from deepresearch_agent.search.source_ranker import rank_search_results
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


class GroundedResearchBuilder:
    def __init__(
        self,
        search_provider: BaseSearchProvider,
        fetcher: WebFetcher,
        validator: CitationValidator,
    ) -> None:
        self.search_provider = search_provider
        self.fetcher = fetcher
        self.validator = validator

    async def build_evidence_for_task(
        self,
        question: str,
        task: TaskNode,
        llm: BaseLLM,
        max_queries: int = 3,
        search_top_k: int = 3,
    ) -> tuple[list[Evidence], SearchStats]:
        query_generator = SearchQueryGenerator(llm)
        queries = await query_generator.generate_queries(question, task, max_queries=max_queries)
        results: list[SearchResult] = []
        provider_errors: list[str] = []
        for query in queries:
            try:
                results.extend(self.search_provider.search(query.query, top_k=search_top_k))
            except Exception as exc:
                provider_errors.append(str(exc)[:300])
        results = rank_search_results(results)
        if not results:
            stats = SearchStats(
                enabled=True,
                provider=self.search_provider.provider_name,
                query_count=len(queries),
                result_count=0,
                fetched_document_count=0,
                validated_citation_count=0,
                search_queries=[query.query for query in queries],
                top_domains={},
                fallback_used=True,
                provider_errors=provider_errors,
                api_base_host=_api_base_host(self.search_provider),
                search_timeout=int(getattr(self.search_provider, "timeout", 20)),
                search_provider_mode="mock" if self.search_provider.provider_name == "mock" else "raw_search",
            )
            return [], stats
        documents_by_url: dict[str, FetchedDocument] = {}
        for result in results:
            if result.url not in documents_by_url:
                documents_by_url[result.url] = self.fetcher.fetch(result)

        evidences = await self._llm_evidences(question, task, llm, results, documents_by_url)
        if not evidences:
            evidences = self._fallback_evidences(task, results)

        validated: list[Evidence] = []
        status_counts: Counter[str] = Counter()
        for index, evidence in enumerate(evidences, start=1):
            result = _result_for_evidence(evidence, results, index)
            if result is not None:
                evidence.source_url = result.url
                evidence.metadata["search_query"] = result.query_id
                evidence.metadata["retrieval_relevance_score"] = result.metadata.get("retrieval_relevance_score")
                evidence.metadata["retrieval_relevance_reason"] = result.metadata.get("retrieval_relevance_reason")
                evidence.metadata["reranked_by_relevance"] = bool(result.metadata.get("reranked_by_relevance", False))
                if result.metadata:
                    evidence.metadata["source_metadata"] = result.metadata
                    if result.metadata.get("source_type"):
                        evidence.metadata["source_type"] = result.metadata["source_type"]
            evidence.metadata["grounded"] = True
            validation = self.validator.validate(evidence, documents_by_url.get(evidence.source_url or ""))
            evidence.metadata["citation_validation_status"] = validation.validation_status
            evidence.metadata["citation_support_score"] = validation.support_score
            evidence.metadata["citation_reason"] = validation.reason
            evidence.confidence = _grounded_confidence(evidence, validation.support_score)
            status_counts[validation.validation_status] += 1
            validated.append(evidence)

        stats = SearchStats(
            enabled=True,
            provider=self.search_provider.provider_name,
            query_count=len(queries),
            result_count=len(results),
            fetched_document_count=sum(1 for document in documents_by_url.values() if document.fetch_success),
            validated_citation_count=len(validated),
            supported_count=status_counts["supported"],
            partially_supported_count=status_counts["partially_supported"],
            unsupported_count=status_counts["unsupported"],
            unreachable_count=status_counts["url_unreachable"],
            no_source_count=status_counts["no_source"],
            search_queries=[query.query for query in queries],
            top_domains=dict(Counter(result.domain or "unknown" for result in results)),
            fallback_used=bool(query_generator.last_stats.get("fallback_used", False) or provider_errors),
            provider_errors=provider_errors,
            api_base_host=_api_base_host(self.search_provider),
            search_timeout=int(getattr(self.search_provider, "timeout", 20)),
            search_provider_mode="mock" if self.search_provider.provider_name == "mock" else "raw_search",
        )
        return validated, stats

    async def _llm_evidences(
        self,
        question: str,
        task: TaskNode,
        llm: BaseLLM,
        results: list[SearchResult],
        documents_by_url: dict[str, FetchedDocument],
    ) -> list[Evidence]:
        search_context = "\n".join(
            f"- {result.title}: {result.snippet} ({result.url})"
            for result in results[:6]
        )
        prompt = (
            "You are a grounded researcher. Generate concise evidence using only these search results.\n"
            f"Question: {question}\n"
            f"Task ID: {task.id}\n"
            f"Task name: {task.name}\n"
            f"Task description: {task.description}\n"
            f"Search results:\n{search_context}\n"
            "Return valid JSON only with schema {\"evidences\":[{\"title\":\"...\",\"content\":\"...\",\"confidence\":0.8}]}."
        )
        try:
            raw = await llm.agenerate(prompt, prompt_type="researcher")
            data = safe_json_loads(strip_thinking(raw))
            evidences: list[Evidence] = []
            for index, item in enumerate(data.get("evidences", [])[: max(1, min(3, len(results) or 1))], start=1):
                if not isinstance(item, dict):
                    continue
                result = results[(index - 1) % len(results)] if results else None
                content = str(item.get("content") or item.get("summary") or "")
                if result and not _has_overlap(content, documents_by_url.get(result.url)):
                    content = _short_result_snippet(result)
                metadata = {"task_name": task.name}
                if result and result.metadata:
                    metadata["source_metadata"] = result.metadata
                    if result.metadata.get("source_type"):
                        metadata["source_type"] = result.metadata["source_type"]
                evidences.append(
                    Evidence(
                        id=f"{task.id}_grounded_evidence_{index}",
                        task_id=task.id,
                        title=str(item.get("title") or (result.title if result else f"Grounded evidence {index}")),
                        content=content,
                        source_url=result.url if result else None,
                        confidence=_confidence(item.get("confidence", 0.75)),
                        metadata=metadata,
                    )
                )
            return evidences
        except Exception:
            return []

    def _fallback_evidences(self, task: TaskNode, results: list[SearchResult]) -> list[Evidence]:
        return [
            Evidence(
                id=f"{task.id}_grounded_fallback_{index}",
                task_id=task.id,
                title=result.title,
                content=_short_result_snippet(result),
                source_url=result.url,
                confidence=0.5,
                metadata={
                    "task_name": task.name,
                    "grounded_fallback": True,
                    "source_metadata": result.metadata,
                    "source_type": result.metadata.get("source_type"),
                },
            )
            for index, result in enumerate(results[:3], start=1)
        ]


def _result_for_evidence(evidence: Evidence, results: list[SearchResult], index: int) -> SearchResult | None:
    for result in results:
        if result.url == evidence.source_url:
            return result
    if not results:
        return None
    return results[(index - 1) % len(results)]


def _confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.75
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def _grounded_confidence(evidence: Evidence, support_score: float) -> float:
    confidence = max(0.0, min(1.0, (evidence.confidence * 0.55) + (support_score * 0.45)))
    if evidence.metadata.get("grounded_fallback"):
        confidence = min(confidence * 0.72, 0.62)
    if evidence.metadata.get("degraded") or str(evidence.source_url or "").startswith("mock://degraded"):
        confidence = min(confidence * 0.5, 0.35)
    if evidence.metadata.get("citation_validation_status") in {"unsupported", "url_unreachable", "no_source"}:
        confidence = min(confidence, 0.45)
    return round(max(0.0, min(1.0, confidence)), 6)


def _has_overlap(content: str, document: FetchedDocument | None) -> bool:
    if document is None:
        return False
    left = {token for token in content.lower().split() if len(token) > 3}
    right = {token for token in f"{document.title or ''} {document.text or ''} {document.snippet or ''}".lower().split() if len(token) > 3}
    return bool(left & right)


def _short_result_snippet(result: SearchResult, max_chars: int = 420) -> str:
    text = " ".join((result.snippet or result.title or "").split())
    if " Authors: " in text:
        text = text.split(" Authors: ", 1)[0].strip()
    if len(text) <= max_chars:
        return text
    cutoff = text.rfind(". ", 0, max_chars)
    if cutoff < 120:
        cutoff = max_chars
    return text[:cutoff].rstrip(" .,") + "."


def _api_base_host(provider: BaseSearchProvider) -> str | None:
    from urllib.parse import urlparse

    api_base = getattr(provider, "api_base", None)
    if not api_base:
        return None
    return urlparse(str(api_base)).netloc or None
