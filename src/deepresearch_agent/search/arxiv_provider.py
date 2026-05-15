from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET

import httpx

from deepresearch_agent.search.provider import BaseSearchProvider, domain_from_url, handle_http_error
from deepresearch_agent.search.schemas import SearchResult


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "assess",
    "challenges",
    "for",
    "from",
    "how",
    "in",
    "large",
    "language",
    "limitations",
    "main",
    "methods",
    "model",
    "models",
    "of",
    "on",
    "or",
    "problems",
    "recent",
    "requirements",
    "the",
    "to",
    "what",
    "with",
}
RESEARCH_DOMAIN_KEYWORDS = {
    "ai",
    "artificial intelligence",
    "llm",
    "llms",
    "large language model",
    "large language models",
    "language model",
    "language models",
    "transformer",
    "benchmark",
    "benchmarks",
    "evaluation",
    "evaluating",
    "metric",
    "metrics",
    "reasoning",
    "retrieval",
    "faithfulness",
    "summarization",
    "longbench",
    "l-eval",
    "infinitebench",
    "lost in the middle",
    "needle in a haystack",
    "semantic communication",
    "semantic communications",
    "task-oriented communication",
    "task oriented communication",
    "wireless",
    "communication",
    "communications",
}
OFF_TOPIC_KEYWORDS = {
    "robot",
    "robotic",
    "robotics",
    "policy learning",
    "diffusion policy",
    "trajectory",
    "trajectories",
    "control",
    "manipulation",
    "navigation",
    "vision-language navigation",
}
PREFERRED_AI_CATEGORIES = {"cs.CL", "cs.AI", "cs.LG", "cs.IR", "cs.NE"}
OFF_TOPIC_CATEGORIES = ("cs.RO", "cs.CV")
MIN_RELEVANCE_SCORE = 0.24
LONG_CONTEXT_ANCHORS = [
    'all:"long context"',
    'all:"long-context"',
    'all:"LongBench"',
    'all:"L-Eval"',
    'all:"InfiniteBench"',
    'all:"lost in the middle"',
    'all:"needle in a haystack"',
    'all:"large language model"',
]
SEMANTIC_COMMUNICATION_ANCHORS = [
    'all:"semantic communication"',
    'all:"semantic communications"',
    'all:"task-oriented semantic communication"',
    'all:"task oriented semantic communication"',
    'all:"semantic-aware communication"',
    'all:"semantic transmission"',
    '(all:"large language model" AND all:"semantic communication")',
    '(all:"LLM" AND all:"semantic communication")',
]


class ArxivSearchProvider(BaseSearchProvider):
    """Search arXiv papers and map results into the shared SearchResult schema."""

    provider_name = "arxiv"
    _cache: dict[tuple[str, int, str, str], list[SearchResult]] = {}
    _last_request_at: float = 0.0

    def __init__(
        self,
        api_base: str = "https://export.arxiv.org/api/query",
        timeout: int = 20,
        sort_by: str = "relevance",
        sort_order: str = "descending",
        max_retries: int = 2,
        retry_backoff_seconds: float = 3.0,
        min_request_interval_seconds: float = 3.0,
    ) -> None:
        self.api_base = api_base
        self.timeout = timeout
        self.sort_by = sort_by
        self.sort_order = sort_order
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.min_request_interval_seconds = min_request_interval_seconds

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        search_query = _build_arxiv_query(query)
        cache_key = (search_query, top_k, self.sort_by, self.sort_order)
        if cache_key in self._cache:
            return _copy_results(self._cache[cache_key])

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._respect_rate_limit()
                response = httpx.get(
                    self.api_base,
                    params={
                        "search_query": search_query,
                        "start": 0,
                        "max_results": max(1, top_k * 5),
                        "sortBy": self.sort_by,
                        "sortOrder": self.sort_order,
                    },
                    timeout=self.timeout,
                )
                response.raise_for_status()
                results = _parse_arxiv_feed(response.text, query=query, top_k=top_k, provider_name=self.provider_name)
                self._cache[cache_key] = _copy_results(results)
                return results
            except Exception as exc:
                last_error = exc
                if not _is_retryable_arxiv_error(exc) or attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * (attempt + 1))
        raise handle_http_error(self.provider_name, self.api_base, last_error or RuntimeError("unknown arxiv search failure"))

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self.__class__._last_request_at
        wait_seconds = self.min_request_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        self.__class__._last_request_at = time.monotonic()


class PaperSearchProvider(ArxivSearchProvider):
    """Semantic alias for paper-focused search; currently backed by arXiv."""

    provider_name = "paper"


def _build_arxiv_query(query: str) -> str:
    cleaned = " ".join(str(query or "").split())
    if not cleaned:
        return 'all:"large language model" OR all:"long context"'
    if re.search(r"\b(?:all|ti|au|abs|cat):", cleaned, flags=re.IGNORECASE):
        return cleaned
    terms = _salient_terms(cleaned)
    clauses: list[str] = []
    if _looks_like_semantic_communication_query(cleaned):
        clauses.extend(SEMANTIC_COMMUNICATION_ANCHORS)
    elif _looks_like_long_context_query(cleaned):
        clauses.extend(LONG_CONTEXT_ANCHORS)
    clauses.extend(f"all:{term}" for term in terms[:8])
    if not clauses:
        clauses.append(f'all:"{_escape_phrase(cleaned[:80])}"')
    return " OR ".join(_dedupe(clauses))


def _looks_like_long_context_query(query: str) -> bool:
    normalized = query.lower()
    return any(
        marker in normalized
        for marker in (
            "long-context",
            "long context",
            "lost-in-the-middle",
            "lost in the middle",
            "context retention",
            "longbench",
            "l-eval",
            "infinitebench",
            "needle in a haystack",
        )
    )


def _salient_terms(query: str) -> list[str]:
    normalized = query.replace("-", " ")
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9]+", normalized)
    terms: list[str] = []
    for token in tokens:
        lower = token.lower()
        if lower in STOPWORDS or len(lower) < 4:
            continue
        terms.append(_escape_term(token))
    return _dedupe(terms)


def _escape_term(term: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "", term)


def _escape_phrase(phrase: str) -> str:
    return phrase.replace('"', "").strip()


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _parse_arxiv_feed(feed_text: str, query: str, top_k: int, provider_name: str) -> list[SearchResult]:
    root = ET.fromstring(feed_text)
    results: list[SearchResult] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        url = _text(entry, "atom:id") or ""
        result_rank = len(results) + 1
        title = _clean_text(_text(entry, "atom:title")) or url or f"arXiv result {result_rank}"
        summary = _clean_text(_text(entry, "atom:summary"))
        published_date = _text(entry, "atom:published")
        updated_date = _text(entry, "atom:updated")
        paper_id = _paper_id_from_url(url) or f"result_{result_rank}"
        authors = [
            _clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        authors = [author for author in authors if author]
        pdf_url = _pdf_url(entry)
        primary_category = _primary_category(entry)
        categories = _categories(entry)
        doi = _text(entry, "arxiv:doi")
        snippet = _paper_snippet(
            summary=summary,
            authors=authors,
            published_date=published_date,
            primary_category=primary_category,
            doi=doi,
        )
        result = SearchResult(
            id=f"arxiv_{_safe_id(paper_id)}",
            query_id=query[:80] or "arxiv_query",
            title=title,
            url=url,
            snippet=snippet,
            rank=len(results) + 1,
            provider=provider_name,
            domain=domain_from_url(url),
            metadata={
                "paper_id": paper_id,
                "authors": authors,
                "published_date": published_date,
                "updated_date": updated_date,
                "pdf_url": pdf_url,
                "primary_category": primary_category,
                "categories": categories,
                "doi": doi,
                "source_type": "paper",
            },
        )
        relevance_score, relevance_reason = _paper_relevance_score(result, query)
        result.metadata["retrieval_relevance_score"] = relevance_score
        result.metadata["retrieval_relevance_reason"] = relevance_reason
        result.metadata["reranked_by_relevance"] = True
        if _is_relevant_paper(result, query):
            results.append(result)
        if len(results) >= top_k:
            break
    return sorted(results, key=lambda item: float(item.metadata.get("retrieval_relevance_score") or 0.0), reverse=True)


def _is_relevant_paper(result: SearchResult, query: str) -> bool:
    relevance_score = float(result.metadata.get("retrieval_relevance_score") or 0.0)
    if not _looks_like_research_query(query):
        return relevance_score >= 0.12
    text = f"{result.title} {result.snippet}".lower()
    categories = set(result.metadata.get("categories") or [])
    primary_category = str(result.metadata.get("primary_category") or "")
    preferred_category = bool(categories & PREFERRED_AI_CATEGORIES) or primary_category in PREFERRED_AI_CATEGORIES
    off_topic_category = primary_category.startswith(OFF_TOPIC_CATEGORIES)
    has_language_model = any(term in text for term in ("llm", "language model", "large language model", "transformer"))
    has_long_context = any(term in text for term in ("long context", "long-context", "long input", "long sequence", "context window"))
    has_evaluation = any(term in text for term in ("evaluation", "benchmark", "metric", "reasoning", "retrieval", "faithfulness", "needle", "lost in the middle"))
    if relevance_score < MIN_RELEVANCE_SCORE:
        return False
    if _looks_like_semantic_communication_query(query):
        return _has_semantic_communication_signal(text) and not off_topic_category
    if preferred_category and (has_long_context or has_language_model) and has_evaluation:
        return True
    if off_topic_category and not (has_language_model and has_evaluation and relevance_score >= 0.5):
        return False
    return has_language_model and has_long_context


def _looks_like_research_query(query: str) -> bool:
    normalized = query.lower()
    return _looks_like_long_context_query(query) or _looks_like_semantic_communication_query(query) or any(
        term in normalized for term in ("llm", "large language model", "artificial intelligence", "machine learning", "rag", "agent")
    )


def _looks_like_semantic_communication_query(query: str) -> bool:
    normalized = query.lower()
    return (
        "semantic communication" in normalized
        or "semantic communications" in normalized
        or "task-oriented communication" in normalized
        or "task oriented communication" in normalized
        or "semantic transmission" in normalized
        or ("communication" in normalized and ("llm" in normalized or "large language model" in normalized))
    )


def _has_semantic_communication_signal(text: str) -> bool:
    return (
        "semantic communication" in text
        or "semantic communications" in text
        or "task-oriented semantic" in text
        or "task oriented semantic" in text
        or ("task-oriented" in text and "communication" in text)
        or ("task oriented" in text and "communication" in text)
        or "semantic-aware communication" in text
        or "semantic aware communication" in text
        or "semantic transmission" in text
        or "wireless semantic" in text
        or "semantic coding" in text
    )


def _paper_relevance_score(result: SearchResult, query: str) -> tuple[float, str]:
    query_terms = set(_salient_terms(query.lower()))
    title_terms = set(_salient_terms(result.title.lower()))
    summary_terms = set(_salient_terms(result.snippet.lower()))
    title_overlap = _overlap(query_terms, title_terms)
    summary_overlap = _overlap(query_terms, summary_terms)
    text = f"{result.title} {result.snippet}".lower()
    domain_hits = sum(1 for keyword in RESEARCH_DOMAIN_KEYWORDS if keyword in text)
    domain_score = min(1.0, domain_hits / 5.0)
    primary_category = str(result.metadata.get("primary_category") or "")
    categories = set(result.metadata.get("categories") or [])
    category_bonus = 0.12 if (primary_category in PREFERRED_AI_CATEGORIES or bool(categories & PREFERRED_AI_CATEGORIES)) else 0.0
    off_topic_hits = [keyword for keyword in OFF_TOPIC_KEYWORDS if keyword in text]
    off_topic_penalty = 0.3 if off_topic_hits else 0.0
    if primary_category.startswith(OFF_TOPIC_CATEGORIES) and off_topic_hits:
        off_topic_penalty += 0.2
    semantic_signal = _has_semantic_communication_signal(text)
    semantic_bonus = 0.22 if _looks_like_semantic_communication_query(query) and semantic_signal else 0.0
    semantic_penalty = 0.35 if _looks_like_semantic_communication_query(query) and not semantic_signal else 0.0
    score = (
        (0.35 * title_overlap)
        + (0.25 * summary_overlap)
        + (0.28 * domain_score)
        + category_bonus
        + semantic_bonus
        - semantic_penalty
        - off_topic_penalty
    )
    score = round(max(0.0, min(1.0, score)), 6)
    reason = (
        f"title_overlap={title_overlap:.2f}; summary_overlap={summary_overlap:.2f}; "
        f"domain_hits={domain_hits}; category={primary_category or 'unknown'}; "
        f"semantic_signal={semantic_signal}; off_topic_hits={','.join(off_topic_hits[:3]) or 'none'}"
    )
    return score, reason


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


def _text(entry: ET.Element, path: str) -> str | None:
    value = entry.findtext(path, default=None, namespaces=ATOM_NS)
    return value.strip() if isinstance(value, str) else None


def _clean_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _paper_id_from_url(url: str) -> str | None:
    match = re.search(r"/abs/([^/?#]+)", url)
    return match.group(1) if match else None


def _pdf_url(entry: ET.Element) -> str | None:
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
            return link.attrib.get("href")
    return None


def _primary_category(entry: ET.Element) -> str | None:
    primary = entry.find("arxiv:primary_category", ATOM_NS)
    if primary is not None:
        return primary.attrib.get("term")
    return None


def _categories(entry: ET.Element) -> list[str]:
    return [category.attrib["term"] for category in entry.findall("atom:category", ATOM_NS) if category.attrib.get("term")]


def _paper_snippet(
    summary: str,
    authors: list[str],
    published_date: str | None,
    primary_category: str | None,
    doi: str | None,
) -> str:
    metadata_parts = []
    if authors:
        metadata_parts.append("Authors: " + ", ".join(authors[:5]))
    if published_date:
        metadata_parts.append(f"Published: {published_date[:10]}")
    if primary_category:
        metadata_parts.append(f"Category: {primary_category}")
    if doi:
        metadata_parts.append(f"DOI: {doi}")
    metadata = " | ".join(metadata_parts)
    return f"{summary} {metadata}".strip()[:1600]


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "paper"


def _is_retryable_arxiv_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {429, 500, 502, 503, 504}
    return isinstance(exc, (httpx.TimeoutException, httpx.TransportError))


def _copy_results(results: list[SearchResult]) -> list[SearchResult]:
    return [result.model_copy(deep=True) for result in results]
