from __future__ import annotations

import re
from urllib.parse import urlparse

from deepresearch_agent.search.provider import BaseSearchProvider
from deepresearch_agent.search.schemas import SearchResult


class MockSearchProvider(BaseSearchProvider):
    provider_name = "mock"

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        normalized = query.lower()
        records = _records_for_query(normalized)
        results: list[SearchResult] = []
        for rank, record in enumerate(records[:top_k], start=1):
            url = record["url"]
            results.append(
                SearchResult(
                    id=f"mock_{_slug(query)}_{rank}",
                    query_id=_slug(query),
                    title=record["title"],
                    url=url,
                    snippet=record["snippet"],
                    rank=rank,
                    provider=self.provider_name,
                    domain=urlparse(url).netloc or None,
                )
            )
        return results


def _records_for_query(query: str) -> list[dict[str, str]]:
    if "long" in query and ("context" in query or "llm" in query):
        return [
            {
                "title": "LongBench long context LLM evaluation benchmark",
                "url": "https://research.example.org/longbench-long-context-evaluation",
                "snippet": "LongBench evaluates long-context LLMs on context retention, retrieval, reasoning, coherence, and summarization tasks.",
            },
            {
                "title": "Needle retrieval and position bias in long context models",
                "url": "https://research.example.org/long-context-position-bias",
                "snippet": "Needle-style tests and position bias protocols measure whether relevant evidence can be found across long prompts with distractors.",
            },
        ]
    if "semantic communication" in query:
        return [
            {
                "title": "Task-oriented semantic communication survey",
                "url": "https://research.example.org/semantic-communication-survey",
                "snippet": "Semantic communication studies semantic encoding, channel noise, task relevance, and reconstruction fidelity.",
            },
            {
                "title": "Knowledge assisted semantic communication",
                "url": "https://research.example.org/knowledge-semantic-communication",
                "snippet": "Shared knowledge can improve compression, disambiguation, and robustness in semantic communication systems.",
            },
        ]
    if "rag" in query or "retrieval" in query:
        return [
            {
                "title": "Retrieval augmented generation failure modes",
                "url": "https://research.example.org/rag-failure-modes",
                "snippet": "RAG systems fail through retrieval misses, irrelevant context, hallucination, and citation mismatch.",
            },
            {
                "title": "Chunking and reranking for grounded RAG",
                "url": "https://research.example.org/rag-chunking-reranking",
                "snippet": "Chunking and reranking influence evidence relevance, answer grounding, and citation quality in RAG pipelines.",
            },
        ]
    if "agent" in query:
        return [
            {
                "title": "Reliable LLM agent workflow design",
                "url": "https://research.example.org/llm-agent-workflows",
                "snippet": "Reliable LLM agents use planning, tool control, shared memory, trace logging, retries, and evaluation.",
            }
        ]
    return [
        {
            "title": "General deep research system evaluation",
            "url": "https://research.example.org/deep-research-evaluation",
            "snippet": "Deep research systems combine planning, evidence collection, citation support, report writing, and evaluation metrics.",
        }
    ]


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug[:60] or "query"
