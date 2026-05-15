import httpx
import pytest

from deepresearch_agent.search.arxiv_provider import ArxivSearchProvider, PaperSearchProvider
from deepresearch_agent.search.errors import SearchProviderError, SearchRateLimitError


def setup_function() -> None:
    ArxivSearchProvider._cache.clear()
    ArxivSearchProvider._last_request_at = 0.0


def arxiv_provider() -> ArxivSearchProvider:
    return ArxivSearchProvider(min_request_interval_seconds=0, retry_backoff_seconds=0)


def paper_provider() -> PaperSearchProvider:
    return PaperSearchProvider(min_request_interval_seconds=0, retry_backoff_seconds=0)


ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2307.03172v1</id>
    <updated>2023-07-06T00:00:00Z</updated>
    <published>2023-07-06T00:00:00Z</published>
    <title>Lost in the Middle: How Language Models Use Long Contexts</title>
    <summary>We analyze how language models use relevant information in long input contexts.</summary>
    <author><name>Nelson F. Liu</name></author>
    <author><name>Percy Liang</name></author>
    <arxiv:primary_category term="cs.CL" />
    <category term="cs.CL" />
    <link href="http://arxiv.org/abs/2307.03172v1" rel="alternate" type="text/html" />
    <link title="pdf" href="http://arxiv.org/pdf/2307.03172v1" rel="related" type="application/pdf" />
    <arxiv:doi>10.0000/example</arxiv:doi>
  </entry>
</feed>
"""

MIXED_ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2501.00001v1</id>
    <updated>2025-01-01T00:00:00Z</updated>
    <published>2025-01-01T00:00:00Z</published>
    <title>Learning Long-Context Diffusion Policies for Robot Control</title>
    <summary>This paper studies robotic control trajectories and policy learning.</summary>
    <author><name>Robot Author</name></author>
    <arxiv:primary_category term="cs.RO" />
    <category term="cs.RO" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2307.03172v1</id>
    <updated>2023-07-06T00:00:00Z</updated>
    <published>2023-07-06T00:00:00Z</published>
    <title>Lost in the Middle: How Language Models Use Long Contexts</title>
    <summary>We analyze how language models use relevant information in long input contexts.</summary>
    <author><name>Nelson F. Liu</name></author>
    <arxiv:primary_category term="cs.CL" />
    <category term="cs.CL" />
  </entry>
</feed>
"""

SEMANTIC_COMMUNICATION_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2502.17129v2</id>
    <updated>2025-11-11T00:00:00Z</updated>
    <published>2025-02-24T00:00:00Z</published>
    <title>Thus Spake Long-Context Large Language Model</title>
    <summary>Long context is a crucial topic in Natural Language Processing and Large Language Models, with evaluation, architecture, infrastructure, and training challenges.</summary>
    <author><name>Long Context Author</name></author>
    <arxiv:primary_category term="cs.CL" />
    <category term="cs.CL" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2503.00001v1</id>
    <updated>2025-03-01T00:00:00Z</updated>
    <published>2025-03-01T00:00:00Z</published>
    <title>Task-Oriented Semantic Communication with Large Language Models</title>
    <summary>This paper studies semantic communication and semantic transmission for wireless task-oriented communication systems using large language models.</summary>
    <author><name>Semantic Author</name></author>
    <arxiv:primary_category term="cs.IT" />
    <category term="cs.IT" />
    <category term="cs.AI" />
  </entry>
</feed>
"""


class Response:
    def __init__(self, text: str = ARXIV_FEED, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("GET", "https://export.arxiv.org/api/query"),
                response=httpx.Response(self.status_code),
            )


def test_arxiv_provider_parses_paper_metadata(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response())

    results = arxiv_provider().search("long context llm evaluation", top_k=2)

    assert len(results) == 1
    assert results[0].provider == "arxiv"
    assert results[0].title.startswith("Lost in the Middle")
    assert results[0].domain == "arxiv.org"
    assert results[0].metadata["paper_id"] == "2307.03172v1"
    assert results[0].metadata["pdf_url"] == "http://arxiv.org/pdf/2307.03172v1"
    assert results[0].metadata["authors"] == ["Nelson F. Liu", "Percy Liang"]
    assert "long input contexts" in results[0].snippet
    assert results[0].metadata["reranked_by_relevance"] is True
    assert results[0].metadata["retrieval_relevance_score"] > 0
    assert "title_overlap" in results[0].metadata["retrieval_relevance_reason"]


def test_arxiv_provider_expands_long_context_queries(monkeypatch) -> None:
    captured = {}

    def fake_get(*args, **kwargs):
        captured["params"] = kwargs["params"]
        return Response()

    monkeypatch.setattr(httpx, "get", fake_get)

    arxiv_provider().search("limitations of benchmarks for evaluating long-context LLMs", top_k=2)

    search_query = captured["params"]["search_query"]
    assert 'all:"LongBench"' in search_query
    assert 'all:"lost in the middle"' in search_query
    assert " OR " in search_query


def test_arxiv_provider_does_not_treat_plain_llm_query_as_long_context(monkeypatch) -> None:
    captured = {}

    def fake_get(*args, **kwargs):
        captured["params"] = kwargs["params"]
        return Response()

    monkeypatch.setattr(httpx, "get", fake_get)

    arxiv_provider().search("task-oriented semantic communication with LLMs", top_k=1)

    search_query = captured["params"]["search_query"]
    assert 'all:"LongBench"' not in search_query
    assert 'all:"lost in the middle"' not in search_query
    assert 'all:"semantic communication"' in search_query
    assert 'all:"task-oriented semantic communication"' in search_query


def test_arxiv_provider_filters_off_topic_long_context_papers(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response(text=MIXED_ARXIV_FEED))

    results = arxiv_provider().search("long-context LLM evaluation", top_k=2)

    assert [result.title for result in results] == ["Lost in the Middle: How Language Models Use Long Contexts"]


def test_arxiv_provider_requires_semantic_signal_for_semantic_communication(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response(text=SEMANTIC_COMMUNICATION_FEED))

    results = arxiv_provider().search("recent methods for task-oriented semantic communication with LLMs", top_k=2)

    assert [result.title for result in results] == [
        "Task-Oriented Semantic Communication with Large Language Models"
    ]
    assert results[0].metadata["retrieval_relevance_score"] > 0.5
    assert "semantic_signal=True" in results[0].metadata["retrieval_relevance_reason"]


def test_paper_provider_uses_paper_provider_name(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response())

    results = paper_provider().search("long context llm evaluation", top_k=1)

    assert results[0].provider == "paper"


@pytest.mark.parametrize("status_code", [401, 500])
def test_arxiv_http_errors_are_search_provider_errors(monkeypatch, status_code: int) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response(status_code=status_code))

    with pytest.raises(SearchProviderError) as exc:
        arxiv_provider().search("q")

    assert str(status_code) in str(exc.value)


def test_arxiv_rate_limit_error(monkeypatch) -> None:
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: Response(status_code=429))

    with pytest.raises(SearchRateLimitError):
        arxiv_provider().search("q")


def test_arxiv_retries_rate_limit_then_succeeds(monkeypatch) -> None:
    calls = []
    responses = [Response(status_code=429), Response()]

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(httpx, "get", fake_get)

    results = arxiv_provider().search("long context llm evaluation", top_k=1)

    assert len(calls) == 2
    assert results


def test_arxiv_uses_successful_cache(monkeypatch) -> None:
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(httpx, "get", fake_get)
    provider = arxiv_provider()

    first = provider.search("long context llm evaluation", top_k=1)
    second = provider.search("long context llm evaluation", top_k=1)

    assert len(calls) == 1
    assert first[0].id == second[0].id
