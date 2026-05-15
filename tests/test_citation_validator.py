from deepresearch_agent.schemas import Evidence
from deepresearch_agent.search.citation_validator import CitationValidator
from deepresearch_agent.search.schemas import FetchedDocument


def evidence(source_url: str | None = "https://example.org/a", content: str = "Long context evaluation tests retrieval and coherence.") -> Evidence:
    return Evidence(id="e1", task_id="task_1", title="Long context evaluation", content=content, source_url=source_url, confidence=0.8)


def test_no_source_status() -> None:
    result = CitationValidator().validate(evidence(None), None)

    assert result.validation_status == "no_source"


def test_fetch_failed_status() -> None:
    doc = FetchedDocument(url="https://example.org/a", fetch_success=False, error="fail")

    result = CitationValidator().validate(evidence(), doc)

    assert result.validation_status == "url_unreachable"


def test_supported_overlap() -> None:
    doc = FetchedDocument(
        url="https://example.org/a",
        title="Long context evaluation benchmark",
        text="Long context evaluation tests retrieval and coherence across long documents.",
        fetch_success=True,
    )

    result = CitationValidator().validate(evidence(), doc)

    assert result.validation_status == "supported"


def test_partially_supported_overlap() -> None:
    doc = FetchedDocument(
        url="https://example.org/a",
        title="Long context benchmark",
        text="Long context benchmark includes retrieval tasks but does not discuss coherence.",
        fetch_success=True,
    )

    result = CitationValidator().validate(evidence(), doc)

    assert result.validation_status == "partially_supported"


def test_unsupported_overlap() -> None:
    doc = FetchedDocument(url="https://example.org/a", title="Database indexes", text="Indexes improve database reads.", fetch_success=True)

    result = CitationValidator().validate(evidence(), doc)

    assert result.validation_status == "unsupported"


def test_degraded_source_is_not_high_quality_citation() -> None:
    result = CitationValidator().validate(evidence("mock://degraded/task_1"), None)

    assert result.validation_status == "unsupported"
    assert result.support_score < 0.1
    assert "Degraded fallback" in result.reason


def test_weak_overlap_is_more_conservative() -> None:
    doc = FetchedDocument(
        url="https://example.org/a",
        title="Long context benchmark",
        text="The benchmark has retrieval tasks.",
        fetch_success=True,
    )

    result = CitationValidator().validate(
        evidence(content="Long context evaluation proves robust coherence, faithful reasoning, scalable summarization, and reliable citation quality."),
        doc,
    )

    assert result.validation_status in {"partially_supported", "unsupported"}
    assert result.validation_status != "supported"
