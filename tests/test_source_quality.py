from deepresearch_agent.memory.source_quality import SourceQuality, classify_source_url


def test_source_quality_classification() -> None:
    assert classify_source_url(None) == SourceQuality.NULL_SOURCE
    assert classify_source_url("https://example.com/page") == SourceQuality.EXAMPLE_URL
    assert classify_source_url("mock://long-context") == SourceQuality.MOCK_SOURCE
    assert classify_source_url("model://unstructured-output") == SourceQuality.MODEL_GENERATED
    assert classify_source_url("https://arxiv.org/abs/1234") == SourceQuality.REAL_URL
