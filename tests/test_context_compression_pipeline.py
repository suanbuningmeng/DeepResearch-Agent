from deepresearch_agent.compression import ContextCompressionPipeline
from deepresearch_agent.memory import HashingEmbeddingProvider
from deepresearch_agent.schemas import Evidence


def evidence(evidence_id: str, content: str) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id="task",
        title=evidence_id,
        content=content,
        source_url="mock://test",
        confidence=0.8,
    )


def test_context_compression_disabled_returns_original_evidence() -> None:
    evidences = [evidence("e1", "alpha"), evidence("e2", "beta")]
    pipeline = ContextCompressionPipeline(HashingEmbeddingProvider(dim=64), enabled=False)

    selected, stats = pipeline.compress("alpha", evidences)

    assert selected == evidences
    assert stats.enabled is False


def test_context_compression_enabled_reduces_evidence_count() -> None:
    evidences = [
        evidence("e1", "long context retrieval benchmark"),
        evidence("e2", "long context synthesis benchmark"),
        evidence("e3", "position bias evaluation"),
        evidence("e4", "cooking recipe"),
    ]
    pipeline = ContextCompressionPipeline(
        HashingEmbeddingProvider(dim=64),
        l1_top_n=3,
        l2_top_k=2,
        enabled=True,
    )

    selected, stats = pipeline.compress("long context benchmark", evidences)

    assert len(selected) <= 2
    assert stats.enabled is True
    assert stats.selected_evidence_ids
    assert stats.dropped_evidence_ids
    assert 0 < stats.compression_ratio <= 1
