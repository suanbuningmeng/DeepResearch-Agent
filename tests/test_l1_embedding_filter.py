from deepresearch_agent.compression import L1EmbeddingFilter
from deepresearch_agent.memory import HashingEmbeddingProvider
from deepresearch_agent.schemas import Evidence


def make_evidence(evidence_id: str, title: str, content: str) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id="task_1",
        title=title,
        content=content,
        source_url="mock://test",
        confidence=0.8,
    )


def test_l1_embedding_filter_selects_relevant_evidence() -> None:
    evidences = [
        make_evidence("relevant", "retrieval benchmark", "long context retrieval evaluation benchmark"),
        make_evidence("irrelevant", "recipe", "cooking pasta with tomato sauce"),
    ]
    filter_stage = L1EmbeddingFilter(HashingEmbeddingProvider(dim=64), top_n=1)

    selected = filter_stage.filter("long context retrieval benchmark", evidences)

    assert selected[0].evidence.id == "relevant"
    assert selected[0].l1_similarity_score > 0


def test_l1_embedding_filter_keeps_all_when_less_than_top_n() -> None:
    evidences = [
        make_evidence("a", "A", "alpha"),
        make_evidence("b", "B", "beta"),
    ]
    filter_stage = L1EmbeddingFilter(HashingEmbeddingProvider(dim=64), top_n=20)

    selected = filter_stage.filter("alpha", evidences)

    assert len(selected) == 2
    assert all(item.l1_similarity_score >= 0 for item in selected)
