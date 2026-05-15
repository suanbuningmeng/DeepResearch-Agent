from deepresearch_agent.compression import L2TextRankFilter
from deepresearch_agent.schemas import CompressedEvidence, Evidence


def compressed(evidence_id: str, content: str) -> CompressedEvidence:
    return CompressedEvidence(
        evidence=Evidence(
            id=evidence_id,
            task_id="task",
            title=evidence_id,
            content=content,
            confidence=0.8,
        ),
        l1_similarity_score=0.5,
    )


def test_l2_textrank_single_evidence_score_is_one() -> None:
    item = compressed("e1", "long context evaluation")

    selected = L2TextRankFilter(top_k=1).filter([item])

    assert selected[0].l2_textrank_score == 1.0


def test_l2_textrank_outputs_top_k() -> None:
    items = [
        compressed("e1", "long context retrieval benchmark"),
        compressed("e2", "long context synthesis benchmark"),
        compressed("e3", "unrelated cooking recipe"),
    ]

    selected = L2TextRankFilter(top_k=2, similarity_threshold=0.0).filter(items)

    assert len(selected) == 2
    assert all(item.l2_textrank_score >= 0 for item in selected)


def test_l2_textrank_no_edges_gives_equal_scores() -> None:
    items = [
        compressed("e1", "alpha"),
        compressed("e2", "beta"),
    ]

    selected = L2TextRankFilter(top_k=2, similarity_threshold=1.1).filter(items)

    assert selected[0].l2_textrank_score == selected[1].l2_textrank_score
