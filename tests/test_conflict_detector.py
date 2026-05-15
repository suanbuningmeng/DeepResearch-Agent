from deepresearch_agent.conflict.detector import EvidenceConflictDetector
from deepresearch_agent.memory import HashingEmbeddingProvider
from deepresearch_agent.schemas import ConflictType, Evidence


def evidence(evidence_id: str, title: str, content: str, source_url: str | None = "mock://source") -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id="task_1",
        title=title,
        content=content,
        source_url=source_url,
        confidence=0.8,
    )


def detector() -> EvidenceConflictDetector:
    return EvidenceConflictDetector(
        embedding_provider=HashingEmbeddingProvider(),
        near_duplicate_threshold=0.90,
        semantic_similarity_threshold=0.55,
    )


def test_detector_returns_empty_for_less_than_two_evidences() -> None:
    assert detector().detect([evidence("e1", "A", "B")]) == []


def test_detector_identifies_exact_duplicate() -> None:
    conflicts = detector().detect(
        [
            evidence("e1", "Same claim", "Accuracy improves."),
            evidence("e2", "Same claim", "Accuracy improves."),
        ]
    )

    assert conflicts[0].conflict_type == ConflictType.DUPLICATE


def test_detector_identifies_near_duplicate() -> None:
    conflicts = detector().detect(
        [
            evidence("e1", "Accuracy improvement", "Method A improves accuracy and robustness."),
            evidence("e2", "Accuracy improvement", "Method A improves accuracy and robustness slightly."),
        ]
    )

    assert any(conflict.conflict_type == ConflictType.NEAR_DUPLICATE for conflict in conflicts)


def test_detector_identifies_antonym_contradiction() -> None:
    conflicts = detector().detect(
        [
            evidence("e1", "Method A accuracy", "Method A improves accuracy for long-context tasks."),
            evidence("e2", "Method A accuracy", "Method A degrades accuracy for long-context tasks."),
        ]
    )

    assert any(conflict.conflict_type == ConflictType.ANTONYM_CONTRADICTION for conflict in conflicts)


def test_detector_identifies_semantic_opposition() -> None:
    conflicts = detector().detect(
        [
            evidence("e1", "Method A reliability", "Method A is reliable for long-context synthesis."),
            evidence("e2", "Method A reliability", "Method A is not reliable for long-context synthesis."),
        ]
    )

    assert any(conflict.conflict_type == ConflictType.SEMANTIC_OPPOSITION for conflict in conflicts)
