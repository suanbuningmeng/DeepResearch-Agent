from deepresearch_agent.conflict.resolver import EvidenceConflictResolver
from deepresearch_agent.schemas import (
    ConflictSeverity,
    ConflictType,
    Evidence,
    EvidenceConflict,
    ResolutionAction,
)


def evidence(
    evidence_id: str,
    source_url: str | None,
    confidence: float = 0.8,
) -> Evidence:
    return Evidence(
        id=evidence_id,
        task_id="task_1",
        title=f"Evidence {evidence_id}",
        content="Method A improves accuracy.",
        source_url=source_url,
        confidence=confidence,
    )


def conflict(
    conflict_type: ConflictType,
    left_score: float,
    right_score: float,
) -> EvidenceConflict:
    return EvidenceConflict(
        id="conflict_e1_e2_1",
        left_evidence_id="e1",
        right_evidence_id="e2",
        conflict_type=conflict_type,
        severity=ConflictSeverity.HIGH,
        reason="test conflict",
        left_score=left_score,
        right_score=right_score,
    )


def test_duplicate_keeps_higher_quality_evidence() -> None:
    left = evidence("e1", "mock://source", confidence=0.6)
    right = evidence("e2", "https://example.org/source", confidence=0.95)
    item = conflict(ConflictType.DUPLICATE, left_score=0.4, right_score=0.9)

    resolved, stats = EvidenceConflictResolver().resolve([left, right], [item])

    assert [item.id for item in resolved] == ["e2"]
    assert stats.dropped_evidence_ids == ["e1"]
    assert stats.conflicts[0].resolution_action == ResolutionAction.DROP_LEFT


def test_example_source_conflict_keeps_real_url() -> None:
    left = evidence("e1", "https://example.com/mock", confidence=0.95)
    right = evidence("e2", "https://real.example.org/source", confidence=0.85)
    item = conflict(ConflictType.ANTONYM_CONTRADICTION, left_score=0.9, right_score=0.8)

    resolved, stats = EvidenceConflictResolver().resolve([left, right], [item])

    assert [item.id for item in resolved] == ["e2"]
    assert stats.dropped_evidence_ids == ["e1"]
    assert stats.conflicts[0].resolution_action == ResolutionAction.DROP_LEFT


def test_close_quality_conflict_marks_for_writer() -> None:
    left = evidence("e1", "mock://left", confidence=0.8)
    right = evidence("e2", "mock://right", confidence=0.79)
    item = conflict(ConflictType.SEMANTIC_OPPOSITION, left_score=0.66, right_score=0.64)

    resolved, stats = EvidenceConflictResolver().resolve([left, right], [item])

    assert len(resolved) == 2
    assert all(item.metadata["conflict_marked"] for item in resolved)
    assert stats.marked_conflict_evidence_ids == ["e1", "e2"]
    assert stats.conflicts[0].resolution_action == ResolutionAction.MARK_FOR_WRITER


def test_downweighted_evidence_metadata_is_written() -> None:
    left = evidence("e1", "mock://left", confidence=0.9)
    right = evidence("e2", "mock://right", confidence=0.5)
    item = conflict(ConflictType.SOURCE_QUALITY_CONFLICT, left_score=0.7, right_score=0.55)

    resolved, stats = EvidenceConflictResolver().resolve([left, right], [item])
    right_result = next(item for item in resolved if item.id == "e2")

    assert right_result.metadata["conflict_downweighted"] is True
    assert right_result.metadata["original_confidence"] == 0.5
    assert right_result.confidence == 0.4
    assert stats.downweighted_evidence_ids == ["e2"]
    assert stats.conflicts[0].resolution_action == ResolutionAction.DOWNWEIGHT_RIGHT
