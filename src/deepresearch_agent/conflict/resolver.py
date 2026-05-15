from __future__ import annotations

from collections import Counter

from deepresearch_agent.compression.scoring import source_quality_to_score
from deepresearch_agent.conflict.base import BaseConflictResolver
from deepresearch_agent.conflict.scoring import evidence_quality_score
from deepresearch_agent.memory.source_quality import SourceQuality, classify_source_url
from deepresearch_agent.schemas import (
    ConflictStats,
    ConflictType,
    Evidence,
    EvidenceConflict,
    ResolutionAction,
)


class EvidenceConflictResolver(BaseConflictResolver):
    def __init__(
        self,
        drop_low_quality_conflicts: bool = True,
        mark_unresolved_conflicts: bool = True,
    ) -> None:
        self.drop_low_quality_conflicts = drop_low_quality_conflicts
        self.mark_unresolved_conflicts = mark_unresolved_conflicts

    def resolve(
        self,
        evidences: list[Evidence],
        conflicts: list[EvidenceConflict],
    ) -> tuple[list[Evidence], ConflictStats]:
        evidence_by_id = {evidence.id: evidence.model_copy(deep=True) for evidence in evidences}
        dropped_ids: set[str] = set()
        downweighted_ids: set[str] = set()
        marked_ids: set[str] = set()

        for conflict in conflicts:
            left = evidence_by_id.get(conflict.left_evidence_id)
            right = evidence_by_id.get(conflict.right_evidence_id)
            if left is None or right is None:
                continue

            if conflict.conflict_type in {ConflictType.DUPLICATE, ConflictType.NEAR_DUPLICATE}:
                self._drop_lower_quality(conflict, dropped_ids)
            elif conflict.conflict_type in {
                ConflictType.ANTONYM_CONTRADICTION,
                ConflictType.NUMERIC_DIRECTION_CONFLICT,
                ConflictType.SEMANTIC_OPPOSITION,
            }:
                self._resolve_directional_conflict(
                    conflict,
                    left,
                    right,
                    dropped_ids,
                    downweighted_ids,
                    marked_ids,
                )
            else:
                self._downweight_or_mark_lower_quality(
                    conflict,
                    left,
                    right,
                    dropped_ids,
                    downweighted_ids,
                    marked_ids,
                )

            conflict.resolved = True

        output = [
            evidence_by_id[evidence.id]
            for evidence in evidences
            if evidence.id in evidence_by_id and evidence.id not in dropped_ids
        ]
        stats = _build_conflict_stats(
            input_evidence_count=len(evidences),
            output_evidence_count=len(output),
            conflicts=conflicts,
            dropped_ids=dropped_ids,
            downweighted_ids=downweighted_ids,
            marked_ids=marked_ids,
        )
        return output, stats

    def _drop_lower_quality(
        self,
        conflict: EvidenceConflict,
        dropped_ids: set[str],
    ) -> None:
        if conflict.left_score >= conflict.right_score:
            conflict.resolution_action = ResolutionAction.DROP_RIGHT
            dropped_ids.add(conflict.right_evidence_id)
        else:
            conflict.resolution_action = ResolutionAction.DROP_LEFT
            dropped_ids.add(conflict.left_evidence_id)

    def _resolve_directional_conflict(
        self,
        conflict: EvidenceConflict,
        left: Evidence,
        right: Evidence,
        dropped_ids: set[str],
        downweighted_ids: set[str],
        marked_ids: set[str],
    ) -> None:
        forced_drop = _low_quality_source_against_real_url(left, right)
        if forced_drop is not None:
            conflict.resolution_action = forced_drop
            dropped_ids.add(_action_target_id(conflict, forced_drop))
            return

        score_gap = abs(conflict.left_score - conflict.right_score)
        left_better = conflict.left_score > conflict.right_score
        better_has_confidence = (
            left.confidence >= right.confidence if left_better else right.confidence >= left.confidence
        )
        if score_gap >= 0.25 and better_has_confidence and self.drop_low_quality_conflicts:
            action = ResolutionAction.DROP_RIGHT if left_better else ResolutionAction.DROP_LEFT
            conflict.resolution_action = action
            dropped_ids.add(_action_target_id(conflict, action))
            return
        if score_gap >= 0.12 and better_has_confidence:
            action = ResolutionAction.DOWNWEIGHT_RIGHT if left_better else ResolutionAction.DOWNWEIGHT_LEFT
            conflict.resolution_action = action
            target = right if left_better else left
            _downweight_evidence(target, conflict)
            downweighted_ids.add(target.id)
            return

        self._mark_for_writer(conflict, left, right, marked_ids)

    def _downweight_or_mark_lower_quality(
        self,
        conflict: EvidenceConflict,
        left: Evidence,
        right: Evidence,
        dropped_ids: set[str],
        downweighted_ids: set[str],
        marked_ids: set[str],
    ) -> None:
        del dropped_ids
        if abs(conflict.left_score - conflict.right_score) >= 0.12:
            left_better = conflict.left_score > conflict.right_score
            action = ResolutionAction.DOWNWEIGHT_RIGHT if left_better else ResolutionAction.DOWNWEIGHT_LEFT
            conflict.resolution_action = action
            target = right if left_better else left
            _downweight_evidence(target, conflict)
            downweighted_ids.add(target.id)
            return
        self._mark_for_writer(conflict, left, right, marked_ids)

    def _mark_for_writer(
        self,
        conflict: EvidenceConflict,
        left: Evidence,
        right: Evidence,
        marked_ids: set[str],
    ) -> None:
        conflict.resolution_action = ResolutionAction.MARK_FOR_WRITER
        if not self.mark_unresolved_conflicts:
            return
        _mark_evidence(left, conflict)
        _mark_evidence(right, conflict)
        marked_ids.update({left.id, right.id})


def _low_quality_source_against_real_url(
    left: Evidence,
    right: Evidence,
) -> ResolutionAction | None:
    left_quality = classify_source_url(left.source_url)
    right_quality = classify_source_url(right.source_url)
    low_quality_sources = {SourceQuality.EXAMPLE_URL, SourceQuality.NULL_SOURCE}
    if left_quality == SourceQuality.REAL_URL and right_quality in low_quality_sources:
        return ResolutionAction.DROP_RIGHT
    if right_quality == SourceQuality.REAL_URL and left_quality in low_quality_sources:
        return ResolutionAction.DROP_LEFT
    return None


def _action_target_id(conflict: EvidenceConflict, action: ResolutionAction) -> str:
    if action in {ResolutionAction.DROP_LEFT, ResolutionAction.DOWNWEIGHT_LEFT}:
        return conflict.left_evidence_id
    return conflict.right_evidence_id


def _downweight_evidence(evidence: Evidence, conflict: EvidenceConflict) -> None:
    original_confidence = evidence.metadata.get("original_confidence", evidence.confidence)
    adjusted_confidence = round(max(0.0, min(1.0, evidence.confidence * 0.8)), 6)
    evidence.metadata["conflict_downweighted"] = True
    evidence.metadata["original_confidence"] = original_confidence
    evidence.metadata["confidence"] = adjusted_confidence
    evidence.metadata.setdefault("conflict_ids", [])
    evidence.metadata["conflict_ids"].append(conflict.id)
    evidence.metadata["conflict_reason"] = conflict.reason
    evidence.confidence = adjusted_confidence


def _mark_evidence(evidence: Evidence, conflict: EvidenceConflict) -> None:
    evidence.metadata["conflict_marked"] = True
    evidence.metadata.setdefault("conflict_ids", [])
    evidence.metadata["conflict_ids"].append(conflict.id)
    evidence.metadata["conflict_reason"] = conflict.reason


def _build_conflict_stats(
    input_evidence_count: int,
    output_evidence_count: int,
    conflicts: list[EvidenceConflict],
    dropped_ids: set[str],
    downweighted_ids: set[str],
    marked_ids: set[str],
) -> ConflictStats:
    counts = Counter(conflict.conflict_type for conflict in conflicts)
    return ConflictStats(
        enabled=True,
        input_evidence_count=input_evidence_count,
        output_evidence_count=output_evidence_count,
        conflict_count=len(conflicts),
        duplicate_count=counts[ConflictType.DUPLICATE],
        near_duplicate_count=counts[ConflictType.NEAR_DUPLICATE],
        antonym_contradiction_count=counts[ConflictType.ANTONYM_CONTRADICTION],
        numeric_direction_conflict_count=counts[ConflictType.NUMERIC_DIRECTION_CONFLICT],
        semantic_opposition_count=counts[ConflictType.SEMANTIC_OPPOSITION],
        source_quality_conflict_count=counts[ConflictType.SOURCE_QUALITY_CONFLICT],
        off_topic_count=counts[ConflictType.OFF_TOPIC],
        dropped_evidence_ids=sorted(dropped_ids),
        downweighted_evidence_ids=sorted(downweighted_ids),
        marked_conflict_evidence_ids=sorted(marked_ids),
        conflicts=conflicts,
    )
