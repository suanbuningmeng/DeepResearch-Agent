from __future__ import annotations

from deepresearch_agent.compression.scoring import source_quality_to_score
from deepresearch_agent.conflict.base import BaseConflictDetector
from deepresearch_agent.conflict.heuristics import (
    detect_near_duplicate,
    detect_numeric_direction_conflict,
    detect_semantic_opposition,
    has_antonym_conflict,
)
from deepresearch_agent.conflict.scoring import evidence_quality_score
from deepresearch_agent.memory.dedupe import content_hash
from deepresearch_agent.memory.embedding import BaseEmbeddingProvider
from deepresearch_agent.memory.source_quality import classify_source_url
from deepresearch_agent.schemas import ConflictSeverity, ConflictType, Evidence, EvidenceConflict


class EvidenceConflictDetector(BaseConflictDetector):
    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        near_duplicate_threshold: float = 0.92,
        semantic_similarity_threshold: float = 0.65,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.near_duplicate_threshold = near_duplicate_threshold
        self.semantic_similarity_threshold = semantic_similarity_threshold

    def detect(self, evidences: list[Evidence]) -> list[EvidenceConflict]:
        if len(evidences) < 2:
            return []

        conflicts: list[EvidenceConflict] = []
        for left_index in range(len(evidences)):
            for right_index in range(left_index + 1, len(evidences)):
                left = evidences[left_index]
                right = evidences[right_index]
                pair_conflicts = self._detect_pair(left, right, len(conflicts) + 1)
                conflicts.extend(pair_conflicts)
        return conflicts

    def _detect_pair(
        self,
        left: Evidence,
        right: Evidence,
        index: int,
    ) -> list[EvidenceConflict]:
        left_text = f"{left.title}\n{left.content}"
        right_text = f"{right.title}\n{right.content}"
        if content_hash(left.title, left.content) == content_hash(right.title, right.content):
            return [
                self._build_conflict(
                    left,
                    right,
                    index,
                    ConflictType.DUPLICATE,
                    ConflictSeverity.LOW,
                    "Evidence title and content have identical normalized hashes.",
                )
            ]

        pair_conflicts: list[EvidenceConflict] = []
        is_near_duplicate, similarity = detect_near_duplicate(
            left,
            right,
            self.embedding_provider,
            self.near_duplicate_threshold,
        )
        if is_near_duplicate:
            pair_conflicts.append(
                self._build_conflict(
                    left,
                    right,
                    index,
                    ConflictType.NEAR_DUPLICATE,
                    ConflictSeverity.MEDIUM,
                    f"Evidence items are near duplicates with similarity {similarity:.3f}.",
                )
            )

        has_antonym, antonym_reason = has_antonym_conflict(left_text, right_text)
        if has_antonym:
            pair_conflicts.append(
                self._build_conflict(
                    left,
                    right,
                    index + len(pair_conflicts),
                    ConflictType.ANTONYM_CONTRADICTION,
                    ConflictSeverity.HIGH,
                    antonym_reason or "Evidence contains opposing terms on a shared subject.",
                )
            )
            return pair_conflicts

        has_numeric_conflict, numeric_reason = detect_numeric_direction_conflict(left_text, right_text)
        if has_numeric_conflict:
            pair_conflicts.append(
                self._build_conflict(
                    left,
                    right,
                    index + len(pair_conflicts),
                    ConflictType.NUMERIC_DIRECTION_CONFLICT,
                    ConflictSeverity.HIGH,
                    numeric_reason or "Evidence contains opposing numeric or metric directions.",
                )
            )
            return pair_conflicts

        has_semantic_opposition, semantic_reason = detect_semantic_opposition(
            left,
            right,
            self.embedding_provider,
            self.semantic_similarity_threshold,
        )
        if has_semantic_opposition:
            pair_conflicts.append(
                self._build_conflict(
                    left,
                    right,
                    index + len(pair_conflicts),
                    ConflictType.SEMANTIC_OPPOSITION,
                    ConflictSeverity.MEDIUM,
                    semantic_reason or "Evidence is semantically similar but directionally opposed.",
                )
            )
            return pair_conflicts

        has_source_quality_gap, source_reason = self._source_quality_conflict(left, right, similarity)
        if has_source_quality_gap:
            pair_conflicts.append(
                self._build_conflict(
                    left,
                    right,
                    index + len(pair_conflicts),
                    ConflictType.SOURCE_QUALITY_CONFLICT,
                    ConflictSeverity.LOW,
                    source_reason,
                )
            )
        return pair_conflicts

    def _source_quality_conflict(
        self,
        left: Evidence,
        right: Evidence,
        similarity: float,
    ) -> tuple[bool, str]:
        if similarity < self.semantic_similarity_threshold:
            return False, ""
        left_quality = source_quality_to_score(classify_source_url(left.source_url).value)
        right_quality = source_quality_to_score(classify_source_url(right.source_url).value)
        if abs(left_quality - right_quality) < 0.45:
            return False, ""
        return True, (
            "Evidence items discuss a similar topic but have a large source-quality gap "
            f"({left_quality:.2f} vs {right_quality:.2f})."
        )

    def _build_conflict(
        self,
        left: Evidence,
        right: Evidence,
        index: int,
        conflict_type: ConflictType,
        severity: ConflictSeverity,
        reason: str,
    ) -> EvidenceConflict:
        return EvidenceConflict(
            id=f"conflict_{left.id}_{right.id}_{index}",
            left_evidence_id=left.id,
            right_evidence_id=right.id,
            conflict_type=conflict_type,
            severity=severity,
            reason=reason,
            left_score=evidence_quality_score(left),
            right_score=evidence_quality_score(right),
        )
