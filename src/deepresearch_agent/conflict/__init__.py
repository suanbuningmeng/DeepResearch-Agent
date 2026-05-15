from deepresearch_agent.conflict.detector import EvidenceConflictDetector
from deepresearch_agent.conflict.heuristics import (
    detect_near_duplicate,
    detect_numeric_direction_conflict,
    detect_semantic_opposition,
    extract_subject_terms,
    has_antonym_conflict,
    normalize_for_conflict,
)
from deepresearch_agent.conflict.resolver import EvidenceConflictResolver
from deepresearch_agent.conflict.scoring import evidence_quality_score

__all__ = [
    "EvidenceConflictDetector",
    "EvidenceConflictResolver",
    "detect_near_duplicate",
    "detect_numeric_direction_conflict",
    "detect_semantic_opposition",
    "evidence_quality_score",
    "extract_subject_terms",
    "has_antonym_conflict",
    "normalize_for_conflict",
]
