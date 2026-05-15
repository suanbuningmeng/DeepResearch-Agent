from __future__ import annotations

from deepresearch_agent.schemas import Evidence, EvidenceConflict


class BaseConflictDetector:
    def detect(self, evidences: list[Evidence]) -> list[EvidenceConflict]:
        raise NotImplementedError


class BaseConflictResolver:
    def resolve(
        self,
        evidences: list[Evidence],
        conflicts: list[EvidenceConflict],
    ):
        raise NotImplementedError
