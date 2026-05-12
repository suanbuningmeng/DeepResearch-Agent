from __future__ import annotations

from deepresearch_agent.memory.base import BaseMemoryStore
from deepresearch_agent.memory.dedupe import content_hash
from deepresearch_agent.schemas import Evidence


class MemoryStore(BaseMemoryStore):
    def __init__(self) -> None:
        self._evidences: list[Evidence] = []
        self._content_hashes: set[str] = set()
        self.inserted_evidence_count = 0
        self.duplicate_evidence_count = 0

    def add_evidence(self, evidence: Evidence) -> bool:
        """Store one evidence item in memory."""
        digest = content_hash(evidence.title, evidence.content)
        if digest in self._content_hashes:
            self.duplicate_evidence_count += 1
            return False
        self._content_hashes.add(digest)
        self._evidences.append(evidence)
        self.inserted_evidence_count += 1
        return True

    def add_evidences(self, evidences: list[Evidence]) -> dict:
        inserted_count = 0
        duplicate_count = 0
        for evidence in evidences:
            if self.add_evidence(evidence):
                inserted_count += 1
            else:
                duplicate_count += 1
        return {"inserted_count": inserted_count, "duplicate_count": duplicate_count}

    def list_evidences(self) -> list[Evidence]:
        """Return a copy of all evidence items currently stored."""
        return list(self._evidences)

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        for evidence in self._evidences:
            if evidence.id == evidence_id:
                return evidence
        return None

    def search_evidences(self, query: str, top_k: int = 5) -> list[Evidence]:
        return sorted(self._evidences, key=lambda evidence: evidence.confidence, reverse=True)[:top_k]

    def clear(self) -> None:
        self._evidences.clear()
        self._content_hashes.clear()
        self.inserted_evidence_count = 0
        self.duplicate_evidence_count = 0
