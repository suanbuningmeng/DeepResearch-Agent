from __future__ import annotations

from deepresearch_agent.schemas import Evidence


class MemoryStore:
    def __init__(self) -> None:
        self._evidences: list[Evidence] = []

    def add_evidence(self, evidence: Evidence) -> None:
        """Store one evidence item in memory."""
        self._evidences.append(evidence)

    def add_evidences(self, evidences: list[Evidence]) -> None:
        self._evidences.extend(evidences)

    def list_evidences(self) -> list[Evidence]:
        """Return a copy of all evidence items currently stored."""
        return list(self._evidences)

    def clear(self) -> None:
        self._evidences.clear()
