from __future__ import annotations

from deepresearch_agent.schemas import Evidence


class BaseMemoryStore:
    """Shared memory interface for evidence storage and retrieval."""

    def add_evidence(self, evidence: Evidence) -> bool:
        raise NotImplementedError

    def add_evidences(self, evidences: list[Evidence]) -> dict:
        raise NotImplementedError

    def list_evidences(self) -> list[Evidence]:
        raise NotImplementedError

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        raise NotImplementedError

    def search_evidences(self, query: str, top_k: int = 5) -> list[Evidence]:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError
