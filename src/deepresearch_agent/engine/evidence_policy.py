from __future__ import annotations

from deepresearch_agent.schemas import Evidence


class EvidencePolicy:
    """Evaluate whether collected evidence is sufficient for writing."""

    def __init__(self, min_total_evidences: int = 4) -> None:
        self.min_total_evidences = min_total_evidences

    def is_sufficient(self, evidences: list[Evidence]) -> bool:
        return len(evidences) >= self.min_total_evidences

    def reason_if_insufficient(self, evidences: list[Evidence]) -> str | None:
        if self.is_sufficient(evidences):
            return None
        return (
            f"Collected evidence count {len(evidences)} is below minimum "
            f"{self.min_total_evidences}."
        )
