from __future__ import annotations

from deepresearch_agent.schemas import CompressedEvidence


class BaseCompressionStage:
    def filter(self, *args: object, **kwargs: object) -> list[CompressedEvidence]:
        raise NotImplementedError
