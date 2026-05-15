from __future__ import annotations

import numpy as np

from deepresearch_agent.memory.embedding import BaseEmbeddingProvider
from deepresearch_agent.schemas import CompressedEvidence, Evidence


class L1EmbeddingFilter:
    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        top_n: int = 20,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.top_n = top_n

    def filter(
        self,
        query: str,
        evidences: list[Evidence],
    ) -> list[CompressedEvidence]:
        if not evidences:
            return []
        query_vector = self.embedding_provider.embed_text(query)
        scored: list[CompressedEvidence] = []
        for evidence in evidences:
            evidence_vector = self.embedding_provider.embed_text(f"{evidence.title}\n{evidence.content}")
            similarity = _cosine_similarity(query_vector, evidence_vector)
            scored.append(
                CompressedEvidence(
                    evidence=evidence,
                    l1_similarity_score=similarity,
                    confidence_score=evidence.confidence,
                )
            )
        return sorted(scored, key=lambda item: item.l1_similarity_score, reverse=True)[: self.top_n]


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right) / (left_norm * right_norm))))
