from __future__ import annotations

import numpy as np

from deepresearch_agent.memory.embedding import HashingEmbeddingProvider
from deepresearch_agent.schemas import CompressedEvidence


class L2TextRankFilter:
    def __init__(
        self,
        top_k: int = 8,
        similarity_threshold: float = 0.1,
        damping: float = 0.85,
        max_iter: int = 50,
        tolerance: float = 1e-6,
    ) -> None:
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.damping = damping
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.embedding_provider = HashingEmbeddingProvider()

    def filter(
        self,
        compressed_evidences: list[CompressedEvidence],
    ) -> list[CompressedEvidence]:
        count = len(compressed_evidences)
        if count == 0:
            return []
        if count == 1:
            compressed_evidences[0].l2_textrank_score = 1.0
            return compressed_evidences

        vectors = [
            self.embedding_provider.embed_text(f"{item.evidence.title}\n{item.evidence.content}")
            for item in compressed_evidences
        ]
        adjacency = np.zeros((count, count), dtype=np.float32)
        for i in range(count):
            for j in range(count):
                if i == j:
                    continue
                similarity = _cosine_similarity(vectors[i], vectors[j])
                if similarity > self.similarity_threshold:
                    adjacency[i, j] = similarity

        scores = _pagerank(adjacency, self.damping, self.max_iter, self.tolerance)
        max_score = float(scores.max()) if scores.size else 0.0
        if max_score > 0:
            scores = scores / max_score
        for item, score in zip(compressed_evidences, scores):
            item.l2_textrank_score = float(score)

        return sorted(
            compressed_evidences,
            key=lambda item: item.l2_textrank_score,
            reverse=True,
        )[: self.top_k]


def _pagerank(
    adjacency: np.ndarray,
    damping: float,
    max_iter: int,
    tolerance: float,
) -> np.ndarray:
    count = adjacency.shape[0]
    if count == 0:
        return np.array([], dtype=np.float32)
    if not adjacency.any():
        return np.ones(count, dtype=np.float32)

    row_sums = adjacency.sum(axis=1)
    transition = np.zeros_like(adjacency)
    for index, row_sum in enumerate(row_sums):
        if row_sum > 0:
            transition[index] = adjacency[index] / row_sum
        else:
            transition[index] = np.ones(count, dtype=np.float32) / count

    scores = np.ones(count, dtype=np.float32) / count
    teleport = (1.0 - damping) / count
    for _ in range(max_iter):
        updated = teleport + damping * transition.T @ scores
        if np.linalg.norm(updated - scores, ord=1) < tolerance:
            scores = updated
            break
        scores = updated
    return scores


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right) / (left_norm * right_norm))))
