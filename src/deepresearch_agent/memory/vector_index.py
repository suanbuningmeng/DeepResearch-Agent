from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyVectorIndex:
    """Small in-process cosine-similarity vector index backed by numpy arrays."""

    def __init__(self) -> None:
        self.evidence_ids: list[str] = []
        self.vectors: list[np.ndarray] = []

    def add(self, evidence_id: str, vector: np.ndarray) -> None:
        if evidence_id in self.evidence_ids:
            return
        self.evidence_ids.append(evidence_id)
        self.vectors.append(vector.astype(np.float32))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if not self.vectors or top_k < 1:
            return []
        matrix = np.vstack(self.vectors)
        scores = matrix @ query_vector.astype(np.float32)
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.evidence_ids[index], float(scores[index])) for index in order]

    def save(self, index_path: str) -> None:
        path = Path(index_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.vstack(self.vectors) if self.vectors else np.empty((0, 0), dtype=np.float32)
        np.savez(path, evidence_ids=np.array(self.evidence_ids), vectors=matrix)

    @classmethod
    def load(cls, index_path: str) -> "NumpyVectorIndex":
        index = cls()
        path = Path(index_path)
        if not path.exists():
            return index
        data = np.load(path, allow_pickle=False)
        index.evidence_ids = [str(item) for item in data["evidence_ids"].tolist()]
        vectors = data["vectors"]
        if vectors.size:
            index.vectors = [vectors[row].astype(np.float32) for row in range(vectors.shape[0])]
        return index
