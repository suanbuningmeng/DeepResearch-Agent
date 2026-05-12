from __future__ import annotations

import hashlib
import re

import numpy as np


class BaseEmbeddingProvider:
    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError


class HashingEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic local hashing embeddings for lightweight retrieval."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.astype(np.float32)
