from __future__ import annotations

import hashlib
import re

from deepresearch_agent.schemas import Evidence


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def content_hash(title: str, content: str) -> str:
    normalized = normalize_text(f"{title}\n{content}")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_near_duplicate(existing: Evidence, new: Evidence, threshold: float = 0.95) -> bool:
    existing_tokens = set(normalize_text(f"{existing.title} {existing.content}").split())
    new_tokens = set(normalize_text(f"{new.title} {new.content}").split())
    if not existing_tokens or not new_tokens:
        return False
    overlap = len(existing_tokens & new_tokens) / len(existing_tokens | new_tokens)
    return overlap >= threshold
