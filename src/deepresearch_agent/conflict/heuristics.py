from __future__ import annotations

import re

import numpy as np

from deepresearch_agent.memory.embedding import BaseEmbeddingProvider
from deepresearch_agent.schemas import Evidence


ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("improve", "degrade"),
    ("increase", "decrease"),
    ("enhance", "reduce"),
    ("support", "oppose"),
    ("effective", "ineffective"),
    ("accurate", "inaccurate"),
    ("reliable", "unreliable"),
    ("scalable", "not scalable"),
    ("robust", "fragile"),
    ("efficient", "inefficient"),
    ("beneficial", "harmful"),
    ("consistent", "inconsistent"),
    ("possible", "impossible"),
    ("safe", "unsafe"),
    ("valid", "invalid"),
)

NEGATION_TERMS = {
    "not",
    "no",
    "never",
    "lack",
    "lacks",
    "without",
    "fail",
    "fails",
    "cannot",
    "unable",
    "impossible",
}

METRIC_TERMS = {
    "accuracy",
    "latency",
    "memory",
    "cost",
    "performance",
    "scalability",
    "coherence",
}

POSITIVE_DIRECTION_TERMS = {
    "increase",
    "increases",
    "increased",
    "increasing",
    "improve",
    "improves",
    "improved",
    "improving",
    "higher",
    "enhance",
    "enhances",
    "enhanced",
}

NEGATIVE_DIRECTION_TERMS = {
    "decrease",
    "decreases",
    "decreased",
    "decreasing",
    "reduce",
    "reduces",
    "reduced",
    "reducing",
    "lower",
    "degrade",
    "degrades",
    "degraded",
    "degrading",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "can",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def normalize_for_conflict(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9.%\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def extract_subject_terms(text: str) -> set[str]:
    normalized = normalize_for_conflict(text)
    tokens = set(re.findall(r"[a-z][a-z0-9-]*", normalized))
    directional = POSITIVE_DIRECTION_TERMS | NEGATIVE_DIRECTION_TERMS | NEGATION_TERMS
    antonyms = {term for pair in ANTONYM_PAIRS for item in pair for term in item.split()}
    return {
        token
        for token in tokens
        if len(token) > 2
        and token not in STOPWORDS
        and token not in directional
        and token not in antonyms
    }


def has_antonym_conflict(text_a: str, text_b: str) -> tuple[bool, str | None]:
    normalized_a = normalize_for_conflict(text_a)
    normalized_b = normalize_for_conflict(text_b)
    if not _has_shared_subject(normalized_a, normalized_b):
        return False, None

    for left, right in ANTONYM_PAIRS:
        if _contains_term(normalized_a, left) and _contains_term(normalized_b, right):
            return True, f"Matched opposing terms '{left}' and '{right}' on a shared subject."
        if _contains_term(normalized_a, right) and _contains_term(normalized_b, left):
            return True, f"Matched opposing terms '{right}' and '{left}' on a shared subject."
    return False, None


def detect_numeric_direction_conflict(text_a: str, text_b: str) -> tuple[bool, str | None]:
    normalized_a = normalize_for_conflict(text_a)
    normalized_b = normalize_for_conflict(text_b)
    metrics_a = _matched_terms(normalized_a, METRIC_TERMS)
    metrics_b = _matched_terms(normalized_b, METRIC_TERMS)
    shared_metrics = sorted(metrics_a & metrics_b)
    if not shared_metrics:
        return False, None

    direction_a = _direction(normalized_a)
    direction_b = _direction(normalized_b)
    if direction_a and direction_b and direction_a != direction_b:
        metric = shared_metrics[0]
        return True, f"Metric '{metric}' has opposing directions: {direction_a} vs {direction_b}."
    return False, None


def detect_near_duplicate(
    evidence_a: Evidence,
    evidence_b: Evidence,
    embedding_provider: BaseEmbeddingProvider,
    threshold: float = 0.92,
) -> tuple[bool, float]:
    left = embedding_provider.embed_text(_evidence_text(evidence_a))
    right = embedding_provider.embed_text(_evidence_text(evidence_b))
    similarity = _cosine_similarity(left, right)
    return similarity >= threshold, similarity


def detect_semantic_opposition(
    evidence_a: Evidence,
    evidence_b: Evidence,
    embedding_provider: BaseEmbeddingProvider,
    similarity_threshold: float = 0.65,
) -> tuple[bool, str | None]:
    text_a = _evidence_text(evidence_a)
    text_b = _evidence_text(evidence_b)
    similarity = _cosine_similarity(
        embedding_provider.embed_text(text_a),
        embedding_provider.embed_text(text_b),
    )
    if similarity < similarity_threshold:
        return False, None

    antonym_conflict, antonym_reason = has_antonym_conflict(text_a, text_b)
    if antonym_conflict:
        return True, f"Semantic similarity {similarity:.3f}; {antonym_reason}"

    numeric_conflict, numeric_reason = detect_numeric_direction_conflict(text_a, text_b)
    if numeric_conflict:
        return True, f"Semantic similarity {similarity:.3f}; {numeric_reason}"

    normalized_a = normalize_for_conflict(text_a)
    normalized_b = normalize_for_conflict(text_b)
    if _has_shared_subject(normalized_a, normalized_b) and _negation_mismatch(normalized_a, normalized_b):
        return True, f"Semantic similarity {similarity:.3f} with opposing negation on a shared subject."
    return False, None


def _evidence_text(evidence: Evidence) -> str:
    return f"{evidence.title}\n{evidence.content}"


def _has_shared_subject(text_a: str, text_b: str) -> bool:
    return bool(extract_subject_terms(text_a) & extract_subject_terms(text_b))


def _contains_term(text: str, term: str) -> bool:
    if " " in term:
        return term in text
    return re.search(rf"\b{re.escape(term)}(?:s|d|ing)?\b", text) is not None


def _matched_terms(text: str, terms: set[str]) -> set[str]:
    return {term for term in terms if _contains_term(text, term)}


def _direction(text: str) -> str | None:
    has_positive = bool(_matched_terms(text, POSITIVE_DIRECTION_TERMS))
    has_negative = bool(_matched_terms(text, NEGATIVE_DIRECTION_TERMS))
    if has_positive and not has_negative:
        return "positive"
    if has_negative and not has_positive:
        return "negative"
    return None


def _negation_mismatch(text_a: str, text_b: str) -> bool:
    return bool(_matched_terms(text_a, NEGATION_TERMS)) != bool(_matched_terms(text_b, NEGATION_TERMS))


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right) / (left_norm * right_norm))))
