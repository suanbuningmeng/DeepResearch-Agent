from __future__ import annotations


DEFAULT_WEIGHTS = {
    "l1_similarity_score": 0.35,
    "l2_textrank_score": 0.25,
    "confidence_score": 0.25,
    "source_quality_score": 0.15,
}


def source_quality_to_score(source_quality: str | None) -> float:
    return {
        "real_url": 1.0,
        "null_source": 0.55,
        "example_url": 0.2,
        "mock_source": 0.4,
        "model_generated": 0.35,
        "unknown": 0.5,
    }.get(source_quality or "unknown", 0.5)


def compute_final_score(
    l1_similarity_score: float,
    l2_textrank_score: float,
    confidence_score: float,
    source_quality_score: float,
    weights: dict | None = None,
) -> float:
    active_weights = DEFAULT_WEIGHTS | (weights or {})
    score = (
        _clean(l1_similarity_score) * active_weights.get("l1_similarity_score", 0.0)
        + _clean(l2_textrank_score) * active_weights.get("l2_textrank_score", 0.0)
        + _clean(confidence_score) * active_weights.get("confidence_score", 0.0)
        + _clean(source_quality_score) * active_weights.get("source_quality_score", 0.0)
    )
    return max(0.0, min(1.0, float(score)))


def _clean(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))
