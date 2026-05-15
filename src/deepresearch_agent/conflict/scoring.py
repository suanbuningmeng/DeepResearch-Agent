from __future__ import annotations

from deepresearch_agent.compression.scoring import source_quality_to_score
from deepresearch_agent.memory.source_quality import classify_source_url
from deepresearch_agent.schemas import Evidence


def evidence_quality_score(evidence: Evidence) -> float:
    compression_score = _compression_final_score(evidence)
    source_quality = classify_source_url(evidence.source_url).value
    score = (
        _clean(evidence.confidence) * 0.4
        + source_quality_to_score(source_quality) * 0.35
        + compression_score * 0.25
    )
    return round(max(0.0, min(1.0, score)), 6)


def _compression_final_score(evidence: Evidence) -> float:
    compression = evidence.metadata.get("compression")
    if not isinstance(compression, dict):
        return 0.0
    return _clean(compression.get("final_score"))


def _clean(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
