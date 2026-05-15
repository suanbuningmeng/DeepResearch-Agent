from __future__ import annotations

from deepresearch_agent.schemas import CompressedEvidence, Evidence


class L3OriginalContextBuilder:
    def build_context(
        self,
        compressed_evidences: list[CompressedEvidence],
    ) -> list[Evidence]:
        selected: list[Evidence] = []
        for item in sorted(compressed_evidences, key=lambda evidence: evidence.final_score, reverse=True):
            evidence = item.evidence.model_copy(deep=True)
            evidence.metadata["compression"] = {
                "l1_similarity_score": item.l1_similarity_score,
                "l2_textrank_score": item.l2_textrank_score,
                "confidence_score": item.confidence_score,
                "source_quality_score": item.source_quality_score,
                "final_score": item.final_score,
                "compression_level": item.compression_level,
            }
            selected.append(evidence)
        return selected
