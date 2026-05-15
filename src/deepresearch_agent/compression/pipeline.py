from __future__ import annotations

from deepresearch_agent.compression.l1_embedding_filter import L1EmbeddingFilter
from deepresearch_agent.compression.l2_textrank_filter import L2TextRankFilter
from deepresearch_agent.compression.l3_original_context import L3OriginalContextBuilder
from deepresearch_agent.compression.scoring import compute_final_score, source_quality_to_score
from deepresearch_agent.memory.embedding import BaseEmbeddingProvider
from deepresearch_agent.memory.source_quality import classify_source_url
from deepresearch_agent.schemas import CompressionStats, Evidence


class ContextCompressionPipeline:
    def __init__(
        self,
        embedding_provider: BaseEmbeddingProvider,
        l1_top_n: int = 20,
        l2_top_k: int = 8,
        enabled: bool = True,
        scoring_weights: dict | None = None,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.l1_top_n = l1_top_n
        self.l2_top_k = l2_top_k
        self.enabled = enabled
        self.scoring_weights = scoring_weights
        self.strategy = "L1_embedding_to_L2_textrank_to_L3_original"

    def compress(
        self,
        query: str,
        evidences: list[Evidence],
    ) -> tuple[list[Evidence], CompressionStats]:
        original_token_estimate = _estimate_tokens(evidences)
        if not self.enabled:
            stats = CompressionStats(
                enabled=False,
                l1_input_count=len(evidences),
                l1_selected_count=len(evidences),
                l2_input_count=len(evidences),
                l2_selected_count=len(evidences),
                final_selected_count=len(evidences),
                original_token_estimate=original_token_estimate,
                compressed_token_estimate=original_token_estimate,
                compression_ratio=1.0,
                selected_evidence_ids=[evidence.id for evidence in evidences],
                dropped_evidence_ids=[],
                strategy=self.strategy,
            )
            return evidences, stats

        l1_filter = L1EmbeddingFilter(self.embedding_provider, top_n=self.l1_top_n)
        l1_results = l1_filter.filter(query, evidences)
        l2_filter = L2TextRankFilter(top_k=self.l2_top_k)
        l2_results = l2_filter.filter(l1_results)

        for item in l2_results:
            source_quality = classify_source_url(item.evidence.source_url).value
            item.source_quality_score = source_quality_to_score(source_quality)
            item.confidence_score = item.evidence.confidence
            item.final_score = compute_final_score(
                item.l1_similarity_score,
                item.l2_textrank_score,
                item.confidence_score,
                item.source_quality_score,
                self.scoring_weights,
            )

        selected = L3OriginalContextBuilder().build_context(l2_results)
        selected_ids = [evidence.id for evidence in selected]
        dropped_ids = [evidence.id for evidence in evidences if evidence.id not in set(selected_ids)]
        compressed_token_estimate = _estimate_tokens(selected)
        compression_ratio = (
            compressed_token_estimate / original_token_estimate
            if original_token_estimate > 0
            else 1.0
        )
        stats = CompressionStats(
            enabled=True,
            l1_input_count=len(evidences),
            l1_selected_count=len(l1_results),
            l2_input_count=len(l1_results),
            l2_selected_count=len(l2_results),
            final_selected_count=len(selected),
            original_token_estimate=original_token_estimate,
            compressed_token_estimate=compressed_token_estimate,
            compression_ratio=round(compression_ratio, 6),
            selected_evidence_ids=selected_ids,
            dropped_evidence_ids=dropped_ids,
            strategy=self.strategy,
        )
        return selected, stats


def _estimate_tokens(evidences: list[Evidence]) -> int:
    total = 0
    for evidence in evidences:
        total += len(f"{evidence.title} {evidence.content}".split())
    return total
