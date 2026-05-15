from __future__ import annotations

import re

from deepresearch_agent.schemas import Evidence
from deepresearch_agent.search.schemas import CitationValidationResult, FetchedDocument


class CitationValidator:
    def validate(
        self,
        evidence: Evidence,
        document: FetchedDocument | None,
    ) -> CitationValidationResult:
        if not evidence.source_url:
            return CitationValidationResult(
                evidence_id=evidence.id,
                source_url=evidence.source_url,
                validation_status="no_source",
                url_reachable=False,
                title_match_score=0.0,
                content_overlap_score=0.0,
                support_score=0.0,
                reason="Evidence has no source_url.",
            )
        if str(evidence.source_url).startswith("mock://degraded"):
            return CitationValidationResult(
                evidence_id=evidence.id,
                source_url=evidence.source_url,
                validation_status="unsupported",
                url_reachable=False,
                title_match_score=0.0,
                content_overlap_score=0.0,
                support_score=0.05,
                reason="Degraded fallback sources are not treated as reliable citations.",
            )
        if document is None or not document.fetch_success:
            return CitationValidationResult(
                evidence_id=evidence.id,
                source_url=evidence.source_url,
                validation_status="url_unreachable",
                url_reachable=False,
                title_match_score=0.0,
                content_overlap_score=0.0,
                support_score=0.1,
                reason="Source document could not be fetched.",
            )
        title_text = f"{document.title or ''} {document.snippet or ''}"
        content_text = f"{document.text or ''} {document.snippet or ''}"
        title_score = _token_overlap(evidence.title, title_text)
        content_score = _token_overlap(evidence.content, content_text)
        support_score = max(0.0, min(1.0, 0.35 * title_score + 0.65 * content_score))
        penalties: list[str] = []
        if content_score < 0.35:
            support_score *= 0.75
            penalties.append("weak claim-to-content overlap")
        if title_score < 0.15:
            support_score *= 0.9
            penalties.append("weak title match")
        if evidence.metadata.get("grounded_fallback"):
            support_score *= 0.85
            penalties.append("grounded fallback evidence")
        category_penalty = _paper_category_penalty(evidence)
        if category_penalty:
            support_score *= 1.0 - category_penalty
            penalties.append("paper category is only weakly aligned with the task")
        support_score = max(0.0, min(1.0, support_score))
        if support_score >= 0.68 and title_score >= 0.18 and content_score >= 0.45 and not category_penalty:
            status = "supported"
        elif support_score >= 0.28:
            status = "partially_supported"
        else:
            status = "unsupported"
        reason_parts = [
            f"classified as {status}",
            f"title_overlap={title_score:.2f}",
            f"content_overlap={content_score:.2f}",
        ]
        if penalties:
            reason_parts.append("penalties=" + ", ".join(penalties))
        return CitationValidationResult(
            evidence_id=evidence.id,
            source_url=evidence.source_url,
            validation_status=status,
            url_reachable=True,
            title_match_score=round(title_score, 6),
            content_overlap_score=round(content_score, 6),
            support_score=round(support_score, 6),
            reason="Citation support proxy " + "; ".join(reason_parts) + ".",
        )


def _token_overlap(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _tokens(text: str) -> set[str]:
    stopwords = {"the", "and", "for", "with", "that", "this", "are", "can", "into", "from"}
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2 and token not in stopwords}


def _paper_category_penalty(evidence: Evidence) -> float:
    metadata = evidence.metadata.get("source_metadata") or {}
    if not isinstance(metadata, dict):
        return 0.0
    primary_category = str(metadata.get("primary_category") or "")
    text = f"{evidence.metadata.get('task_name') or ''} {evidence.title} {evidence.content}".lower()
    if primary_category.startswith(("cs.RO", "cs.CV")) and any(term in text for term in ("llm", "large language model", "language model", "evaluation", "benchmark")):
        return 0.25
    return 0.0
