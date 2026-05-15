from __future__ import annotations

import re

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.report.quality import ensure_report_completeness
from deepresearch_agent.schemas import Evidence


class WriterAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self.last_quality_check: dict | None = None

    async def write(
        self,
        question: str,
        evidences: list[Evidence],
        top_k_per_task: int = 2,
    ) -> str:
        selected_evidences = select_top_evidences_per_task(evidences, top_k_per_task)
        evidence_lines = "\n".join(_evidence_prompt_line(evidence) for evidence in selected_evidences)
        conflict_notice = _conflict_notice(selected_evidences)
        fallback_notice = _fallback_notice(selected_evidences)

        prompt = (
            "You are a research writer for AI, LLM, and communication research surveys.\n"
            "Write a concise Markdown report grounded only in the provided evidence.\n"
            f"Question: {question}\n"
            f"Evidence:\n{evidence_lines}\n"
            f"{conflict_notice}"
            f"{fallback_notice}"
            "Use Markdown level-2 headings exactly as: ## Abstract, ## Key Findings, ## Evidence Summary, ## Limitations, ## Conclusion, ## References.\n"
            "Every key claim should cite at least one evidence id in square brackets, for example [task_1_grounded_evidence_1].\n"
            "Evidence Summary bullets must include the evidence id at the start of each bullet.\n"
            "References must list evidence id, title, source_url, and citation_validation_status.\n"
            "Do not use anonymous numeric citations such as [1] or [2] unless they are also mapped to evidence ids in References.\n"
            "Do not use degraded or grounded_fallback evidence as a core Key Finding unless no stronger evidence exists.\n"
            "If degraded or grounded_fallback evidence is used, state that limitation explicitly in ## Limitations.\n"
            "Keep the report concise, finish every bullet, and end with a complete ## References section."
        )

        markdown = await self.llm.agenerate(prompt, prompt_type="writer")
        markdown, quality = ensure_report_completeness(markdown)
        markdown = _ensure_evidence_summary_ids(markdown, selected_evidences)
        markdown = _ensure_fallback_limitations(markdown, selected_evidences)
        markdown = _ensure_references(markdown, selected_evidences)
        self.last_quality_check = quality
        return markdown


def select_top_evidences_per_task(
    evidences: list[Evidence],
    top_k_per_task: int = 2,
) -> list[Evidence]:
    if top_k_per_task < 1:
        return []

    grouped: dict[str, list[Evidence]] = {}
    for evidence in evidences:
        grouped.setdefault(evidence.task_id, []).append(evidence)

    selected: list[Evidence] = []
    for task_id in sorted(grouped):
        task_evidences = grouped[task_id]
        strong_candidates = [
            evidence
            for evidence in task_evidences
            if not _is_degraded(evidence)
            and not evidence.metadata.get("grounded_fallback")
            and _citation_rank(evidence) >= 1
        ]
        normal_candidates = [
            evidence
            for evidence in task_evidences
            if not _is_degraded(evidence)
            and _citation_rank(evidence) >= 0
        ]
        candidates = strong_candidates or normal_candidates or task_evidences
        selected.extend(sorted(candidates, key=_evidence_selection_key, reverse=True)[:top_k_per_task])
    return selected


def _evidence_selection_key(evidence: Evidence) -> tuple[float, float, float, float]:
    real_source = 1.0 if _is_real_url(evidence.source_url) else 0.0
    fallback_penalty = -0.35 if evidence.metadata.get("grounded_fallback") else 0.0
    degraded_penalty = -1.0 if _is_degraded(evidence) else 0.0
    return (
        real_source + degraded_penalty,
        float(_citation_rank(evidence)) + fallback_penalty,
        float(evidence.metadata.get("retrieval_relevance_score") or 0.0),
        evidence.confidence,
    )


def _citation_rank(evidence: Evidence) -> int:
    status = str(evidence.metadata.get("citation_validation_status") or "").lower()
    if status == "supported":
        return 2
    if status == "partially_supported":
        return 1
    if not status and _is_real_url(evidence.source_url):
        return 1
    return 0


def _is_degraded(evidence: Evidence) -> bool:
    return bool(evidence.metadata.get("degraded")) or str(evidence.source_url or "").startswith("mock://degraded/")


def _is_real_url(source_url: str | None) -> bool:
    return bool(source_url and source_url.startswith(("http://", "https://")))


def _evidence_prompt_line(evidence: Evidence) -> str:
    status = str(evidence.metadata.get("citation_validation_status") or "unverified")
    flags = []
    if _is_degraded(evidence):
        flags.append("degraded")
    if evidence.metadata.get("grounded_fallback"):
        flags.append("grounded_fallback")
    if evidence.metadata.get("conflict_marked"):
        flags.append("potential_conflict")
    flag_text = f" Flags: {', '.join(flags)}." if flags else ""
    return (
        f"- [{evidence.id}] {evidence.title}: {evidence.content} "
        f"Source: {evidence.source_url or 'none'}. "
        f"Citation status: {status}."
        f"{flag_text}"
    )


def _conflict_notice(evidences: list[Evidence]) -> str:
    if any(evidence.metadata.get("conflict_marked") for evidence in evidences):
        return (
            "Some evidence items are marked as potentially conflicting. "
            "Use cautious language and avoid unsupported absolute claims.\n"
        )
    return ""


def _fallback_notice(evidences: list[Evidence]) -> str:
    if any(_is_degraded(evidence) or evidence.metadata.get("grounded_fallback") for evidence in evidences):
        return (
            "Some evidence items are degraded or grounded_fallback. "
            "Use them only as secondary support and mention this in Limitations.\n"
        )
    return ""


def _ensure_evidence_summary_ids(markdown: str, evidences: list[Evidence]) -> str:
    if not evidences:
        return markdown
    summary = _extract_section(markdown, "Evidence Summary")
    if any(f"[{evidence.id}]" in summary for evidence in evidences):
        return markdown
    bullets = "\n".join(
        f"- [{evidence.id}] {evidence.title}. Source: {evidence.source_url or 'none'}; citation_status={evidence.metadata.get('citation_validation_status') or 'unverified'}."
        for evidence in evidences[:6]
    )
    return _prepend_to_section(markdown, "Evidence Summary", bullets)


def _ensure_fallback_limitations(markdown: str, evidences: list[Evidence]) -> str:
    fallback_ids = [
        evidence.id
        for evidence in evidences
        if _is_degraded(evidence) or evidence.metadata.get("grounded_fallback")
    ]
    if not fallback_ids:
        return markdown
    limitation = (
        "Some supporting evidence is degraded or grounded_fallback and should be treated as secondary rather than core support: "
        + ", ".join(fallback_ids)
        + "."
    )
    limitations = _extract_section(markdown, "Limitations")
    if "grounded_fallback" in limitations or "degraded" in limitations:
        return markdown
    return _append_to_section(markdown, "Limitations", limitation)


def _ensure_references(markdown: str, evidences: list[Evidence]) -> str:
    reference_lines = ["## References"]
    if evidences:
        for evidence in evidences:
            source = evidence.source_url or "none"
            if not _is_real_url(evidence.source_url):
                source = f"{source} (degraded / fallback source)"
            status = evidence.metadata.get("citation_validation_status") or "unverified"
            reference_lines.append(f"- [{evidence.id}] {evidence.title} | source_url: {source} | citation_validation_status: {status}")
    else:
        reference_lines.append("- No evidence references were available.")
    references = "\n".join(reference_lines)
    if re.search(r"^\s*##\s+References\b", markdown, flags=re.IGNORECASE | re.MULTILINE):
        return _replace_section(markdown, "References", "\n".join(reference_lines[1:]))
    return markdown.rstrip() + "\n\n" + references + "\n"


def _extract_section(markdown: str, section: str) -> str:
    lines = markdown.splitlines()
    bounds = _find_section_bounds(lines, section)
    if bounds is None:
        return ""
    start, end = bounds
    return "\n".join(lines[start + 1 : end])


def _append_to_section(markdown: str, section: str, text: str) -> str:
    lines = markdown.splitlines()
    bounds = _find_section_bounds(lines, section)
    if bounds is None:
        return markdown.rstrip() + f"\n\n## {section}\n{text}\n"
    _start, end = bounds
    lines.insert(end, text)
    return "\n".join(lines).rstrip() + "\n"


def _prepend_to_section(markdown: str, section: str, text: str) -> str:
    lines = markdown.splitlines()
    bounds = _find_section_bounds(lines, section)
    if bounds is None:
        return markdown.rstrip() + f"\n\n## {section}\n{text}\n"
    start, _end = bounds
    insertion = text.splitlines() + [""]
    return "\n".join(lines[: start + 1] + insertion + lines[start + 1 :]).rstrip() + "\n"


def _replace_section(markdown: str, section: str, content: str) -> str:
    lines = markdown.splitlines()
    bounds = _find_section_bounds(lines, section)
    if bounds is None:
        return markdown.rstrip() + f"\n\n## {section}\n{content}\n"
    start, end = bounds
    return "\n".join(lines[: start + 1] + content.splitlines() + lines[end:]).rstrip() + "\n"


def _find_section_bounds(lines: list[str], section: str) -> tuple[int, int] | None:
    start = None
    heading_pattern = re.compile(r"^\s*#+\s+(.+?)\s*$")
    for index, line in enumerate(lines):
        match = heading_pattern.match(line)
        if not match:
            continue
        heading = match.group(1).strip().lower()
        if start is None and heading == section.lower():
            start = index
            continue
        if start is not None:
            return start, index
    if start is None:
        return None
    return start, len(lines)
