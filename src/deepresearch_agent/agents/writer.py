from __future__ import annotations

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
        """Create a Markdown research report from a question and collected evidence."""
        selected_evidences = select_top_evidences_per_task(evidences, top_k_per_task)
        evidence_lines = "\n".join(
            f"- [{evidence.id}] {evidence.title}: {evidence.content}"
            for evidence in selected_evidences
        )
        prompt = (
            "You are a writer agent. Produce a Markdown research report.\n"
            f"Question: {question}\n"
            f"Evidence:\n{evidence_lines}\n"
            "Keep the report concise.\n"
            "Use at most 6 evidence bullets.\n"
            "Each evidence bullet must be a complete sentence.\n"
            "Do not start a bullet unless you can finish it.\n"
            "Do not leave unfinished bullets.\n"
            "Finish all required sections.\n"
            "The report must include Abstract, Key Findings, Evidence Summary, Limitations, and Conclusion.\n"
            "The report must end with a complete Conclusion section."
        )
        markdown = await self.llm.agenerate(prompt, prompt_type="writer")
        markdown, quality = ensure_report_completeness(markdown)
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
        selected.extend(
            sorted(
                grouped[task_id],
                key=lambda evidence: evidence.confidence,
                reverse=True,
            )[:top_k_per_task]
        )
    return selected
