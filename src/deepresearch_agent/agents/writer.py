from __future__ import annotations

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import Evidence


class WriterAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def write(self, question: str, evidences: list[Evidence]) -> str:
        evidence_lines = "\n".join(
            f"- [{evidence.id}] {evidence.title}: {evidence.content}"
            for evidence in evidences
        )
        prompt = (
            "You are a writer agent. Produce a Markdown research report.\n"
            f"Question: {question}\n"
            f"Evidence:\n{evidence_lines}\n"
            "Include Title, Abstract, Key Findings, Evidence Summary, Limitations, and Conclusion."
        )
        return await self.llm.agenerate(prompt, prompt_type="writer")
