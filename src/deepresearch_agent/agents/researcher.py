from __future__ import annotations

import re

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import Evidence, TaskNode
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


class ResearcherAgent:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    async def research(self, task: TaskNode) -> list[Evidence]:
        """Generate mock evidence items for a single research task."""
        prompt = (
            "You are a researcher agent. Generate concise mock evidence for this subtask.\n"
            f"Task ID: {task.id}\n"
            f"Task name: {task.name}\n"
            f"Task description: {task.description}\n"
            "Return valid JSON only.\n"
            "Do not include markdown fences.\n"
            "Do not include explanations.\n"
            "Return 2-3 evidence items only.\n"
            "Use this schema exactly:\n"
            "{\n"
            '  "evidences": [\n'
            "    {\n"
            '      "id": "string",\n'
            '      "title": "string",\n'
            '      "content": "string",\n'
            '      "source_url": "string or null",\n'
            '      "confidence": 0.0\n'
            "    }\n"
            "  ]\n"
            "}"
        )
        raw = await self.llm.agenerate(prompt, prompt_type="researcher")
        cleaned = strip_thinking(raw)
        try:
            data = safe_json_loads(cleaned)
        except ValueError:
            return _parse_unstructured_evidence(cleaned, task)

        return [
            Evidence(
                id=f"{task.id}_evidence_{index}",
                task_id=task.id,
                title=str(item.get("title") or f"Evidence {index} for {task.name}"),
                content=str(item.get("content") or item.get("summary") or ""),
                source_url=_optional_str(item.get("source_url") or item.get("url")),
                confidence=_coerce_confidence(item.get("confidence", 0.7)),
                metadata={"task_name": task.name},
            )
            for index, item in enumerate(data["evidences"], start=1)
        ]


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_confidence(value: object) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.7
    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))


def _parse_unstructured_evidence(text: str, task: TaskNode) -> list[Evidence]:
    lines = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        for line in text.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line)
    ]
    if not lines:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        lines = paragraphs or [text.strip()]

    evidences: list[Evidence] = []
    for index, line in enumerate(lines[:3], start=1):
        title, content = _split_unstructured_item(line, task, index)
        evidences.append(
            Evidence(
                id=f"{task.id}_fallback_evidence_{index}",
                task_id=task.id,
                title=title,
                content=content,
                source_url="model://unstructured-output",
                confidence=0.55,
                metadata={"task_name": task.name, "fallback_parse": True},
            )
        )
    return evidences


def _split_unstructured_item(line: str, task: TaskNode, index: int) -> tuple[str, str]:
    if ":" in line:
        title, content = line.split(":", 1)
        return title.strip() or f"Fallback evidence {index} for {task.name}", content.strip() or line
    sentence = line.strip()
    title = sentence[:80].rstrip(".")
    return title or f"Fallback evidence {index} for {task.name}", sentence
