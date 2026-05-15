from __future__ import annotations

from typing import Any

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.schemas import TaskNode
from deepresearch_agent.search.schemas import SearchQuery
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


class SearchQueryGenerator:
    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm
        self.last_stats: dict[str, Any] = {"parse_success": False, "fallback_used": False}

    async def generate_queries(
        self,
        question: str,
        task: TaskNode,
        max_queries: int = 3,
    ) -> list[SearchQuery]:
        if self.llm is None:
            self.last_stats = {"parse_success": False, "fallback_used": True}
            return _fallback_queries(question, task, max_queries)
        prompt = (
            "Generate 2-3 specific academic search queries for this research subtask.\n"
            f"Question: {question}\n"
            f"Task: {task.name}\n"
            f"Description: {task.description}\n"
            "For arXiv or paper search, prefer short exact research phrases over broad sentences.\n"
            "For semantic communication with LLMs, include phrases such as semantic communication, task-oriented semantic communication, large language model, and semantic transmission.\n"
            "Avoid unrelated long-context LLM benchmark queries unless the task explicitly asks about long-context evaluation.\n"
            "Return JSON only with schema: {\"queries\":[{\"query\":\"...\",\"intent\":\"...\"}]}."
        )
        raw = await self.llm.agenerate(prompt, prompt_type="search_query")
        try:
            data = safe_json_loads(strip_thinking(raw))
            queries = []
            for index, item in enumerate(data.get("queries", [])[:max_queries], start=1):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("query") or "").strip()
                if not text:
                    continue
                queries.append(
                    SearchQuery(
                        id=f"{task.id}_query_{index}",
                        query=text,
                        task_id=task.id,
                        intent=str(item.get("intent") or "research"),
                    )
                )
            if not queries:
                raise ValueError("No usable search queries.")
            self.last_stats = {"parse_success": True, "fallback_used": False}
            return queries
        except (TypeError, ValueError):
            self.last_stats = {"parse_success": False, "fallback_used": True}
            return _fallback_queries(question, task, max_queries)


def _fallback_queries(question: str, task: TaskNode, max_queries: int) -> list[SearchQuery]:
    if _looks_like_semantic_communication_request(question, task):
        candidates = [
            '"task-oriented semantic communication" "large language model"',
            '"semantic communication" "large language model"',
            '"semantic transmission" "large language model"',
        ]
    else:
        candidates = [
            f"{task.name} {question}",
            f"{task.description} benchmark evidence",
            f"{question} {task.name} recent methods",
        ]
    return [
        SearchQuery(
            id=f"{task.id}_query_{index}",
            query=" ".join(candidate.split())[:240],
            task_id=task.id,
            intent="fallback",
        )
        for index, candidate in enumerate(candidates[:max_queries], start=1)
        if candidate.strip()
    ]


def _looks_like_semantic_communication_request(question: str, task: TaskNode) -> bool:
    text = f"{question} {task.name} {task.description}".lower()
    return (
        "semantic communication" in text
        or "semantic communications" in text
        or "task-oriented communication" in text
        or "task oriented communication" in text
        or "semantic transmission" in text
    )
