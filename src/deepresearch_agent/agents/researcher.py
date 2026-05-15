from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.search import BaseSearchProvider, CitationValidator, GroundedResearchBuilder, MockSearchProvider, SearchStats, WebFetcher
from deepresearch_agent.schemas import Evidence, TaskNode
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking
from deepresearch_agent.utils.structured_output import repair_json_with_llm, salvage_json_objects_from_text, try_parse_json_lenient


RESEARCHER_SCHEMA_HINT = """
{
  "evidences": [
    {
      "id": "string",
      "title": "string",
      "content": "string",
      "source_url": "string or null",
      "confidence": 0.0
    }
  ]
}
"""


class ResearcherAgent:
    def __init__(
        self,
        llm: BaseLLM,
        enable_web_search: bool = False,
        search_provider: BaseSearchProvider | None = None,
        fetcher: WebFetcher | None = None,
        citation_validator: CitationValidator | None = None,
        max_search_queries: int = 3,
        search_top_k: int = 3,
    ) -> None:
        self.llm = llm
        self.enable_web_search = enable_web_search
        self.search_provider = search_provider or MockSearchProvider()
        self.fetcher = fetcher or WebFetcher()
        self.citation_validator = citation_validator or CitationValidator()
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k
        self.search_stats = SearchStats(
            enabled=enable_web_search,
            provider=self.search_provider.provider_name,
            api_base_host=_api_base_host(self.search_provider),
            search_timeout=int(getattr(self.search_provider, "timeout", 20)),
            search_provider_mode="mock" if self.search_provider.provider_name == "mock" else "raw_search",
        )

    async def research(self, task: TaskNode) -> list[Evidence]:
        """Generate mock evidence items for a single research task."""
        if self.enable_web_search:
            try:
                builder = GroundedResearchBuilder(
                    search_provider=self.search_provider,
                    fetcher=self.fetcher,
                    validator=self.citation_validator,
                )
                evidences, stats = await builder.build_evidence_for_task(
                    question=str(task.input.get("question") or task.description),
                    task=task,
                    llm=self.llm,
                    max_queries=self.max_search_queries,
                    search_top_k=self.search_top_k,
                )
                self._merge_search_stats(stats)
                if evidences:
                    return evidences
            except Exception:
                self.search_stats.fallback_used = True

        prompt = (
            "You are a researcher agent. Generate concise evidence for this subtask.\n"
            f"Task ID: {task.id}\n"
            f"Task name: {task.name}\n"
            f"Task description: {task.description}\n"
            f"Original question: {task.input.get('question') or task.description}\n"
            "Every evidence item must directly answer both the current task and the original question.\n"
            "Do not output unrelated business reports, market analysis, user surveys, generic enterprise workflows, ordinary machine-learning algorithm tutorials, decision trees, linear regression, or content outside the research question.\n"
            "If the task is about long-context LLM evaluation, focus on benchmarks, metrics, context retention, multi-hop reasoning, lost-in-the-middle effects, faithfulness, latency, cost, scalability, retrieval, synthesis, and citation quality.\n"
            "If no real source_url is available, set source_url to null. Do not invent URLs.\n"
            "If source_url is null, do not include precise quantitative claims, percentages, named deployments, or case-study results unless they appear in the provided task context.\n"
            "Return at most 2 evidence items.\n"
            "Return valid JSON only.\n"
            "Do not output markdown.\n"
            "Do not include markdown fences.\n"
            "Do not include explanations.\n"
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
        parse_stats: dict[str, Any] = {
            "json_parse_success": False,
            "repair_attempted": False,
            "repair_success": False,
            "schema_coercion_success": False,
            "partial_json_salvaged": False,
            "fallback_parse": False,
        }
        data: Any | None = None
        try:
            data = safe_json_loads(cleaned)
            parse_stats["json_parse_success"] = True
        except (KeyError, TypeError, ValueError):
            _write_debug_researcher_output(raw, task.id)
            lenient_data = try_parse_json_lenient(cleaned)
            if lenient_data is not None:
                try:
                    evidences = coerce_evidence_data(lenient_data, task, raw)
                    for evidence in evidences:
                        evidence.metadata.update(parse_stats)
                        evidence.metadata["schema_coercion_success"] = True
                    return evidences
                except ValueError:
                    pass

            salvaged = salvage_json_objects_from_text(raw)
            if salvaged:
                try:
                    evidences = coerce_evidence_data(salvaged, task, raw)
                    for evidence in evidences:
                        evidence.metadata.update(parse_stats)
                        evidence.metadata["schema_coercion_success"] = True
                        evidence.metadata["partial_json_salvaged"] = True
                    return evidences
                except ValueError:
                    pass

            repaired, repair_stats = await repair_json_with_llm(
                self.llm,
                raw_output=raw,
                schema_hint=RESEARCHER_SCHEMA_HINT,
                task_name=f"researcher_output_{task.id}",
            )
            parse_stats["repair_attempted"] = bool(repair_stats["repair_attempted"])
            parse_stats["repair_success"] = bool(repair_stats["repair_success"])
            if repaired is not None:
                data = repaired
            else:
                parse_stats["fallback_parse"] = True
                return _parse_unstructured_evidence(cleaned, task, parse_stats)

        try:
            evidences = coerce_evidence_data(data, task, raw)
            for evidence in evidences:
                evidence.metadata.update(parse_stats)
                evidence.metadata["schema_coercion_success"] = evidence.metadata.get("schema_coercion_success", False) or not parse_stats["json_parse_success"]
            return evidences
        except (KeyError, TypeError, ValueError):
            parse_stats["fallback_parse"] = True
            return _parse_unstructured_evidence(cleaned, task, parse_stats)

    def _merge_search_stats(self, stats: SearchStats) -> None:
        self.search_stats.enabled = self.search_stats.enabled or stats.enabled
        self.search_stats.provider = stats.provider
        self.search_stats.query_count += stats.query_count
        self.search_stats.result_count += stats.result_count
        self.search_stats.fetched_document_count += stats.fetched_document_count
        self.search_stats.validated_citation_count += stats.validated_citation_count
        self.search_stats.supported_count += stats.supported_count
        self.search_stats.partially_supported_count += stats.partially_supported_count
        self.search_stats.unsupported_count += stats.unsupported_count
        self.search_stats.unreachable_count += stats.unreachable_count
        self.search_stats.no_source_count += stats.no_source_count
        self.search_stats.search_queries.extend(stats.search_queries)
        for domain, count in stats.top_domains.items():
            self.search_stats.top_domains[domain] = self.search_stats.top_domains.get(domain, 0) + count
        self.search_stats.fallback_used = self.search_stats.fallback_used or stats.fallback_used
        self.search_stats.provider_errors.extend(stats.provider_errors)
        self.search_stats.api_base_host = stats.api_base_host
        self.search_stats.search_timeout = stats.search_timeout
        self.search_stats.search_provider_mode = stats.search_provider_mode


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


def coerce_evidence_data(data: Any, task: TaskNode, raw_output: str) -> list[Evidence]:
    items: Any = None
    partial_json_salvaged = False
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        for key in ("evidences", "evidence", "results", "findings"):
            value = data.get(key)
            if isinstance(value, list):
                items = value
                break
            if isinstance(value, dict):
                items = [value]
                break
    if items is None:
        salvaged = salvage_json_objects_from_text(raw_output)
        if salvaged:
            items = salvaged
            partial_json_salvaged = True
    if not isinstance(items, list):
        raise ValueError("Researcher returned no evidence items.")

    evidences: list[Evidence] = []
    for index, item in enumerate(items[:3], start=1):
        if not isinstance(item, dict):
            continue
        content_value = item.get("content") or item.get("claim") or item.get("summary") or item.get("text")
        title_value = item.get("title") or item.get("name")
        content = _non_empty_content(content_value or title_value, _fallback_content(raw_output, task))
        title = str(title_value or content[:40] or f"Evidence {index} for {task.name}").strip()
        source_url = _optional_str(item.get("source_url") or item.get("url") or item.get("source"))
        if not _is_real_source_url(source_url):
            content = _sanitize_unverified_content(content)
        confidence = _coerce_confidence(item.get("confidence", item.get("confidence_score", 0.7)))
        if not _is_real_source_url(source_url):
            confidence = min(confidence, 0.5)
        evidences.append(
            Evidence(
                id=_coerce_evidence_id(item.get("id"), task, index),
                task_id=task.id,
                title=title,
                content=content,
                source_url=source_url,
                confidence=confidence,
                metadata={
                    "task_name": task.name,
                    "schema_coercion_success": True,
                    "partial_json_salvaged": partial_json_salvaged,
                    **(_unverified_metadata(source_url) if not _is_real_source_url(source_url) else {}),
                },
            )
        )
    if not evidences:
        raise ValueError("Researcher returned no usable evidence items.")
    return evidences


def _parse_unstructured_evidence(text: str, task: TaskNode, parse_stats: dict[str, Any] | None = None) -> list[Evidence]:
    parse_stats = {
        "json_parse_success": False,
        "repair_attempted": False,
        "repair_success": False,
        "fallback_parse": True,
        **(parse_stats or {}),
    }
    parse_stats["fallback_parse"] = True
    lines = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        for line in text.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line)
    ]
    if not lines:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        fallback_content = _fallback_content(text, task)
        lines = paragraphs or [fallback_content]

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
                confidence=0.45,
                metadata={
                    "task_name": task.name,
                    "model_generated": True,
                    "citation_validation_status": "unverified",
                    "citation_reason": "Unstructured model fallback evidence has no verified external source.",
                    **parse_stats,
                },
            )
        )
    return evidences


def _split_unstructured_item(line: str, task: TaskNode, index: int) -> tuple[str, str]:
    line = line.strip() or task.description or task.name
    if ":" in line:
        title, content = line.split(":", 1)
        return title.strip() or f"Fallback evidence {index} for {task.name}", content.strip() or line
    sentence = line.strip()
    title = sentence[:80].rstrip(".")
    return title or f"Fallback evidence {index} for {task.name}", sentence or task.description or task.name


def _non_empty_content(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _fallback_content(raw_output: str, task: TaskNode) -> str:
    return (
        raw_output.strip()[:500]
        or task.description
        or f"No structured evidence was parsed, but the task concerns: {task.description}"
    )


def _coerce_evidence_id(value: object, task: TaskNode, index: int) -> str:
    raw = str(value or "").strip()
    if raw.startswith(f"{task.id}_"):
        return raw
    suffix = raw or f"evidence_{index}"
    suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", suffix).strip("_") or f"evidence_{index}"
    return f"{task.id}_{suffix}"


def _is_real_source_url(source_url: str | None) -> bool:
    return bool(source_url and source_url.startswith(("http://", "https://")))


def _unverified_metadata(source_url: str | None) -> dict[str, Any]:
    return {
        "model_generated": True,
        "citation_validation_status": "unverified",
        "citation_reason": f"Evidence has no verified external source_url; source={source_url or 'null'}.",
    }


def _sanitize_unverified_content(content: str) -> str:
    text = re.sub(r"\b\d+(?:\.\d+)?\s*%\s+([A-Za-z-]+)", r"an unverified percentage \1", content)
    text = re.sub(r"\b\d+(?:\.\d+)?\s*%", "an unverified percentage", text)
    text = re.sub(r"\b(?:\d+(?:\.\d+)?x)\b", "an unverified multiplier", text, flags=re.IGNORECASE)
    if text != content and "requires citation validation" not in text.lower():
        text = text.rstrip(".") + ". Quantitative details require citation validation."
    return text


def _write_debug_researcher_output(raw: str, task_id: str) -> None:
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_task_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", task_id)
        (output_dir / f"debug_last_researcher_output_{safe_task_id}.txt").write_text(raw, encoding="utf-8")
    except OSError:
        return


def _api_base_host(provider: BaseSearchProvider) -> str | None:
    from urllib.parse import urlparse

    api_base = getattr(provider, "api_base", None)
    if not api_base:
        return None
    return urlparse(str(api_base)).netloc or None
