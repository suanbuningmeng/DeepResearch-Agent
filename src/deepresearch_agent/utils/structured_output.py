from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from deepresearch_agent.llm.base import BaseLLM
from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


async def repair_json_with_llm(
    llm: BaseLLM,
    raw_output: str,
    schema_hint: str,
    task_name: str,
    max_tokens: int = 1024,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Ask the configured LLM to convert malformed structured output into valid JSON."""
    stats: dict[str, Any] = {
        "repair_attempted": True,
        "repair_success": False,
        "repair_error": None,
    }
    _write_debug_output(task_name, raw_output)

    prompt = (
        "Convert the following model output into valid JSON matching this schema.\n"
        "Return one single-line minified JSON object only.\n"
        "Do not output markdown.\n"
        "Do not output explanation.\n"
        "Do not output partial JSON.\n"
        "Do not include <think> blocks.\n"
        "If information is missing, fill reasonable defaults.\n"
        "Do not invent missing facts; preserve the original meaning as much as possible.\n"
        f"Schema:\n{schema_hint}\n\n"
        "Model output to repair:\n"
        f"{raw_output[:6000]}"
    )
    try:
        repaired_raw = await llm.agenerate(
            prompt,
            prompt_type=f"{_safe_task_name(task_name)}_json_repair",
            max_tokens=max_tokens,
        )
        parsed = try_parse_json_lenient(repaired_raw)
        if not isinstance(parsed, dict):
            raise ValueError("JSON repair output was not an object.")
    except Exception as exc:
        stats["repair_error"] = _short_error(str(exc))
        return None, stats

    stats["repair_success"] = True
    return parsed, stats


def extract_json_like_text(raw: str) -> str | None:
    """Return the most likely JSON object or array substring from a model response."""
    text = strip_thinking(raw).strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        text = fenced_match.group(1).strip()

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end > object_start:
        return text[object_start : object_end + 1].strip()

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end > array_start:
        return text[array_start : array_end + 1].strip()

    return text or None


def try_parse_json_lenient(raw: str) -> dict[str, Any] | list[Any] | None:
    """Parse JSON from imperfect model output, accepting common near-JSON variants."""
    candidates: list[str] = []
    stripped = strip_thinking(raw).strip()
    if stripped:
        candidates.append(stripped)
    extracted = extract_json_like_text(raw)
    if extracted and extracted not in candidates:
        candidates.append(extracted)

    for candidate in list(candidates):
        repaired = _repair_common_json_issues(candidate)
        if repaired not in candidates:
            candidates.append(repaired)

    for candidate in candidates:
        parsed = _parse_json_value(candidate)
        if isinstance(parsed, (dict, list)):
            return parsed
    return None


def salvage_json_objects_from_text(raw: str) -> list[dict[str, Any]]:
    """Extract complete JSON-like objects from text, useful for truncated arrays."""
    text = strip_thinking(raw)
    objects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for object_text in _iter_balanced_objects_from_each_start(text):
        if object_text in seen:
            continue
        seen.add(object_text)
        parsed = try_parse_json_lenient(object_text)
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def _write_debug_output(task_name: str, raw_output: str) -> None:
    try:
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"debug_last_{_safe_task_name(task_name)}_output.txt"
        (output_dir / filename).write_text(raw_output, encoding="utf-8")
    except OSError:
        return


def _safe_task_name(task_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "_", task_name.strip())
    return normalized.strip("_") or "structured"


def _short_error(error: str | None) -> str | None:
    if not error:
        return None
    return error[:300]


def _parse_json_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        parsed = safe_json_loads(text)
        if isinstance(parsed, dict):
            return parsed
    except ValueError:
        pass
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def _repair_common_json_issues(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r"^```(?:json)?\s*", "", repaired, flags=re.IGNORECASE)
    repaired = re.sub(r"\s*```$", "", repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)
    if _looks_like_single_quoted_json(repaired):
        repaired = repaired.replace("'", '"')
    return repaired


def _looks_like_single_quoted_json(text: str) -> bool:
    if '"' in text:
        return False
    if "'" not in text:
        return False
    return bool(re.search(r"[{,]\s*'[^']+'\s*:", text))


def _iter_balanced_objects_from_each_start(text: str) -> list[str]:
    objects: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        object_text = _balanced_object_from(text, start)
        if object_text:
            objects.append(object_text)
    objects.sort(key=len)
    return objects


def _balanced_object_from(text: str, start: int) -> str | None:
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    in_string = False
    quote_char = ""
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote_char:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            if depth:
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
    return None
