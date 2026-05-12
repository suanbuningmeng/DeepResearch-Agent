from __future__ import annotations

import json
import re
from typing import Any


def strip_thinking(text: str) -> str:
    """Remove model thinking blocks and common JSON fences from text."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    stripped = text.strip()
    fenced_match = re.fullmatch(
        r"```(?:json)?\s*(.*?)```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_match:
        return fenced_match.group(1).strip()
    return stripped


def safe_json_loads(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse JSON text, fenced JSON blocks, or the first JSON object in a string."""
    text = strip_thinking(text)
    parsed = _try_parse_dict(text)
    if parsed is not None:
        return parsed

    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        parsed = _try_parse_dict(fenced_match.group(1).strip())
        if parsed is not None:
            return parsed

    object_text = _extract_first_json_object(text)
    if object_text:
        parsed = _try_parse_dict(object_text)
        if parsed is not None:
            return parsed

    if fallback is not None:
        return fallback
    raise ValueError("Unable to parse JSON object from LLM output.")


def _try_parse_dict(text: str) -> dict[str, Any] | None:
    candidates = [text, _remove_trailing_commas(text)]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None
