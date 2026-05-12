from __future__ import annotations

import json
import re
from typing import Any


def strip_thinking(text: str) -> str:
    """Remove model thinking blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


def safe_json_loads(text: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parse JSON text, fenced JSON blocks, or the first JSON object in a string."""
    text = strip_thinking(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError("Parsed JSON is not an object.")
    except (json.JSONDecodeError, ValueError):
        pass

    fenced_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        try:
            parsed = json.loads(fenced_match.group(1).strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if object_match:
        try:
            parsed = json.loads(object_match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    if fallback is not None:
        return fallback
    raise ValueError("Unable to parse JSON object from LLM output.")
