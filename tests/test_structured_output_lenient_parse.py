from __future__ import annotations

from deepresearch_agent.utils.structured_output import salvage_json_objects_from_text, try_parse_json_lenient


def test_try_parse_json_lenient_parses_fenced_json() -> None:
    parsed = try_parse_json_lenient("""```json
{"a": 1}
```""")

    assert parsed == {"a": 1}


def test_try_parse_json_lenient_repairs_trailing_comma() -> None:
    parsed = try_parse_json_lenient('{"a": 1, "b": 2,}')

    assert parsed == {"a": 1, "b": 2}


def test_try_parse_json_lenient_handles_python_literals() -> None:
    parsed = try_parse_json_lenient("{'a': None, 'ok': True}")

    assert parsed == {"a": None, "ok": True}


def test_salvage_json_objects_from_truncated_array() -> None:
    raw = """
{
  "evidences": [
    {"title": "A", "content": "Complete object"},
    {"title": "B", "content":
"""

    objects = salvage_json_objects_from_text(raw)

    assert objects == [{"title": "A", "content": "Complete object"}]
