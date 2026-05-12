import pytest

from deepresearch_agent.utils.json_utils import safe_json_loads, strip_thinking


def test_safe_json_loads_parses_standard_json() -> None:
    assert safe_json_loads('{"answer": 42}') == {"answer": 42}


def test_safe_json_loads_parses_fenced_json_block() -> None:
    text = """Here is the JSON:

```json
{"subtasks": [{"id": "task_1"}]}
```
"""
    assert safe_json_loads(text) == {"subtasks": [{"id": "task_1"}]}


def test_safe_json_loads_returns_fallback_on_failure() -> None:
    assert safe_json_loads("not json", fallback={"ok": False}) == {"ok": False}


def test_safe_json_loads_raises_without_fallback() -> None:
    with pytest.raises(ValueError):
        safe_json_loads("not json")


def test_strip_thinking_removes_think_block() -> None:
    text = "<think>hidden reasoning</think>{\"ok\": true}"

    assert strip_thinking(text) == '{"ok": true}'


def test_safe_json_loads_parses_json_after_thinking_block() -> None:
    text = "<think>hidden reasoning</think>\n```json\n{\"ok\": true}\n```"

    assert safe_json_loads(text) == {"ok": True}
