from __future__ import annotations

import asyncio
import json

from deepresearch_agent.llm import BaseLLM
from deepresearch_agent.utils.structured_output import repair_json_with_llm


class StaticRepairLLM(BaseLLM):
    def __init__(self, response: str) -> None:
        self.response = response

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        assert "Return one single-line minified JSON object only." in prompt
        assert "Do not output markdown." in prompt
        return self.response


def test_repair_json_with_llm_success() -> None:
    async def run() -> None:
        parsed, stats = await repair_json_with_llm(
            StaticRepairLLM(json.dumps({"ok": True})),
            raw_output="Here is the JSON: ok true",
            schema_hint='{"ok": true}',
            task_name="test",
        )

        assert parsed == {"ok": True}
        assert stats["repair_attempted"] is True
        assert stats["repair_success"] is True
        assert stats["repair_error"] is None

    asyncio.run(run())


def test_repair_json_with_llm_returns_none_on_bad_repair() -> None:
    async def run() -> None:
        parsed, stats = await repair_json_with_llm(
            StaticRepairLLM("still not json"),
            raw_output="bad output",
            schema_hint='{"ok": true}',
            task_name="test",
        )

        assert parsed is None
        assert stats["repair_attempted"] is True
        assert stats["repair_success"] is False
        assert stats["repair_error"]

    asyncio.run(run())
