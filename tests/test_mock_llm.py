import asyncio
import json

from deepresearch_agent.llm import MockLLM


def test_mock_llm_returns_expected_prompt_types() -> None:
    async def run() -> None:
        llm = MockLLM()

        planner = json.loads(await llm.agenerate("planner prompt", prompt_type="planner"))
        researcher = json.loads(await llm.agenerate("researcher prompt", prompt_type="researcher"))
        writer = await llm.agenerate("writer prompt", prompt_type="writer")
        judge = json.loads(await llm.agenerate("judge prompt", prompt_type="judge"))

        assert len(planner["subtasks"]) >= 3
        assert "evidences" in researcher
        assert "# Long-Context LLM Evaluation" in writer
        assert judge["overall"] == 86

    asyncio.run(run())
