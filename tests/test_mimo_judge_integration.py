from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepresearch_agent.llm.mock_client import MockLLM
from scripts import run_demo as run_demo_module


class NonJsonJudgeMockLLM(MockLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        if str(kwargs.get("prompt_type") or "").lower() == "judge":
            return "The report is decent. Factuality: 82. Coverage: 80. Reasoning depth: 78. Citation quality: 70. Clarity: 88. Overall: 81."
        return await super().agenerate(prompt, **kwargs)


class BrokenJudgeMockLLM(MockLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        if str(kwargs.get("prompt_type") or "").lower() == "judge":
            return "This is not JSON and contains no numeric rubric."
        return await super().agenerate(prompt, **kwargs)


class MiMoLikeRepairMockLLM(MockLLM):
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        prompt_type = str(kwargs.get("prompt_type") or "").lower()
        if prompt_type == "planner":
            return "I will plan several tasks in prose instead of JSON."
        if prompt_type == "planner_json_repair":
            return """
{
  "subtasks": [
    {"id": "task_1", "name": "Identify challenges", "description": "Identify long-context evaluation challenges.", "agent_type": "researcher", "dependencies": []},
    {"id": "task_2", "name": "Review methods", "description": "Review recent long-context evaluation methods.", "agent_type": "researcher", "dependencies": []}
  ]
}
"""
        if prompt_type == "researcher":
            return "This task has evidence, but the first answer is not JSON."
        if prompt_type.startswith("researcher_output_") and prompt_type.endswith("_json_repair"):
            return """
{
  "evidences": [
    {
      "title": "Repaired long-context evidence",
      "content": "Long-context evaluation should measure retrieval, synthesis, robustness, and citation quality.",
      "source_url": "model://repaired",
      "confidence": 0.76
    }
  ]
}
"""
        if prompt_type == "judge":
            return "The report is acceptable, but no structured rubric is provided."
        if prompt_type == "judge_json_repair":
            return """
{
  "factuality": 82,
  "coverage": 80,
  "reasoning_depth": 78,
  "citation_quality": 72,
  "clarity": 88,
  "overall": 81,
  "comments": "Recovered from MiMo-like unstructured output."
}
"""
        return await super().agenerate(prompt, **kwargs)


def test_run_demo_records_judge_stats_when_text_scores_are_extracted(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_demo_module, "create_llm", lambda *args, **kwargs: NonJsonJudgeMockLLM())

    async def run() -> None:
        trace = await run_demo_module.run_demo(
            question="long context LLM evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
        )

        assert trace["final_judge_score"]["overall"] == 81
        assert trace["judge_stats"]["json_parse_success"] is False
        assert trace["judge_stats"]["extracted_from_text"] is True
        assert trace["judge_stats"]["fallback_used"] is False
        assert (tmp_path / "demo_trace.json").exists()

    asyncio.run(run())


def test_run_demo_records_judge_stats_when_fallback_is_used(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_demo_module, "create_llm", lambda *args, **kwargs: BrokenJudgeMockLLM())

    async def run() -> None:
        trace = await run_demo_module.run_demo(
            question="long context LLM evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
        )

        assert trace["final_judge_score"]["overall"] > 0
        assert trace["judge_stats"]["json_parse_success"] is False
        assert trace["judge_stats"]["extracted_from_text"] is False
        assert trace["judge_stats"]["fallback_used"] is True

    asyncio.run(run())


def test_run_demo_records_structured_repair_stats_for_mimo_like_outputs(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(run_demo_module, "create_llm", lambda *args, **kwargs: MiMoLikeRepairMockLLM())

    async def run() -> None:
        trace = await run_demo_module.run_demo(
            question="long context LLM evaluation",
            backend="mock",
            output_dir=tmp_path,
            mode="dag",
        )

        assert trace["planner_stats"]["repair_attempted"] is True
        assert trace["planner_stats"]["repair_success"] is True
        assert trace["planner_stats"]["fallback_used"] is False
        assert trace["judge_stats"]["repair_attempted"] is True
        assert trace["judge_stats"]["repair_success"] is True
        assert trace["judge_stats"]["fallback_used"] is False
        assert any(
            evidence["metadata"].get("repair_success") is True
            for evidence in trace["collected_evidences"]
        )

    asyncio.run(run())
