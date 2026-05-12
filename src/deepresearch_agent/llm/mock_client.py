from __future__ import annotations

import json

from deepresearch_agent.llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Deterministic mock backend for the first-stage MVP."""

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        prompt_type = str(kwargs.get("prompt_type") or "").lower()
        normalized_prompt = prompt.lower()

        if prompt_type == "planner" or "planner" in normalized_prompt:
            return self._planner_response()
        if prompt_type == "researcher" or "researcher" in normalized_prompt:
            return self._researcher_response()
        if prompt_type == "writer" or "writer" in normalized_prompt:
            return self._writer_response()
        if prompt_type == "judge" or "judge" in normalized_prompt:
            return self._judge_response()

        return "MockLLM received an unknown prompt type."

    def _planner_response(self) -> str:
        return json.dumps(
            {
                "subtasks": [
                    {
                        "id": "task_1",
                        "name": "Identify key challenges",
                        "description": "Analyze the main challenges of long-context LLM evaluation.",
                    },
                    {
                        "id": "task_2",
                        "name": "Review evaluation benchmarks",
                        "description": "Summarize benchmark designs used for long-context model evaluation.",
                    },
                    {
                        "id": "task_3",
                        "name": "Compare recent methods",
                        "description": "Identify recent methods for measuring retrieval, reasoning, and robustness.",
                    },
                    {
                        "id": "task_4",
                        "name": "Assess limitations",
                        "description": "Describe gaps in current long-context evaluation practices.",
                    },
                ]
            }
        )

    def _researcher_response(self) -> str:
        return json.dumps(
            {
                "evidences": [
                    {
                        "title": "Needle-style tests are narrow",
                        "content": "Needle-in-a-haystack tests can measure retrieval but may not fully reflect real-world long-context reasoning.",
                        "source_url": "mock://long-context-eval/needle",
                        "confidence": 0.88,
                    },
                    {
                        "title": "Long-context reasoning needs synthesis",
                        "content": "Realistic evaluation often requires combining scattered evidence, tracking conflicts, and avoiding position bias.",
                        "source_url": "mock://long-context-eval/synthesis",
                        "confidence": 0.84,
                    },
                ]
            }
        )

    def _writer_response(self) -> str:
        return """# Long-Context LLM Evaluation

## Abstract
This mock report summarizes core challenges and recent methods for evaluating long-context language models.

## Key Findings
- Long-context evaluation must test more than retrieval; it should also measure synthesis, reasoning, and robustness.
- Needle-in-a-haystack tasks are useful diagnostics but can overstate practical long-context competence.
- Recent evaluation approaches combine synthetic probes, realistic multi-document QA, position-sensitivity tests, and citation checks.

## Evidence Summary
The collected mock evidence suggests that robust evaluation should include tasks where evidence is distributed across long inputs and where distractors or conflicting details are present.

## Limitations
This first-stage report is generated from deterministic mock evidence rather than live papers, search results, or benchmark data.

## Conclusion
Long-context LLM evaluation is moving toward broader task suites that measure retrieval, cross-document synthesis, faithfulness, and stability across context positions.
"""

    def _judge_response(self) -> str:
        return json.dumps(
            {
                "factuality": 88,
                "coverage": 85,
                "reasoning_depth": 86,
                "citation_quality": 82,
                "clarity": 90,
                "overall": 86,
                "comments": "The report is coherent but uses mock evidence.",
            }
        )
