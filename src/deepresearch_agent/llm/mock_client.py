from __future__ import annotations

import json

from deepresearch_agent.llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Deterministic mock backend for the first-stage MVP."""

    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        """Return a deterministic mock response for planner, researcher, writer, or judge prompts."""
        prompt_type = str(kwargs.get("prompt_type") or "").lower()
        normalized_prompt = prompt.lower()

        if prompt_type == "planner" or "planner" in normalized_prompt:
            return self._planner_response()
        if prompt_type == "researcher" or "researcher" in normalized_prompt:
            return self._researcher_response(prompt)
        if prompt_type == "writer" or "writer" in normalized_prompt:
            return self._writer_response()
        if prompt_type == "judge" or "judge" in normalized_prompt:
            return self._judge_response()
        if prompt_type == "red" or "redagent" in normalized_prompt:
            return self._red_response()
        if prompt_type == "blue" or "blueagent" in normalized_prompt:
            return self._blue_response()

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

    def _researcher_response(self, prompt: str) -> str:
        normalized_prompt = prompt.lower()
        if "identify key challenges" in normalized_prompt:
            evidences = [
                {
                    "title": "Needle-style tests are narrow",
                    "content": "Needle-in-a-haystack tests can measure retrieval but may not fully reflect real-world long-context reasoning.",
                    "source_url": "mock://long-context-eval/challenges/needle",
                    "confidence": 0.88,
                },
                {
                    "title": "Position bias affects long contexts",
                    "content": "Models may perform differently depending on whether relevant evidence appears early, late, or in the middle of a long prompt.",
                    "source_url": "mock://long-context-eval/challenges/position-bias",
                    "confidence": 0.85,
                },
            ]
        elif "review evaluation benchmarks" in normalized_prompt:
            evidences = [
                {
                    "title": "Benchmarks mix synthetic and realistic tasks",
                    "content": "Long-context benchmark suites often combine synthetic retrieval probes with multi-document QA and summarization-style tasks.",
                    "source_url": "mock://long-context-eval/benchmarks/task-mix",
                    "confidence": 0.86,
                },
                {
                    "title": "Benchmark design needs distractors",
                    "content": "Strong evaluations include irrelevant passages and conflicting details to test robustness beyond simple lookup.",
                    "source_url": "mock://long-context-eval/benchmarks/distractors",
                    "confidence": 0.83,
                },
            ]
        elif "compare recent methods" in normalized_prompt:
            evidences = [
                {
                    "title": "Recent methods compare retrieval and synthesis",
                    "content": "Evaluation methods increasingly separate exact retrieval accuracy from cross-document synthesis and attribution quality.",
                    "source_url": "mock://long-context-eval/methods/retrieval-synthesis",
                    "confidence": 0.87,
                },
                {
                    "title": "Stress tests reveal context sensitivity",
                    "content": "Recent protocols vary evidence placement and distractor density to expose brittle long-context behavior.",
                    "source_url": "mock://long-context-eval/methods/stress-tests",
                    "confidence": 0.84,
                },
            ]
        elif "assess limitations" in normalized_prompt:
            evidences = [
                {
                    "title": "Current evaluations can overfit to probes",
                    "content": "Models may perform well on popular diagnostic tasks while still struggling with messy real-world research contexts.",
                    "source_url": "mock://long-context-eval/limitations/probe-overfit",
                    "confidence": 0.82,
                },
                {
                    "title": "Citation quality remains hard to measure",
                    "content": "Long-context reports need evidence-grounded citations, but automatic citation quality metrics remain imperfect.",
                    "source_url": "mock://long-context-eval/limitations/citations",
                    "confidence": 0.8,
                },
            ]
        else:
            evidences = [
                {
                    "title": "Long-context evaluation challenge",
                    "content": "Needle-in-a-haystack tests may not fully reflect real-world long-context reasoning.",
                    "source_url": "mock://long-context-eval/generic",
                    "confidence": 0.88,
                }
            ]
        return json.dumps(
            {
                "evidences": evidences
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

    def _red_response(self) -> str:
        return json.dumps(
            {
                "issues": [],
                "summary": "No high-severity issues found in the deterministic mock report.",
            }
        )

    def _blue_response(self) -> str:
        return json.dumps(
            {
                "revised_report": self._writer_response(),
                "patches": [],
                "summary": "No repair was needed for the deterministic mock report.",
            }
        )
