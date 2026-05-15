from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field


class BenchmarkExample(BaseModel):
    id: str
    domain: str
    question: str
    expected_key_points: list[str]
    reference_answer: str | None = None
    difficulty: str = "medium"
    tags: list[str] = Field(default_factory=list)


class BenchmarkSuite(BaseModel):
    name: str
    examples: list[BenchmarkExample]


def load_benchmark(path: str) -> BenchmarkSuite:
    benchmark_path = Path(path)
    examples: list[BenchmarkExample] = []
    seen_ids: set[str] = set()
    with benchmark_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            data = json.loads(line)
            example = BenchmarkExample.model_validate(data)
            if example.id in seen_ids:
                raise ValueError(f"Duplicate benchmark id at line {line_number}: {example.id}")
            if not example.question.strip():
                raise ValueError(f"Empty benchmark question at line {line_number}: {example.id}")
            seen_ids.add(example.id)
            examples.append(example)
    return BenchmarkSuite(name=benchmark_path.stem, examples=examples)


def filter_benchmark(
    suite: BenchmarkSuite,
    domain: str | None = None,
    difficulty: str | None = None,
    limit: int | None = None,
) -> BenchmarkSuite:
    examples = list(suite.examples)
    if domain:
        examples = [example for example in examples if example.domain == domain]
    if difficulty:
        examples = [example for example in examples if example.difficulty == difficulty]
    if limit is not None:
        examples = examples[:limit]
    return BenchmarkSuite(name=suite.name, examples=examples)
