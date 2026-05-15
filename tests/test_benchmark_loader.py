from pathlib import Path

import pytest

from deepresearch_agent.evaluation.benchmark import filter_benchmark, load_benchmark


def test_load_benchmark_jsonl() -> None:
    suite = load_benchmark("benchmarks/researchbench.jsonl")

    assert suite.name == "researchbench"
    assert len(suite.examples) == 35
    assert suite.examples[0].question


def test_duplicate_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"id":"x","domain":"d","question":"q","expected_key_points":[]}\n'
        '{"id":"x","domain":"d","question":"q2","expected_key_points":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate"):
        load_benchmark(str(path))


def test_empty_question_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"x","domain":"d","question":"","expected_key_points":[]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Empty"):
        load_benchmark(str(path))


def test_filter_benchmark() -> None:
    suite = load_benchmark("benchmarks/researchbench.jsonl")

    filtered = filter_benchmark(suite, domain="rag_system", limit=2)

    assert len(filtered.examples) == 2
    assert all(example.domain == "rag_system" for example in filtered.examples)
