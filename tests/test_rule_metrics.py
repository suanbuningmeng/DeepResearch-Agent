from deepresearch_agent.evaluation.benchmark import BenchmarkExample
from deepresearch_agent.evaluation.rule_metrics import compute_rule_metrics


def example() -> BenchmarkExample:
    return BenchmarkExample(
        id="ex1",
        domain="test",
        question="q",
        expected_key_points=["context retention", "citation quality"],
    )


def trace() -> dict:
    return {
        "memory_stats": {
            "source_quality_summary": {
                "real_url": 1,
                "mock_source": 1,
                "example_url": 0,
                "null_source": 0,
                "model_generated": 0,
                "unknown": 0,
            }
        },
        "report_quality_check": {"is_complete": True},
        "conflict_stats": {"enabled": True, "conflict_count": 1, "dropped_evidence_ids": [], "marked_conflict_evidence_ids": ["e1"]},
        "compression_stats": {"compression_ratio": 0.5},
        "red_blue_stats": {"score_delta": 2},
    }


def test_rule_metrics_cover_expected_points_and_citations() -> None:
    report = "## Evidence Summary\n- [task_1_evidence_1] Context retention affects citation quality."

    metrics = compute_rule_metrics(example(), report, trace())

    assert metrics.key_point_coverage == 1.0
    assert metrics.citation_coverage == 1.0
    assert 0.0 <= metrics.rule_overall <= 100.0


def test_rule_metrics_source_quality_and_hallucination_proxy() -> None:
    report = "## Evidence Summary\n- This always proves all systems are safe."

    metrics = compute_rule_metrics(example(), report, trace())

    assert metrics.source_quality_score == 0.7
    assert metrics.hallucination_risk > 0
    assert metrics.conflict_penalty > 0
