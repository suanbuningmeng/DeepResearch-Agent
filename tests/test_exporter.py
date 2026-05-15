import json
from pathlib import Path

from deepresearch_agent.evaluation.exporter import export_results_csv, export_results_jsonl, export_summary_markdown
from deepresearch_agent.evaluation.judge_metrics import JudgeMetricResult
from deepresearch_agent.evaluation.rule_metrics import RuleMetricResult
from deepresearch_agent.evaluation.runner import EvaluationCaseResult, EvaluationSummary


def result() -> EvaluationCaseResult:
    return EvaluationCaseResult(
        example_id="ex1",
        domain="test",
        question="q",
        report_path="report.md",
        trace_path="trace.json",
        rule_metrics=RuleMetricResult(
            example_id="ex1",
            key_point_coverage=1,
            citation_coverage=1,
            source_quality_score=1,
            hallucination_risk=0,
            report_completeness=1,
            conflict_penalty=0,
            rule_overall=100,
        ),
        judge_metrics=JudgeMetricResult(overall=90),
    )


def test_exporters_write_files(tmp_path: Path) -> None:
    results = [result()]
    summary = EvaluationSummary(
        benchmark_name="bench",
        case_count=1,
        success_count=1,
        error_count=0,
        average_rule_overall=100,
        average_judge_overall=90,
        bootstrap_ci_rule_overall=(100, 100),
        bootstrap_ci_judge_overall=(90, 90),
        domain_breakdown={"test": {"case_count": 1}},
        config={"backend": "mock"},
    )

    export_results_jsonl(results, str(tmp_path / "results.jsonl"))
    export_results_csv(results, str(tmp_path / "results.csv"))
    export_summary_markdown(summary, str(tmp_path / "summary.md"))

    assert json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])["example_id"] == "ex1"
    assert "rule_overall" in (tmp_path / "results.csv").read_text(encoding="utf-8")
    assert "Evaluation Summary" in (tmp_path / "summary.md").read_text(encoding="utf-8")
