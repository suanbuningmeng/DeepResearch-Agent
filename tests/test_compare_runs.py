import json
from pathlib import Path

from deepresearch_agent.evaluation.comparison import compare_run_summaries


def write_results(path: Path, values: list[int]) -> None:
    path.write_text(
        "\n".join(json.dumps({"judge_metrics": {"overall": value}, "rule_metrics": {"rule_overall": value}}) for value in values)
        + "\n",
        encoding="utf-8",
    )


def test_compare_run_summaries_outputs_delta_and_effect_size(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    treatment = tmp_path / "treatment.jsonl"
    write_results(baseline, [80, 82, 84])
    write_results(treatment, [84, 86, 88])

    result = compare_run_summaries(str(baseline), str(treatment), metric_name="judge_overall")

    assert result.mean_delta == 4
    assert result.cohens_d > 0
