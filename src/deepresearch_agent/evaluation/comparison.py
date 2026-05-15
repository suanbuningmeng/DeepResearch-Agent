from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from deepresearch_agent.evaluation.statistics import cohens_d


class RunComparisonResult(BaseModel):
    baseline_path: str
    treatment_path: str
    metric_name: str
    baseline_mean: float
    treatment_mean: float
    mean_delta: float
    cohens_d: float


def compare_run_summaries(
    baseline_results_jsonl: str,
    treatment_results_jsonl: str,
    metric_name: str = "judge_overall",
) -> RunComparisonResult:
    baseline = _read_metric_values(baseline_results_jsonl, metric_name)
    treatment = _read_metric_values(treatment_results_jsonl, metric_name)
    baseline_mean = _mean(baseline)
    treatment_mean = _mean(treatment)
    return RunComparisonResult(
        baseline_path=baseline_results_jsonl,
        treatment_path=treatment_results_jsonl,
        metric_name=metric_name,
        baseline_mean=baseline_mean,
        treatment_mean=treatment_mean,
        mean_delta=round(treatment_mean - baseline_mean, 6),
        cohens_d=cohens_d(baseline, treatment),
    )


def _read_metric_values(path: str, metric_name: str) -> list[float]:
    values: list[float] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            value = _extract_metric(data, metric_name)
            if value is not None:
                values.append(float(value))
    return values


def _extract_metric(data: dict, metric_name: str):
    if metric_name in {"judge_overall", "overall"}:
        return (data.get("judge_metrics") or {}).get("overall")
    if metric_name == "rule_overall":
        return (data.get("rule_metrics") or {}).get("rule_overall")
    if metric_name.startswith("rule_"):
        return (data.get("rule_metrics") or {}).get(metric_name.removeprefix("rule_"))
    if metric_name.startswith("judge_"):
        return (data.get("judge_metrics") or {}).get(metric_name.removeprefix("judge_"))
    return data.get(metric_name)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)
