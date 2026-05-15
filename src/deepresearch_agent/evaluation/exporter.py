from __future__ import annotations

import csv
import json
from pathlib import Path


def export_results_jsonl(results: list, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            data = result.model_dump(mode="json") if hasattr(result, "model_dump") else dict(result)
            file.write(json.dumps(data, ensure_ascii=False) + "\n")


def export_results_csv(results: list, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "example_id",
        "domain",
        "question",
        "rule_overall",
        "judge_overall",
        "error",
        "report_path",
        "trace_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            rule_metrics = getattr(result, "rule_metrics", None)
            judge_metrics = getattr(result, "judge_metrics", None)
            writer.writerow(
                {
                    "example_id": result.example_id,
                    "domain": result.domain,
                    "question": result.question,
                    "rule_overall": getattr(rule_metrics, "rule_overall", None),
                    "judge_overall": getattr(judge_metrics, "overall", None),
                    "error": result.error,
                    "report_path": result.report_path,
                    "trace_path": result.trace_path,
                }
            )


def export_summary_markdown(summary, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_cases = []
    markdown = [
        f"# Evaluation Summary: {summary.benchmark_name}",
        "",
        "## Config",
        "```json",
        json.dumps(summary.config, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Results",
        f"- Case count: {summary.case_count}",
        f"- Success count: {summary.success_count}",
        f"- Error count: {summary.error_count}",
        f"- Average rule overall: {summary.average_rule_overall}",
        f"- Average judge overall: {summary.average_judge_overall}",
        f"- Bootstrap 95% CI rule overall: {summary.bootstrap_ci_rule_overall}",
        f"- Bootstrap 95% CI judge overall: {summary.bootstrap_ci_judge_overall}",
        "",
        "## Domain Breakdown",
    ]
    for domain, item in summary.domain_breakdown.items():
        markdown.append(
            f"- {domain}: cases={item.get('case_count')}, success={item.get('success_count')}, "
            f"avg_rule={item.get('average_rule_overall')}"
        )
    markdown.extend(
        [
            "",
            "## Top Failed Cases",
            "\n".join(failed_cases) if failed_cases else "No failed cases listed in summary.",
            "",
            "## Notes",
            "Rule metrics are proxy metrics based on benchmark key points, local trace fields, and report structure. They are not external factual verification.",
        ]
    )
    output_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
