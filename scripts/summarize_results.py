from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print key metrics from summary.json or results.jsonl.")
    parser.add_argument("--path", required=True)
    args = parser.parse_args()
    path = Path(args.path)
    if path.name.endswith(".jsonl"):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        judge = [(row.get("judge_metrics") or {}).get("overall") for row in rows]
        rule = [(row.get("rule_metrics") or {}).get("rule_overall") for row in rows]
        print(f"cases={len(rows)} rule_avg={_avg(rule)} judge_avg={_avg(judge)}")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps({
            "benchmark_name": data.get("benchmark_name"),
            "case_count": data.get("case_count"),
            "success_count": data.get("success_count"),
            "error_count": data.get("error_count"),
            "average_rule_overall": data.get("average_rule_overall"),
            "average_judge_overall": data.get("average_judge_overall"),
        }, indent=2))


def _avg(values):
    clean = [float(value) for value in values if value is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


if __name__ == "__main__":
    main()
