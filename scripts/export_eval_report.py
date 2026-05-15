from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from deepresearch_agent.evaluation.exporter import export_summary_markdown
from deepresearch_agent.evaluation.runner import EvaluationSummary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export summary.json to Markdown.")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    summary_path = Path(args.summary)
    output_path = Path(args.output) if args.output else summary_path.with_suffix(".md")
    summary = EvaluationSummary.model_validate(json.loads(summary_path.read_text(encoding="utf-8")))
    export_summary_markdown(summary, str(output_path))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
