from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from deepresearch_agent.evaluation import compare_run_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two evaluation runs.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--metric", default="judge_overall")
    args = parser.parse_args()
    result = compare_run_summaries(args.baseline, args.treatment, args.metric)
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
