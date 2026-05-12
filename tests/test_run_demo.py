import json
import subprocess
import sys
from pathlib import Path


def test_run_demo_creates_report_and_trace(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs"

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_demo.py"),
            "--question",
            "What are the main challenges and recent methods for long-context LLM evaluation?",
            "--backend",
            "mock",
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    report_path = output_dir / "demo_report.md"
    trace_path = output_dir / "demo_trace.json"

    assert "demo_report.md" in result.stdout
    assert report_path.exists()
    assert trace_path.exists()

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["backend"] == "mock"
    assert len(trace["planned_subtasks"]) >= 3
    assert trace["final_judge_score"]["overall"] == 86
