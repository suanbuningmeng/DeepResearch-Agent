import json
import subprocess
import sys
from pathlib import Path


def test_global_timeout_forces_partial_report(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs"

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_demo.py"),
            "--question",
            "What are the main challenges and recent methods for long-context LLM evaluation?",
            "--backend",
            "mock",
            "--mode",
            "dag",
            "--failure-scenario",
            "global_timeout",
            "--global-timeout-seconds",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    trace = json.loads((output_dir / "demo_trace.json").read_text(encoding="utf-8"))
    report = (output_dir / "demo_report.md").read_text(encoding="utf-8")

    assert trace["forced_compose"] is True
    assert trace["cancelled_tasks"]
    assert "CANCELLED" in set(trace["final_task_states"].values())
    assert "Some tasks were not completed due to global timeout." in report
