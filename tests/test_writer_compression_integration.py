import json
import subprocess
import sys
from pathlib import Path


def test_run_demo_with_context_compression(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs"

    subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_demo.py"),
            "--question",
            "test long-context evaluation",
            "--backend",
            "mock",
            "--mode",
            "dag",
            "--enable-context-compression",
            "--compression-l1-top-n",
            "6",
            "--compression-l2-top-k",
            "4",
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    trace = json.loads((output_dir / "demo_trace.json").read_text(encoding="utf-8"))
    stats = trace["compression_stats"]

    assert stats["enabled"] is True
    assert stats["final_selected_count"] <= stats["l1_input_count"]
    assert stats["selected_evidence_ids"]
