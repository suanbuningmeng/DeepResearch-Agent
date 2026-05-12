import json
import subprocess
import sys
from pathlib import Path


def test_run_demo_with_mock_and_sqlite_memory(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "outputs"
    db_path = tmp_path / "memory.sqlite"
    index_path = tmp_path / "vector_index.npz"

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
            "--memory-backend",
            "sqlite",
            "--memory-db-path",
            str(db_path),
            "--vector-index-path",
            str(index_path),
            "--clear-memory",
            "--output-dir",
            str(output_dir),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    trace = json.loads((output_dir / "demo_trace.json").read_text(encoding="utf-8"))

    assert trace["memory_stats"]["backend"] == "sqlite"
    assert trace["memory_stats"]["inserted_evidence_count"] > 0
    assert trace["memory_stats"]["total_evidence_count"] > 0
